import itertools
import os
import pickle
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize

from utils import (
    FrameScaleContext,
    PitchColorModel,
    get_center_of_bbox,
    get_foot_position,
    get_video_properties,
    iter_video_batches,
)
from .pitch_model import (
    ALL_PROJECTIVE_INTERSECTIONS,
    LONGITUDINAL_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TRANSVERSE_X,
    build_fifa_template_points,
    get_screen_ref_quad,
)


class HomographyEstimator:
    """
    Camera- and side-agnostic dynamic field registration engine.
    - Uses learned CIE Lab PitchColorModel (zero hardcoded HSV/RGB).
    - Scales all morphological, Hough, gating, and scoring parameters proportionally to frame geometry.
    - Separates longitudinal (Y=const) and transverse (X=const) lines via vanishing-point geometry
      so Left-Half, Midfield (Center Circle), and Right-Half camera angles work symmetrically.
    - Validates and smooths homographies via screen-interior quad projection onto SL(3).
    """

    TARGET_WORK_W = 960

    def __init__(self) -> None:
        self.template_pts = build_fifa_template_points()

    @staticmethod
    def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply a 3x3 homography matrix H to an (N, 2) array of 2D coordinates."""
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts_arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts_arr, H.astype(np.float64))
        return projected.reshape(-1, 2)

    @staticmethod
    def intersect_lines(L1: tuple, L2: tuple) -> tuple[float, float] | None:
        """Compute intersection (x, y) of two homogeneous lines a*x + b*y + c = 0."""
        a1, b1, c1 = L1[:3]
        a2, b2, c2 = L2[:3]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-7:
            return None
        return ((b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det)

    @classmethod
    def _separate_line_families_vp(
        cls,
        segs: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> tuple[list[tuple], list[tuple]]:
        """
        Separate Hough segments into Longitudinal (Y=const) and Transverse (X=const) families
        using dominant horizontal direction / vanishing-point consensus (camera-pan sign agnostic!).
        """
        min_seg_len = 0.028 * frame_w
        candidates: list[tuple[float, float, float, float, float, float]] = []

        for x1, y1, x2, y2 in segs:
            if x2 < x1 or (abs(x2 - x1) < 1e-4 and y2 < y1):
                x1, y1, x2, y2 = x2, y2, x1, y1
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_seg_len:
                continue
            ang_deg = float(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))  # in [-90, +90]
            candidates.append((float(x1), float(y1), float(x2), float(y2), length, ang_deg))

        if not candidates:
            return [], []

        # Step 1: Estimate dominant longitudinal angle mode among near-horizontal segments (|ang| <= 15.5 deg).
        # By 3D sideline perspective geometry, all Y=const lines (touchlines, box side lines) lie within +-15.5 deg
        # across Left, Midfield, and Right camera pans, never colliding with +-20 deg transverse box lines!
        gentle = [c for c in candidates if abs(c[5]) <= 15.5]
        long_segs: list[tuple[float, float, float, float, float]] = []
        trans_segs: list[tuple[float, float, float, float, float]] = []

        if gentle:
            best_mode_ang = 0.0
            best_weight = -1.0
            test_angles = np.linspace(-15.0, 15.0, 61)
            for ta in test_angles:
                w_sum = sum(c[4] for c in gentle if abs(c[5] - ta) <= 3.5)
                if w_sum > best_weight:
                    best_weight = w_sum
                    best_mode_ang = float(ta)

            inlier_gentle = [c for c in gentle if abs(c[5] - best_mode_ang) <= 5.5]
            vp_long: tuple[float, float] | None = None
            if len(inlier_gentle) >= 2:
                vp_candidates = []
                for s_a, s_b in itertools.combinations(inlier_gentle[:10], 2):
                    ym_a = 0.5 * (s_a[1] + s_a[3])
                    ym_b = 0.5 * (s_b[1] + s_b[3])
                    if abs(ym_a - ym_b) < 0.06 * frame_h:
                        continue
                    la = (-(s_a[3] - s_a[1]), s_a[2] - s_a[0], -(s_a[0] * (-(s_a[3] - s_a[1])) + s_a[1] * (s_a[2] - s_a[0])))
                    lb = (-(s_b[3] - s_b[1]), s_b[2] - s_b[0], -(s_b[0] * (-(s_b[3] - s_b[1])) + s_b[1] * (s_b[2] - s_b[0])))
                    pt = cls.intersect_lines(la, lb)
                    if pt is not None and abs(pt[0] - 0.5 * frame_w) > 0.9 * frame_w:
                        vp_candidates.append((pt[0], pt[1], s_a[4] + s_b[4]))
                if vp_candidates:
                    best_vp = max(vp_candidates, key=lambda item: item[2])
                    vp_long = (best_vp[0], best_vp[1])

            for x1, y1, x2, y2, length, ang_deg in candidates:
                mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                if vp_long is not None:
                    rx = vp_long[0] - mx
                    ry = vp_long[1] - my
                    if rx < 0:
                        rx, ry = -rx, -ry
                    expected_long_ang = float(np.rad2deg(np.arctan2(ry, max(1e-6, rx))))
                    diff_long = abs(ang_deg - expected_long_ang)
                else:
                    diff_long = abs(ang_deg - best_mode_ang)

                if diff_long <= 6.0 and abs(ang_deg) <= 17.5:
                    long_segs.append((x1, y1, x2, y2, length))
                elif diff_long >= 9.5:
                    trans_segs.append((x1, y1, x2, y2, length))
        else:
            for x1, y1, x2, y2, length, ang_deg in candidates:
                if abs(ang_deg) <= 18.0:
                    long_segs.append((x1, y1, x2, y2, length))
                elif abs(ang_deg) >= 24.0:
                    trans_segs.append((x1, y1, x2, y2, length))

        return long_segs, trans_segs

    @classmethod
    def extract_lines_and_score_map(
        cls,
        frame: np.ndarray,
        person_bboxes: list[list[float]],
        pitch_model: PitchColorModel,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple], list[tuple], float]:
        """Extract dynamic Lab turf hull, adaptive White Top-Hat + Canny edges, and VP-clustered Hough lines."""
        fh, fw = frame.shape[:2]
        scale = min(1.0, cls.TARGET_WORK_W / float(max(1, fw)))
        sw = max(160, int(round(fw * scale)))
        sh = max(90, int(round(fh * scale)))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        # 1. Dynamic Lab Playable-Pitch Convex Hull Mask (0.004 trim excludes ad-board edge while keeping far touchline)
        pitch_mask = pitch_model.compute_playable_pitch_mask(frame, work_w=sw, erode_border_frac=0.004)

        # Mask out tracked players, goalkeepers, and referees proportionally to bbox size
        clean_mask = pitch_mask.copy()
        pad_px = max(2, int(round(0.004 * sw)))
        for bx1, by1, bx2, by2 in person_bboxes:
            x1 = max(0, int(round(bx1 * scale)) - pad_px)
            y1 = max(0, int(round(by1 * scale)) - pad_px)
            x2 = min(sw, int(round(bx2 * scale)) + pad_px)
            y2 = min(sh, int(round(by2 * scale)) + pad_px)
            clean_mask[y1:y2, x1:x2] = 0

        # 2. Relative-Scale Morphological White Top-Hat + Adaptive Percentile Thresholding
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        k_top = max(5, int(round(0.010 * sw)) | 1)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k_top, k_top)))
        tophat_masked = cv2.bitwise_and(tophat, tophat, mask=clean_mask)

        valid_vals = tophat_masked[clean_mask > 0]
        if len(valid_vals) > 200:
            adaptive_thr = float(np.clip(np.percentile(valid_vals, 96.6), 7.5, 42.0))
        else:
            adaptive_thr = 14.0

        _, white_bin = cv2.threshold(tophat_masked, adaptive_thr, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(white_bin, 50, 150)

        sigma_blur = max(1.5, 0.0032 * sw)
        blurred = cv2.GaussianBlur(white_bin.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma_blur)
        # Penalize projected lines falling on empty green turf (-0.32) so fake penalty boxes never win on midfield shots!
        line_score_map = np.clip(blurred * 2.8, 0.0, 1.0) - 0.32

        # 3. Relative-Scale Probabilistic Hough Transform & Vanishing-Point Family Clustering
        hough_thresh = max(16, int(round(0.028 * sw)))
        min_line_len = max(16, int(round(0.030 * sw)))
        max_line_gap = max(10, int(round(0.022 * sw)))
        raw_segs = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=hough_thresh,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )

        if raw_segs is not None:
            segs_full = raw_segs.reshape(-1, 4).astype(np.float64) / scale
            long_segs, trans_segs = cls._separate_line_families_vp(segs_full, fw, fh)
        else:
            long_segs, trans_segs = [], []

        merge_dist_tol = 0.008 * fw
        min_strong_len = 0.044 * fw
        sample_step = max(8.0, 0.010 * fw)

        def merge_family(seg_list: list[tuple], is_long: bool) -> list[tuple]:
            if not seg_list:
                return []
            seg_list = sorted(seg_list, key=lambda s: s[4], reverse=True)
            clusters: list[dict[str, Any]] = []
            for seg in seg_list:
                x1, y1, x2, y2, length = seg
                ang = np.arctan2(y2 - y1, x2 - x1) % np.pi
                placed = False
                for cl in clusters:
                    a, b, c, cl_ang, _ = cl["line"]
                    d1 = abs(a * x1 + b * y1 + c)
                    d2 = abs(a * x2 + b * y2 + c)
                    ang_diff = min(abs(ang - cl_ang), np.pi - abs(ang - cl_ang))
                    if d1 < merge_dist_tol and d2 < merge_dist_tol and ang_diff < np.deg2rad(3.8):
                        cl["segs"].append(seg)
                        pts = []
                        for sx1, sy1, sx2, sy2, slen in cl["segs"]:
                            for t_val in np.linspace(0.0, 1.0, max(2, int(slen // sample_step))):
                                pts.append([sx1 + t_val * (sx2 - sx1), sy1 + t_val * (sy2 - sy1)])
                        vx, vy, x0, y0 = cv2.fitLine(
                            np.asarray(pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01
                        ).ravel()
                        na, nb = -float(vy), float(vx)
                        nc = -(na * float(x0) + nb * float(y0))
                        nang = np.arctan2(float(vy), float(vx)) % np.pi
                        cl["line"] = (na, nb, nc, nang, sum(s[4] for s in cl["segs"]))
                        placed = True
                        break
                if not placed:
                    vx, vy = (x2 - x1) / length, (y2 - y1) / length
                    na, nb = -vy, vx
                    nc = -(na * x1 + nb * y1)
                    clusters.append({"segs": [seg], "line": (na, nb, nc, ang, length)})

            strong = [cl["line"] for cl in clusters if cl["line"][4] >= min_strong_len]
            if is_long:
                x_anchor = 0.50 * fw
                strong.sort(key=lambda L: -(L[0] * x_anchor + L[2]) / (L[1] if abs(L[1]) > 1e-6 else 1e-6))
            else:
                y_anchor = 0.42 * fh
                strong.sort(key=lambda L: -(L[1] * y_anchor + L[2]) / (L[0] if abs(L[0]) > 1e-6 else 1e-6))
            return strong[:9]

        return (
            line_score_map,
            pitch_mask,
            white_bin,
            merge_family(long_segs, True),
            merge_family(trans_segs, False),
            scale,
        )

    @staticmethod
    def is_plausible_broadcast_homography(H_p2i: np.ndarray, frame_w: int, frame_h: int) -> bool:
        """
        Symmetric Left/Midfield/Right camera plausibility check:
        Projects the 4 visible screen-interior points onto the 2D metric pitch via inv(H_p2i)
        and verifies convexity, camera orientation handedness, and physical FOV dimensions.
        """
        try:
            H_i2p = np.linalg.inv(H_p2i)
        except np.linalg.LinAlgError:
            return False

        scr = get_screen_ref_quad(frame_w, frame_h).astype(np.float64)
        hom = np.hstack([scr, np.ones((4, 1), dtype=np.float64)]) @ H_i2p.T
        if np.any(hom[:, 2] <= 1e-4):
            return False

        pts_m = hom[:, :2] / hom[:, 2:3]
        p_tl, p_tr, p_bl, p_br = pts_m

        # Screen left-to-right must progress positively along +X_m; top-to-bottom along +Y_m
        if p_tr[0] <= p_tl[0] + 5.0 or p_br[0] <= p_bl[0] + 4.0:
            return False
        if p_bl[1] <= p_tl[1] + 4.0 or p_br[1] <= p_tr[1] + 4.0:
            return False

        # Projected screen interior must sit within realistic stadium apron
        if np.any(pts_m[:, 0] < -35.0) or np.any(pts_m[:, 0] > PITCH_LENGTH + 35.0):
            return False
        if np.any(pts_m[:, 1] < -25.0) or np.any(pts_m[:, 1] > PITCH_WIDTH + 25.0):
            return False

        top_width_m = float(np.hypot(*(p_tr - p_tl)))
        bot_width_m = float(np.hypot(*(p_br - p_bl)))
        left_depth_m = float(np.hypot(*(p_bl - p_tl)))
        right_depth_m = float(np.hypot(*(p_br - p_tr)))

        if not (8.0 <= top_width_m <= 115.0 and 6.0 <= bot_width_m <= 95.0):
            return False
        # Far horizon span in broadcast perspective is moderately wider than near foreground span (blocks horizon pinching!)
        if top_width_m < 0.85 * bot_width_m or top_width_m > 3.15 * bot_width_m:
            return False
        if not (6.0 <= left_depth_m <= 75.0 and 6.0 <= right_depth_m <= 75.0):
            return False

        quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)
        return bool(cv2.isContourConvex(quad))

    def score_homography(
        self,
        H_p2i: np.ndarray,
        score_map: np.ndarray,
        pitch_mask: np.ndarray,
        scale: float,
        frame_w: int,
        frame_h: int,
    ) -> float:
        """Score a pitch-to-image homography by relative spatial grid-cell white-line coverage."""
        if not self.is_plausible_broadcast_homography(H_p2i, frame_w, frame_h):
            return -1e6

        sh, sw = score_map.shape[:2]
        ones = np.ones((len(self.template_pts), 1), dtype=np.float64)
        hom = np.hstack([self.template_pts, ones]) @ H_p2i.T
        z = hom[:, 2]
        valid_z = z > 1e-3
        if int(np.sum(valid_z)) < 65:
            return -1e6

        ux = (hom[:, 0] / np.where(valid_z, z, 1.0)) * scale
        uy = (hom[:, 1] / np.where(valid_z, z, 1.0)) * scale
        in_bounds = valid_z & (ux >= 0) & (ux < sw - 1) & (uy >= 0) & (uy < sh - 1)
        if int(np.sum(in_bounds)) < 65:
            return -1e6

        ix = np.round(ux[in_bounds]).astype(np.int32)
        iy = np.round(uy[in_bounds]).astype(np.int32)
        on_grass = pitch_mask[iy, ix] > 0
        if int(np.sum(on_grass)) < 55:
            return -1e6

        cell_px = max(6, int(round(0.0125 * sw)))
        gx, gy = ix[on_grass], iy[on_grass]
        cell_ids = (gy // cell_px) * 200 + (gx // cell_px)
        unique_cells, cell_inv = np.unique(cell_ids, return_inverse=True)
        if len(unique_cells) < 28:
            return -1e6

        vals = score_map[gy, gx]
        cell_max = np.full(len(unique_cells), -1.0, dtype=np.float32)
        np.maximum.at(cell_max, cell_inv, vals)
        return float(np.sum(cell_max))

    def refine_homography_local(
        self,
        H_init: np.ndarray,
        score_map: np.ndarray,
        pitch_mask: np.ndarray,
        scale: float,
        frame_w: int,
        frame_h: int,
        maxiter: int = 120,
    ) -> tuple[np.ndarray, float]:
        """Sub-pixel Powell refinement of H_p2i with perturbations scaled to frame dimensions."""
        H0 = H_init / (H_init[2, 2] if abs(H_init[2, 2]) > 1e-9 else 1.0)
        tx_scale = 0.0052 * frame_w
        ty_scale = 0.0092 * frame_h
        persp_scale = 0.075 / max(1.0, float(frame_w))

        def build_perturbed(d: np.ndarray) -> np.ndarray:
            D = np.eye(3, dtype=np.float64)
            D[0, 0] += d[0] * 0.012
            D[0, 1] += d[1] * 0.012
            D[0, 2] += d[2] * tx_scale
            D[1, 0] += d[3] * 0.012
            D[1, 1] += d[4] * 0.012
            D[1, 2] += d[5] * ty_scale
            D[2, 0] += d[6] * persp_scale
            D[2, 1] += d[7] * persp_scale
            return D @ H0

        res = minimize(
            lambda d: -self.score_homography(build_perturbed(d), score_map, pitch_mask, scale, frame_w, frame_h),
            np.zeros(8, dtype=np.float64),
            method="Powell",
            options={"maxiter": maxiter, "ftol": 2e-3},
        )
        H_opt = build_perturbed(res.x)
        H_opt /= H_opt[2, 2]
        return H_opt, float(-res.fun)

    def _search_midfield_center_circle(
        self,
        score_map: np.ndarray,
        pitch_mask: np.ndarray,
        merged_long: list[tuple],
        merged_trans: list[tuple],
        scale: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[np.ndarray | None, float]:
        """
        Initialize midfield views where the Halfway Line (X=52.5m) + Center Circle (R=9.15m) + Touchlines
        dominate the frame and fewer than 2 transverse straight lines may be visible.
        """
        best_score = -1e9
        best_H: np.ndarray | None = None
        if len(merged_long) < 1 or len(merged_trans) < 1:
            return None, best_score

        long_y_pairs = [(0.0, 68.0), (0.0, 54.16), (13.84, 68.0), (0.0, 43.16), (24.84, 68.0), (24.84, 43.16)]
        scale_fracs = [0.0085, 0.011, 0.014, 0.0175, 0.022, 0.028]

        # Synthesize virtual bottom touchline candidates when Y=68.0 is cropped just below the bottom screen edge
        aug_long = list(merged_long[:6])
        if aug_long:
            L_top = aug_long[0]
            for y_virt_frac in (0.92, 1.04, 1.16):
                y_v = y_virt_frac * frame_h
                c_v = -(L_top[0] * (0.5 * frame_w) + L_top[1] * y_v)
                aug_long.append((L_top[0], L_top[1], c_v, L_top[3], L_top[4]))

        for i1, i2 in itertools.combinations(range(len(aug_long)), 2):
            L1, L2 = aug_long[i1], aug_long[i2]
            vp_l = self.intersect_lines(L1, L2)

            for L_half in merged_trans:
                p_top = self.intersect_lines(L1, L_half)
                p_bot = self.intersect_lines(L2, L_half)
                if p_top is None or p_bot is None:
                    continue
                if abs(p_top[0]) > 2.5 * frame_w or abs(p_bot[0]) > 2.5 * frame_w:
                    continue
                if p_bot[1] <= p_top[1] + 0.12 * frame_h:
                    continue

                # Unit vectors along L1 and L2 pointing left-to-right (+X)
                def unit_dir_along(L: tuple, pt: tuple[float, float]) -> np.ndarray:
                    if vp_l is not None and abs(vp_l[0] - pt[0]) > 0.4 * frame_w:
                        vec = np.array([vp_l[0] - pt[0], vp_l[1] - pt[1]], dtype=np.float64)
                    else:
                        vec = np.array([L[1], -L[0]], dtype=np.float64)
                    if vec[0] < 0:
                        vec = -vec
                    return vec / max(1e-8, float(np.linalg.norm(vec)))

                u_top = unit_dir_along(L1, p_top)
                u_bot = unit_dir_along(L2, p_bot)
                persp_ratio = float(np.clip((p_bot[1] + 0.1 * frame_h) / max(40.0, p_top[1] + 0.1 * frame_h), 1.15, 2.6))

                for y1_m, y2_m in long_y_pairs:
                    pitch_quad = np.array(
                        [
                            [52.5 - 9.15, y1_m],
                            [52.5 + 9.15, y1_m],
                            [52.5 - 9.15, y2_m],
                            [52.5 + 9.15, y2_m],
                        ],
                        dtype=np.float32,
                    )
                    for sf in scale_fracs:
                        r_top_px = 9.15 * sf * frame_w
                        r_bot_px = r_top_px * persp_ratio
                        img_quad = np.array(
                            [
                                np.asarray(p_top) - r_top_px * u_top,
                                np.asarray(p_top) + r_top_px * u_top,
                                np.asarray(p_bot) - r_bot_px * u_bot,
                                np.asarray(p_bot) + r_bot_px * u_bot,
                            ],
                            dtype=np.float32,
                        )
                        H_p2i = cv2.getPerspectiveTransform(pitch_quad, img_quad).astype(np.float64)
                        sc = self.score_homography(H_p2i, score_map, pitch_mask, scale, frame_w, frame_h)
                        if sc > best_score:
                            best_score = sc
                            best_H = H_p2i

        return best_H, best_score

    def initialize_global_homography(
        self,
        score_map: np.ndarray,
        pitch_mask: np.ndarray,
        merged_long: list[tuple],
        merged_trans: list[tuple],
        scale: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[np.ndarray | None, float]:
        """Combinatorial 4-keypoint + Midfield Center-Circle global search across Left, Center, and Right views."""
        best_score = -1e9
        best_H: np.ndarray | None = None
        max_coord = 3.2 * max(frame_w, frame_h)

        # 1. Symmetric 2x2 line-intersection search across all FIFA Y and X markings
        for i1, i2 in itertools.combinations(range(len(merged_long)), 2):
            for j1, j2 in itertools.combinations(range(len(merged_trans)), 2):
                p00 = self.intersect_lines(merged_long[i1], merged_trans[j1])
                p01 = self.intersect_lines(merged_long[i1], merged_trans[j2])
                p10 = self.intersect_lines(merged_long[i2], merged_trans[j1])
                p11 = self.intersect_lines(merged_long[i2], merged_trans[j2])
                if None in (p00, p01, p10, p11):
                    continue
                img_quad = np.array([p00, p01, p10, p11], dtype=np.float32)
                if np.any(np.abs(img_quad) > max_coord):
                    continue

                for y1_m, y2_m in itertools.combinations(LONGITUDINAL_Y, 2):
                    if y2_m - y1_m < 12.0:
                        continue
                    for x1_m, x2_m in itertools.combinations(TRANSVERSE_X, 2):
                        if x2_m - x1_m < 5.0:
                            continue
                        pitch_quad = np.array(
                            [[x1_m, y1_m], [x2_m, y1_m], [x1_m, y2_m], [x2_m, y2_m]],
                            dtype=np.float32,
                        )
                        H_p2i = cv2.getPerspectiveTransform(pitch_quad, img_quad).astype(np.float64)
                        sc = self.score_homography(H_p2i, score_map, pitch_mask, scale, frame_w, frame_h)
                        if sc > best_score:
                            best_score = sc
                            best_H = H_p2i

        # Always evaluate Midfield Center-Circle + Halfway-Line hypotheses alongside 2x2 box hypotheses!
        H_mid, sc_mid = self._search_midfield_center_circle(
            score_map, pitch_mask, merged_long, merged_trans, scale, frame_w, frame_h
        )
        if H_mid is not None and sc_mid > best_score:
            best_score = sc_mid
            best_H = H_mid

        if best_H is not None and best_score > -1e5:
            best_H, best_score = self.refine_homography_local(
                best_H, score_map, pitch_mask, scale, frame_w, frame_h, maxiter=180
            )
        return best_H, best_score

    def snap_to_hough_intersections(
        self,
        H_pred_p2i: np.ndarray,
        merged_long: list[tuple],
        merged_trans: list[tuple],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray | None:
        """Match predicted FIFA intersections against detected Hough intersections using relative pixel gates."""
        gate_px = 0.0115 * frame_w
        ransac_px = max(2.0, 0.0022 * frame_w)
        max_det_x, max_det_y = 1.85 * frame_w, 2.3 * frame_h

        detected_pts = []
        for L in merged_long:
            for T in merged_trans:
                pt = self.intersect_lines(L, T)
                if pt is not None and abs(pt[0]) <= max_det_x and abs(pt[1]) <= max_det_y:
                    detected_pts.append(pt)

        if len(detected_pts) < 4:
            return None

        det_arr = np.asarray(detected_pts, dtype=np.float64)
        # Filter intersections in front of camera before projecting
        ones = np.ones((len(ALL_PROJECTIVE_INTERSECTIONS), 1), dtype=np.float64)
        hom = np.hstack([ALL_PROJECTIVE_INTERSECTIONS, ones]) @ H_pred_p2i.T

        matched_pitch = []
        matched_img = []
        used_det = set()

        margin_x, margin_y = 0.32 * frame_w, 0.36 * frame_h
        for k in range(len(ALL_PROJECTIVE_INTERSECTIONS)):
            if hom[k, 2] <= 1e-3:
                continue
            px = hom[k, 0] / hom[k, 2]
            py = hom[k, 1] / hom[k, 2]
            if not (-margin_x <= px <= frame_w + margin_x and -margin_y <= py <= frame_h + margin_y):
                continue
            dists = np.hypot(det_arr[:, 0] - px, det_arr[:, 1] - py)
            best_d_idx = int(np.argmin(dists))
            if dists[best_d_idx] <= gate_px and best_d_idx not in used_det:
                used_det.add(best_d_idx)
                matched_pitch.append(ALL_PROJECTIVE_INTERSECTIONS[k])
                matched_img.append(det_arr[best_d_idx])

        if len(matched_pitch) < 4:
            return None

        mp_arr = np.asarray(matched_pitch, dtype=np.float64)
        if len(np.unique(np.round(mp_arr[:, 0], 2))) < 2 or len(np.unique(np.round(mp_arr[:, 1], 2))) < 2:
            return None

        H_snap, _ = cv2.findHomography(mp_arr, np.asarray(matched_img, dtype=np.float64), cv2.RANSAC, ransac_px)
        if H_snap is None or not self.is_plausible_broadcast_homography(H_snap, frame_w, frame_h):
            return None
        return H_snap / H_snap[2, 2]

    @staticmethod
    def _smooth_homography_sequence(
        H_p2i_list: list[np.ndarray],
        scores: list[float],
        frame_w: int,
        frame_h: int,
        window: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Smooth homographies strictly on SL(3) without left-penalty-box bias by filtering the
        4 metric pitch coordinates corresponding to the visible screen-interior reference quad!
        """
        n = len(H_p2i_list)
        if n == 0:
            return []

        screen_quad = get_screen_ref_quad(frame_w, frame_h)
        screen_pts64 = screen_quad.astype(np.float64)
        pitch_quad_seq = np.zeros((n, 4, 2), dtype=np.float64)

        for t, H_p2i in enumerate(H_p2i_list):
            try:
                H_i2p = np.linalg.inv(H_p2i)
                pitch_quad_seq[t] = HomographyEstimator.project_points(H_i2p, screen_pts64)
            except np.linalg.LinAlgError:
                pitch_quad_seq[t] = pitch_quad_seq[t - 1] if t > 0 else np.array(
                    [[20.0, 15.0], [85.0, 15.0], [20.0, 53.0], [85.0, 53.0]]
                )

        if n >= window and window >= 3:
            half = window // 2
            sigma = max(1.0, window / 3.0)
            xs = np.arange(-half, half + 1, dtype=np.float64)
            kernel = np.exp(-0.5 * (xs / sigma) ** 2)
            kernel /= np.sum(kernel)
            padded = np.pad(pitch_quad_seq, ((half, half), (0, 0), (0, 0)), mode="edge")
            smoothed_pitch_quad = np.zeros_like(pitch_quad_seq)
            for i in range(n):
                smoothed_pitch_quad[i] = np.tensordot(kernel, padded[i : i + 2 * half + 1], axes=(0, 0))
        else:
            smoothed_pitch_quad = pitch_quad_seq

        results: list[dict[str, Any]] = []
        for t in range(n):
            H_p2i = cv2.getPerspectiveTransform(
                smoothed_pitch_quad[t].astype(np.float32),
                screen_quad,
            ).astype(np.float64)
            H_p2i /= H_p2i[2, 2]
            H_i2p = np.linalg.inv(H_p2i)
            H_i2p /= H_i2p[2, 2]
            results.append(
                {
                    "H_pitch_to_img": H_p2i,
                    "H_img_to_pitch": H_i2p,
                    "confidence": float(max(0.0, scores[t])),
                }
            )
        return results

    def estimate_video_homographies(
        self,
        video_path: str,
        tracks: dict,
        pitch_model: PitchColorModel | None = None,
        scale_ctx: FrameScaleContext | None = None,
        step_homographies: list[np.ndarray] | None = None,
        read_from_stub: bool = True,
        stub_path: str = "stubs/homography_stub.pkl",
        batch_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Compute smooth, view-agnostic per-frame 3x3 homographies across any broadcast video."""
        props = get_video_properties(video_path)
        fw, fh = props["width"], props["height"]

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get("version") == 2 and len(cached.get("homographies", [])) == props["frame_count"]:
                return cached["homographies"]

        if pitch_model is None:
            pitch_model = PitchColorModel.fit_from_video(video_path, read_from_stub=False)
        if scale_ctx is None:
            scale_ctx = FrameScaleContext(fw, fh, props["fps"]).calibrate_from_tracks(tracks)

        num_frames = props["frame_count"]
        polish_stride = max(3, scale_ctx.frames_for_seconds(0.15))

        # 1. Select best geometric anchor keyframe across 10 evenly spaced samples so clips starting
        #    on sparse midfield views inherit exact calibration from frames showing full box + line geometry!
        n_anchors = min(10, max(1, num_frames))
        anchor_indices = np.linspace(0, max(0, num_frames - 1), n_anchors, dtype=np.int32)
        best_anchor_idx = 0
        best_anchor_H: np.ndarray | None = None
        best_anchor_score = -1e9

        cap_anc = cv2.VideoCapture(video_path)
        try:
            for a_idx in anchor_indices:
                cap_anc.set(cv2.CAP_PROP_POS_FRAMES, int(a_idx))
                ret, a_frame = cap_anc.read()
                if not ret:
                    continue
                pbboxes = []
                if int(a_idx) < len(tracks.get("players", [])):
                    pbboxes.extend(p["bbox"] for p in tracks["players"][int(a_idx)].values())
                if int(a_idx) < len(tracks.get("referees", [])):
                    pbboxes.extend(r["bbox"] for r in tracks["referees"][int(a_idx)].values())
                sm, pm, _, ml, mt, sc_f = self.extract_lines_and_score_map(a_frame, pbboxes, pitch_model)
                H_cand, sc_cand = self.initialize_global_homography(sm, pm, ml, mt, sc_f, fw, fh)
                if H_cand is not None and sc_cand > best_anchor_score:
                    best_anchor_score = sc_cand
                    best_anchor_H = H_cand
                    best_anchor_idx = int(a_idx)
        finally:
            cap_anc.release()

        if best_anchor_H is None:
            scr_q = get_screen_ref_quad(fw, fh)
            def_p = np.array([[18.0, 8.0], [87.0, 8.0], [18.0, 60.0], [87.0, 60.0]], dtype=np.float32)
            best_anchor_H = cv2.getPerspectiveTransform(def_p, scr_q).astype(np.float64)
            best_anchor_score = 0.0

        # Extract per-frame compact line features in a single streaming pass (int8 quantized for low RAM)
        frame_features: list[tuple[np.ndarray, np.ndarray, list[tuple], list[tuple], float]] = []
        f_idx = 0
        for frame_batch in iter_video_batches(video_path, batch_size=batch_size):
            for frame in frame_batch:
                pbboxes = []
                if f_idx < len(tracks.get("players", [])):
                    pbboxes.extend(p["bbox"] for p in tracks["players"][f_idx].values())
                if f_idx < len(tracks.get("referees", [])):
                    pbboxes.extend(r["bbox"] for r in tracks["referees"][f_idx].values())
                sm, pm, _, ml, mt, sc_f = self.extract_lines_and_score_map(frame, pbboxes, pitch_model)
                frame_features.append((np.round(sm * 100.0).astype(np.int8), pm, ml, mt, sc_f))
                f_idx += 1

        actual_n = len(frame_features)
        best_anchor_idx = min(best_anchor_idx, max(0, actual_n - 1))
        raw_H_p2i: list[np.ndarray] = [np.eye(3, dtype=np.float64)] * actual_n
        raw_scores: list[float] = [0.0] * actual_n
        raw_H_p2i[best_anchor_idx] = best_anchor_H
        raw_scores[best_anchor_idx] = best_anchor_score

        def track_step(t_curr: int, H_pred: np.ndarray) -> tuple[np.ndarray, float]:
            sm_i8, pm, ml, mt, sc_f = frame_features[t_curr]
            sm = sm_i8.astype(np.float32) * 0.01
            H_pred = H_pred / (H_pred[2, 2] if abs(H_pred[2, 2]) > 1e-9 else 1.0)
            sc_pred = self.score_homography(H_pred, sm, pm, sc_f, fw, fh)

            H_snap = self.snap_to_hough_intersections(H_pred, ml, mt, fw, fh)
            sc_snap = self.score_homography(H_snap, sm, pm, sc_f, fw, fh) if H_snap is not None else -1e9

            H_c = H_snap if sc_snap > sc_pred + 1.5 else H_pred
            sc_c = max(sc_pred, sc_snap)

            if abs(t_curr - best_anchor_idx) % polish_stride == 0 or sc_c < 35.0:
                H_pol, sc_pol = self.refine_homography_local(H_c, sm, pm, sc_f, fw, fh, maxiter=80)
                if sc_pol > sc_c:
                    H_c, sc_c = H_pol, sc_pol

            # Scene-cut recovery only if propagated projection becomes geometrically implausible
            if sc_c < -1e4:
                H_glob, sc_glob = self.initialize_global_homography(sm, pm, ml, mt, sc_f, fw, fh)
                if H_glob is not None and sc_glob > sc_c:
                    H_c, sc_c = H_glob, sc_glob

            return H_c, sc_c

        # 2a. Forward propagation: best_anchor_idx + 1 .. actual_n - 1
        prev_H = best_anchor_H
        for t in range(best_anchor_idx + 1, actual_n):
            if step_homographies is not None and t < len(step_homographies):
                try:
                    H_prev_to_curr = np.linalg.inv(step_homographies[t])
                except np.linalg.LinAlgError:
                    H_prev_to_curr = np.eye(3, dtype=np.float64)
                H_pred = H_prev_to_curr @ prev_H
            else:
                H_pred = prev_H.copy()
            prev_H, sc_t = track_step(t, H_pred)
            raw_H_p2i[t] = prev_H
            raw_scores[t] = sc_t

        # 2b. Backward propagation: best_anchor_idx - 1 .. 0 (note step_homographies[t+1] maps frame t+1 -> frame t!)
        next_H = best_anchor_H
        for t in range(best_anchor_idx - 1, -1, -1):
            if step_homographies is not None and (t + 1) < len(step_homographies):
                H_next_to_curr = step_homographies[t + 1]
                H_pred = H_next_to_curr @ next_H
            else:
                H_pred = next_H.copy()
            next_H, sc_t = track_step(t, H_pred)
            raw_H_p2i[t] = next_H
            raw_scores[t] = sc_t

        print(
            f"Bidirectional homography registration complete ({actual_n} frames, "
            f"anchor={best_anchor_idx} score={best_anchor_score:.1f}, median={float(np.median(raw_scores)):.1f})"
        )

        smooth_win = scale_ctx.frames_for_seconds(0.20, minimum=3)
        if smooth_win % 2 == 0:
            smooth_win += 1
        homographies = self._smooth_homography_sequence(raw_H_p2i, raw_scores, fw, fh, window=smooth_win)

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump({"version": 2, "homographies": homographies}, f)

        return homographies

    def add_transformed_positions_to_tracks(
        self,
        tracks: dict,
        homographies: list[dict[str, Any]],
        scale_ctx: FrameScaleContext | None = None,
    ) -> None:
        """Project bottom-center foot coordinates (and ball coordinates) onto the 2D metric FIFA pitch."""
        num_frames = min(len(homographies), len(tracks.get("players", [])))
        smooth_half = max(1, scale_ctx.frames_for_seconds(0.08)) if scale_ctx is not None else 2

        for f in range(num_frames):
            H_i2p = homographies[f]["H_img_to_pitch"]
            conf = float(homographies[f].get("confidence", 50.0))

            for group in ("players", "referees"):
                obj_dict = tracks[group][f]
                if not obj_dict:
                    continue
                tids = list(obj_dict.keys())
                feet = np.array([get_foot_position(obj_dict[tid]["bbox"]) for tid in tids], dtype=np.float64)
                proj = self.project_points(H_i2p, feet)
                for tid, (xm, ym) in zip(tids, proj):
                    obj_dict[tid]["position_unclamped"] = [float(xm), float(ym)]
                    # Metric tactical bounds check (if homography confidence is weak, preserve in-pitch eligibility)
                    in_metric = -1.2 <= xm <= PITCH_LENGTH + 1.2 and -1.2 <= ym <= PITCH_WIDTH + 1.2
                    obj_dict[tid]["on_pitch"] = bool(in_metric or conf < 20.0)
                    if -6.0 <= xm <= PITCH_LENGTH + 6.0 and -6.0 <= ym <= PITCH_WIDTH + 6.0:
                        obj_dict[tid]["position_transformed"] = [
                            float(np.clip(xm, 0.0, PITCH_LENGTH)),
                            float(np.clip(ym, 0.0, PITCH_WIDTH)),
                        ]
                    else:
                        obj_dict[tid]["position_transformed"] = None

            ball_dict = tracks.get("ball", [])[f] if f < len(tracks.get("ball", [])) else {}
            for _, bdata in ball_dict.items():
                bx, _ = get_center_of_bbox(bdata["bbox"])
                foot_ball = np.array([[float(bx), float(bdata["bbox"][3])]], dtype=np.float64)
                xm, ym = self.project_points(H_i2p, foot_ball)[0]
                bdata["position_transformed"] = [
                    float(np.clip(xm, 0.0, PITCH_LENGTH)),
                    float(np.clip(ym, 0.0, PITCH_WIDTH)),
                ]

        # FPS-scaled trajectory smoothing per track_id
        for group in ("players", "referees"):
            series_by_id: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
            for f in range(num_frames):
                for tid, odata in tracks[group][f].items():
                    pt = odata.get("position_transformed")
                    if pt is not None:
                        series_by_id[tid].append((f, pt))

            for tid, obs in series_by_id.items():
                if len(obs) < 3:
                    continue
                frames_arr = np.array([o[0] for o in obs], dtype=np.int32)
                coords_arr = np.array([o[1] for o in obs], dtype=np.float64)
                smoothed = coords_arr.copy()
                for idx in range(len(obs)):
                    lo = max(0, idx - smooth_half)
                    hi = min(len(obs), idx + smooth_half + 1)
                    window_mask = np.abs(frames_arr[lo:hi] - frames_arr[idx]) <= smooth_half
                    smoothed[idx] = np.mean(coords_arr[lo:hi][window_mask], axis=0)
                for (f_obs, _), sm_pt in zip(obs, smoothed):
                    tracks[group][int(f_obs)][tid]["position_transformed"] = sm_pt.tolist()
