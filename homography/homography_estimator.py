import itertools
import os
import pickle
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize

from utils import get_center_of_bbox, get_foot_position, iter_video_batches
from .pitch_model import (
    ALL_PROJECTIVE_INTERSECTIONS,
    CANONICAL_PITCH_QUAD,
    LONGITUDINAL_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TRANSVERSE_X,
    build_fifa_template_points,
)


class HomographyEstimator:
    """Dynamic broadcast field registration using HSV pitch masking, Canny/Hough keypoints, and temporal tracking."""

    WORK_SCALE = 0.5  # 960x540 processing scale on 1080p footage

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

    @classmethod
    def extract_lines_and_score_map(
        cls,
        frame: np.ndarray,
        person_bboxes: list[list[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple], list[tuple], float]:
        """Extract HSV grass hull, Morphological White Top-Hat + Canny edges, and merged Hough line families."""
        h, w = frame.shape[:2]
        scale = cls.WORK_SCALE
        sw, sh = int(w * scale), int(h * scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        # 1. HSV Playable-Pitch Convex Hull Mask (trimmed along stadium signage, preserved at screen borders)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, np.array([28, 35, 35], dtype=np.uint8), np.array([92, 255, 255], dtype=np.uint8))
        grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pitch_mask = np.zeros((sh, sw), dtype=np.uint8)
        if contours:
            hull = cv2.convexHull(max(contours, key=cv2.contourArea))
            pad = 16
            padded = np.zeros((sh + 2 * pad, sw + 2 * pad), dtype=np.uint8)
            cv2.fillConvexPoly(padded, hull + np.array([[[pad, pad]]], dtype=np.int32), 255)
            inner = padded[pad : pad + sh, pad : pad + sw]
            if np.any(inner[0, :] > 0):
                padded[:pad, pad : pad + sw] = inner[0:1, :]
            if np.any(inner[-1, :] > 0):
                padded[pad + sh :, pad : pad + sw] = inner[-1:, :]
            if np.any(inner[:, 0] > 0):
                padded[pad : pad + sh, :pad] = inner[:, 0:1]
            if np.any(inner[:, -1] > 0):
                padded[pad : pad + sh, pad + sw :] = inner[:, -1:]
            eroded = cv2.erode(padded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            pitch_mask = eroded[pad : pad + sh, pad : pad + sw]

        # Mask out tracked players, goalkeepers, and referees so white kits/socks never create false lines
        clean_mask = pitch_mask.copy()
        for bx1, by1, bx2, by2 in person_bboxes:
            x1 = max(0, int(bx1 * scale) - 4)
            y1 = max(0, int(by1 * scale) - 4)
            x2 = min(sw, int(bx2 * scale) + 4)
            y2 = min(sh, int(by2 * scale) + 4)
            clean_mask[y1:y2, x1:x2] = 0

        # 2. Morphological White Top-Hat + Canny Edge Detection
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))
        tophat_masked = cv2.bitwise_and(tophat, tophat, mask=clean_mask)

        _, white_bin = cv2.threshold(tophat_masked, 15, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(white_bin, 50, 150)

        blurred = cv2.GaussianBlur(white_bin.astype(np.float32) / 255.0, (0, 0), sigmaX=3.0)
        line_score_map = np.clip(blurred * 2.5, 0.0, 1.0) - 0.15

        # 3. Probabilistic Hough Transform & Signed Perspective Family Clustering
        raw_segs = cv2.HoughLinesP(edges, rho=1.0, theta=np.pi / 180.0, threshold=28, minLineLength=30, maxLineGap=22)
        long_segs = []
        trans_segs = []

        if raw_segs is not None:
            segs = raw_segs.reshape(-1, 4).astype(np.float64) / scale
            for x1, y1, x2, y2 in segs:
                if x2 < x1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                length = np.hypot(x2 - x1, y2 - y1)
                if length < 55.0:
                    continue
                ang_deg = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                if -8.0 <= ang_deg <= 68.0:
                    long_segs.append((x1, y1, x2, y2, length))
                elif ang_deg <= -18.0 or ang_deg >= 75.0:
                    trans_segs.append((x1, y1, x2, y2, length))

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
                    if d1 < 15.0 and d2 < 15.0 and ang_diff < np.deg2rad(4.0):
                        cl["segs"].append(seg)
                        pts = []
                        for sx1, sy1, sx2, sy2, slen in cl["segs"]:
                            for t_val in np.linspace(0.0, 1.0, max(2, int(slen // 20))):
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

            strong = [cl["line"] for cl in clusters if cl["line"][4] >= 85.0]
            if is_long:
                strong.sort(key=lambda L: -(L[0] * 450.0 + L[2]) / (L[1] if abs(L[1]) > 1e-6 else 1e-6))
            else:
                strong.sort(key=lambda L: -(L[1] * 400.0 + L[2]) / (L[0] if abs(L[0]) > 1e-6 else 1e-6))
            return strong[:8]

        return (
            line_score_map,
            pitch_mask,
            edges,
            merge_family(long_segs, True),
            merge_family(trans_segs, False),
            scale,
        )

    @staticmethod
    def intersect_lines(L1: tuple, L2: tuple) -> tuple[float, float] | None:
        """Compute intersection (x, y) of two homogeneous lines a*x + b*y + c = 0."""
        a1, b1, c1 = L1[:3]
        a2, b2, c2 = L2[:3]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-7:
            return None
        return ((b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det)

    @staticmethod
    def is_plausible_broadcast_homography(H_p2i: np.ndarray) -> bool:
        """Verify perspective convexity, orientation handedness, and broadcast scale plausibility."""
        ref = np.hstack([CANONICAL_PITCH_QUAD.astype(np.float64), np.ones((4, 1), dtype=np.float64)])
        hom = ref @ H_p2i.T
        if np.any(hom[:, 2] <= 1e-3):
            return False
        pts = hom[:, :2] / hom[:, 2:3]
        p_tl, p_tr, p_bl, p_br = pts

        if p_tl[1] >= p_bl[1] - 50.0 or p_tr[1] >= p_br[1] - 60.0:
            return False
        if p_tr[0] <= p_tl[0] + 50.0 or p_br[0] <= p_bl[0] + 50.0:
            return False

        box_left_span = np.hypot(*(p_bl - p_tl))
        box_right_span = np.hypot(*(p_br - p_tr))
        box_depth_top = np.hypot(*(p_tr - p_tl))
        if not (140.0 <= box_left_span <= 1150.0 and 140.0 <= box_right_span <= 1250.0):
            return False
        if not (70.0 <= box_depth_top <= 850.0):
            return False

        quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)
        return bool(cv2.isContourConvex(quad))

    def score_homography(
        self,
        H_p2i: np.ndarray,
        score_map: np.ndarray,
        pitch_mask: np.ndarray,
        scale: float,
    ) -> float:
        """Score a pitch-to-image homography by spatial grid-cell white-line coverage."""
        if not self.is_plausible_broadcast_homography(H_p2i):
            return -1e6

        sh, sw = score_map.shape[:2]
        ones = np.ones((len(self.template_pts), 1), dtype=np.float64)
        hom = np.hstack([self.template_pts, ones]) @ H_p2i.T
        z = hom[:, 2]
        valid_z = z > 1e-3
        if np.sum(valid_z) < 80:
            return -1e6

        ux = (hom[:, 0] / np.where(valid_z, z, 1.0)) * scale
        uy = (hom[:, 1] / np.where(valid_z, z, 1.0)) * scale
        in_bounds = valid_z & (ux >= 0) & (ux < sw - 1) & (uy >= 0) & (uy < sh - 1)
        if np.sum(in_bounds) < 80:
            return -1e6

        ix = np.round(ux[in_bounds]).astype(np.int32)
        iy = np.round(uy[in_bounds]).astype(np.int32)
        on_grass = pitch_mask[iy, ix] > 0
        if int(np.sum(on_grass)) < 70:
            return -1e6

        gx, gy = ix[on_grass], iy[on_grass]
        cell_ids = (gy // 12) * 100 + (gx // 12)
        unique_cells, cell_inv = np.unique(cell_ids, return_inverse=True)
        if len(unique_cells) < 35:
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
        maxiter: int = 120,
    ) -> tuple[np.ndarray, float]:
        """Sub-pixel Powell refinement of H_p2i against the frame's white-line response map."""
        H0 = H_init / (H_init[2, 2] if abs(H_init[2, 2]) > 1e-9 else 1.0)

        def build_perturbed(d: np.ndarray) -> np.ndarray:
            D = np.eye(3, dtype=np.float64)
            D[0, 0] += d[0] * 0.012
            D[0, 1] += d[1] * 0.012
            D[0, 2] += d[2] * 10.0
            D[1, 0] += d[3] * 0.012
            D[1, 1] += d[4] * 0.012
            D[1, 2] += d[5] * 10.0
            D[2, 0] += d[6] * 4e-5
            D[2, 1] += d[7] * 4e-5
            return D @ H0

        res = minimize(
            lambda d: -self.score_homography(build_perturbed(d), score_map, pitch_mask, scale),
            np.zeros(8, dtype=np.float64),
            method="Powell",
            options={"maxiter": maxiter, "ftol": 2e-3},
        )
        H_opt = build_perturbed(res.x)
        H_opt /= H_opt[2, 2]
        return H_opt, float(-res.fun)

    def initialize_global_homography(
        self,
        score_map: np.ndarray,
        pitch_mask: np.ndarray,
        merged_long: list[tuple],
        merged_trans: list[tuple],
        scale: float,
    ) -> tuple[np.ndarray | None, float]:
        """Combinatorial 4-keypoint cv2.getPerspectiveTransform search over detected Canny/Hough lines."""
        best_score = -1e9
        best_H: np.ndarray | None = None

        # Candidate X lines visible in typical half/midfield views (left or right half)
        trans_candidates = [0.0, 5.5, 16.5, 52.5, 88.5, 99.5, 105.0]

        for i1, i2 in itertools.combinations(range(len(merged_long)), 2):
            for j1, j2 in itertools.combinations(range(len(merged_trans)), 2):
                p00 = self.intersect_lines(merged_long[i1], merged_trans[j1])
                p01 = self.intersect_lines(merged_long[i1], merged_trans[j2])
                p10 = self.intersect_lines(merged_long[i2], merged_trans[j1])
                p11 = self.intersect_lines(merged_long[i2], merged_trans[j2])
                if None in (p00, p01, p10, p11):
                    continue
                img_quad = np.array([p00, p01, p10, p11], dtype=np.float32)
                if np.any(np.abs(img_quad) > 6000):
                    continue

                for y1_m, y2_m in itertools.combinations(LONGITUDINAL_Y, 2):
                    if y2_m - y1_m < 12.0:
                        continue
                    for x1_m, x2_m in itertools.combinations(trans_candidates, 2):
                        if x2_m - x1_m < 5.0:
                            continue
                        pitch_quad = np.array(
                            [[x1_m, y1_m], [x2_m, y1_m], [x1_m, y2_m], [x2_m, y2_m]],
                            dtype=np.float32,
                        )
                        H_p2i = cv2.getPerspectiveTransform(pitch_quad, img_quad).astype(np.float64)
                        sc = self.score_homography(H_p2i, score_map, pitch_mask, scale)
                        if sc > best_score:
                            best_score = sc
                            best_H = H_p2i

        if best_H is not None and best_score > -1e5:
            best_H, best_score = self.refine_homography_local(best_H, score_map, pitch_mask, scale, maxiter=180)
        return best_H, best_score

    def snap_to_hough_intersections(
        self,
        H_pred_p2i: np.ndarray,
        merged_long: list[tuple],
        merged_trans: list[tuple],
        gate_px: float = 22.0,
    ) -> np.ndarray | None:
        """Match predicted FIFA intersections against detected Hough intersections and solve cv2.findHomography."""
        detected_pts = []
        for L in merged_long:
            for T in merged_trans:
                pt = self.intersect_lines(L, T)
                if pt is not None and abs(pt[0]) <= 3500 and abs(pt[1]) <= 2500:
                    detected_pts.append(pt)

        if len(detected_pts) < 4:
            return None

        det_arr = np.asarray(detected_pts, dtype=np.float64)
        pred_img_pts = self.project_points(H_pred_p2i, ALL_PROJECTIVE_INTERSECTIONS)

        matched_pitch = []
        matched_img = []
        used_det = set()

        for k, p_pred in enumerate(pred_img_pts):
            if not (-600 <= p_pred[0] <= 2520 and -400 <= p_pred[1] <= 1480):
                continue
            dists = np.hypot(det_arr[:, 0] - p_pred[0], det_arr[:, 1] - p_pred[1])
            best_d_idx = int(np.argmin(dists))
            if dists[best_d_idx] <= gate_px and best_d_idx not in used_det:
                used_det.add(best_d_idx)
                matched_pitch.append(ALL_PROJECTIVE_INTERSECTIONS[k])
                matched_img.append(det_arr[best_d_idx])

        if len(matched_pitch) < 4:
            return None

        mp_arr = np.asarray(matched_pitch, dtype=np.float64)
        # Require at least 2 distinct X lines and 2 distinct Y lines (non-collinear!)
        if len(np.unique(np.round(mp_arr[:, 0], 2))) < 2 or len(np.unique(np.round(mp_arr[:, 1], 2))) < 2:
            return None

        H_snap, _ = cv2.findHomography(mp_arr, np.asarray(matched_img, dtype=np.float64), cv2.RANSAC, 4.0)
        if H_snap is None or not self.is_plausible_broadcast_homography(H_snap):
            return None
        return H_snap / H_snap[2, 2]

    @staticmethod
    def _smooth_homography_sequence(H_p2i_list: list[np.ndarray], window: int = 5) -> list[dict[str, np.ndarray]]:
        """Smooth homographies strictly on the SL(3) perspective manifold by filtering 4 canonical quad corners."""
        n = len(H_p2i_list)
        if n == 0:
            return []

        quad_pts = CANONICAL_PITCH_QUAD.astype(np.float64)
        quad_seq = np.zeros((n, 4, 2), dtype=np.float64)
        for t, H in enumerate(H_p2i_list):
            quad_seq[t] = HomographyEstimator.project_points(H, quad_pts)

        if n >= window:
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
            pad = window // 2
            padded = np.pad(quad_seq, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            smoothed_quad = np.zeros_like(quad_seq)
            for i in range(n):
                smoothed_quad[i] = np.tensordot(kernel, padded[i : i + window], axes=(0, 0))
        else:
            smoothed_quad = quad_seq

        results: list[dict[str, np.ndarray]] = []
        for t in range(n):
            H_p2i = cv2.getPerspectiveTransform(
                CANONICAL_PITCH_QUAD, smoothed_quad[t].astype(np.float32)
            ).astype(np.float64)
            H_i2p = np.linalg.inv(H_p2i)
            H_i2p /= H_i2p[2, 2]
            results.append(
                {
                    "H_pitch_to_img": H_p2i,
                    "H_img_to_pitch": H_i2p,
                }
            )
        return results

    def estimate_video_homographies(
        self,
        video_path: str,
        tracks: dict,
        step_homographies: list[np.ndarray] | None = None,
        read_from_stub: bool = True,
        stub_path: str = "stubs/homography_stub.pkl",
        batch_size: int = 20,
    ) -> list[dict[str, np.ndarray]]:
        """Compute smooth per-frame 3x3 homographies across a broadcast video, with stub caching."""
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        raw_H_p2i: list[np.ndarray] = []
        prev_H_p2i: np.ndarray | None = None
        frame_idx = 0

        for frame_batch in iter_video_batches(video_path, batch_size=batch_size):
            for frame in frame_batch:
                pbboxes = []
                if frame_idx < len(tracks.get("players", [])):
                    pbboxes.extend(p["bbox"] for p in tracks["players"][frame_idx].values())
                if frame_idx < len(tracks.get("referees", [])):
                    pbboxes.extend(r["bbox"] for r in tracks["referees"][frame_idx].values())

                score_map, pitch_mask, _, merged_long, merged_trans, scale = self.extract_lines_and_score_map(
                    frame, pbboxes
                )

                if prev_H_p2i is None:
                    H_curr, sc_curr = self.initialize_global_homography(
                        score_map, pitch_mask, merged_long, merged_trans, scale
                    )
                    if H_curr is None:
                        H_curr = np.eye(3, dtype=np.float64)
                else:
                    # 1. Optical-flow camera motion propagation: inv(H_{t -> t-1}) @ H_{t-1, p2i}
                    if step_homographies is not None and frame_idx < len(step_homographies):
                        try:
                            H_prev_to_curr = np.linalg.inv(step_homographies[frame_idx])
                        except np.linalg.LinAlgError:
                            H_prev_to_curr = np.eye(3, dtype=np.float64)
                        H_pred = H_prev_to_curr @ prev_H_p2i
                    else:
                        H_pred = prev_H_p2i.copy()
                    H_pred /= H_pred[2, 2]
                    sc_pred = self.score_homography(H_pred, score_map, pitch_mask, scale)

                    # 2. Closed-form Canny/Hough keypoint snap via cv2.findHomography
                    H_snap = self.snap_to_hough_intersections(H_pred, merged_long, merged_trans, gate_px=22.0)
                    sc_snap = (
                        self.score_homography(H_snap, score_map, pitch_mask, scale)
                        if H_snap is not None
                        else -1e9
                    )

                    H_curr = H_snap if sc_snap > sc_pred + 1.5 else H_pred
                    sc_curr = max(sc_pred, sc_snap)

                    # 3. Periodic sub-pixel white-line polish every 4 frames (or whenever alignment dips)
                    if frame_idx % 4 == 0 or sc_curr < 65.0:
                        H_pol, sc_pol = self.refine_homography_local(
                            H_curr, score_map, pitch_mask, scale, maxiter=85
                        )
                        if sc_pol > sc_curr:
                            H_curr, sc_curr = H_pol, sc_pol

                    # 4. Scene-cut / recovery fallback if alignment ever collapses
                    if sc_curr < 38.0:
                        H_glob, sc_glob = self.initialize_global_homography(
                            score_map, pitch_mask, merged_long, merged_trans, scale
                        )
                        if H_glob is not None and sc_glob > sc_curr + 10.0:
                            H_curr, sc_curr = H_glob, sc_glob

                prev_H_p2i = H_curr
                raw_H_p2i.append(H_curr)
                frame_idx += 1
                if frame_idx % 150 == 0:
                    print(f"Dynamic homography registration: {frame_idx} frames (score={sc_curr:.1f})...")

        homographies = self._smooth_homography_sequence(raw_H_p2i, window=5)

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(homographies, f)

        return homographies

    def add_transformed_positions_to_tracks(
        self,
        tracks: dict,
        homographies: list[dict[str, np.ndarray]],
    ) -> None:
        """Project bottom-center foot coordinates (and ball coordinates) onto the 2D metric FIFA pitch."""
        num_frames = min(len(homographies), len(tracks.get("players", [])))

        # Project per-frame coordinates
        for f in range(num_frames):
            H_i2p = homographies[f]["H_img_to_pitch"]

            for group in ("players", "referees"):
                obj_dict = tracks[group][f]
                if not obj_dict:
                    continue
                tids = list(obj_dict.keys())
                feet = np.array([get_foot_position(obj_dict[tid]["bbox"]) for tid in tids], dtype=np.float64)
                proj = self.project_points(H_i2p, feet)
                for tid, (xm, ym) in zip(tids, proj):
                    obj_dict[tid]["position_unclamped"] = [float(xm), float(ym)]
                    obj_dict[tid]["on_pitch"] = bool(
                        -0.8 <= xm <= PITCH_LENGTH + 0.8 and -0.8 <= ym <= PITCH_WIDTH + 0.8
                    )
                    if -4.0 <= xm <= PITCH_LENGTH + 4.0 and -4.0 <= ym <= PITCH_WIDTH + 4.0:
                        obj_dict[tid]["position_transformed"] = [
                            float(np.clip(xm, 0.0, PITCH_LENGTH)),
                            float(np.clip(ym, 0.0, PITCH_WIDTH)),
                        ]
                    else:
                        obj_dict[tid]["position_transformed"] = None

            ball_dict = tracks.get("ball", [])[f] if f < len(tracks.get("ball", [])) else {}
            for bid, bdata in ball_dict.items():
                bx, by = get_center_of_bbox(bdata["bbox"])
                # Use bottom of ball bbox (grass contact approximation)
                foot_ball = np.array([[float(bx), float(bdata["bbox"][3])]], dtype=np.float64)
                xm, ym = self.project_points(H_i2p, foot_ball)[0]
                bdata["position_transformed"] = [
                    float(np.clip(xm, 0.0, PITCH_LENGTH)),
                    float(np.clip(ym, 0.0, PITCH_WIDTH)),
                ]

        # Apply light 5-frame trajectory smoothing per track_id for clean 2D radar & M3 formation clustering
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
                    # Average neighbors within +-2 frames
                    lo = max(0, idx - 2)
                    hi = min(len(obs), idx + 3)
                    window_mask = np.abs(frames_arr[lo:hi] - frames_arr[idx]) <= 2
                    smoothed[idx] = np.mean(coords_arr[lo:hi][window_mask], axis=0)
                for (f_obs, _), sm_pt in zip(obs, smoothed):
                    tracks[group][int(f_obs)][tid]["position_transformed"] = sm_pt.tolist()

