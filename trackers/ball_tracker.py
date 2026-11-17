import os
import pickle
from typing import Any

import cv2
import numpy as np
from sahi.slicing import get_slice_bboxes

from utils import (
    FrameScaleContext,
    PitchColorModel,
    get_foot_position,
    get_video_properties,
    iter_video_batches,
)
from ultralytics import YOLO


class BallTracker:
    """
    Generalized small-object ball tracker obeying Zero Hardcoded Visuals, Relative Geometry Only,
    and Option A Hybrid SAHI Sliced Inference + Camera-Stabilized 4-State Kalman / RTS Smoothing.
    """

    FLOW_WORK_W = 480

    def __init__(self, sahi_stride: int = 4) -> None:
        self.sahi_stride = max(1, int(sahi_stride))

    @classmethod
    def estimate_step_homography(
        cls,
        prev_gray_small: np.ndarray,
        curr_gray_small: np.ndarray,
        person_bboxes_full: list[list[float]],
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        """Estimate full-resolution 3x3 homography H mapping curr_frame background coords -> prev_frame coords."""
        sh, sw = prev_gray_small.shape[:2]
        scale_x = sw / float(max(1, orig_w))
        scale_y = sh / float(max(1, orig_h))
        diag_s = float(np.hypot(sw, sh))

        bg_mask = np.full((sh, sw), 255, dtype=np.uint8)
        pad_px = max(2, int(round(0.006 * diag_s)))
        for bbox in person_bboxes_full:
            x1 = max(0, int(round(bbox[0] * scale_x)) - pad_px)
            y1 = max(0, int(round(bbox[1] * scale_y)) - pad_px)
            x2 = min(sw, int(round(bbox[2] * scale_x)) + pad_px)
            y2 = min(sh, int(round(bbox[3] * scale_y)) + pad_px)
            bg_mask[y1:y2, x1:x2] = 0

        min_dist = max(4, int(round(0.015 * diag_s)))
        win_px = max(11, int(round(0.028 * diag_s)) | 1)
        ransac_thr = max(1.5, 0.0036 * diag_s)

        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray_small,
            maxCorners=350,
            qualityLevel=0.01,
            minDistance=min_dist,
            mask=bg_mask,
        )
        if pts_prev is None or len(pts_prev) < 12:
            return np.eye(3, dtype=np.float64)

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_small,
            curr_gray_small,
            pts_prev,
            None,
            winSize=(win_px, win_px),
            maxLevel=3,
        )
        if pts_curr is None or status is None:
            return np.eye(3, dtype=np.float64)

        good = status.ravel() == 1
        src = pts_curr[good].reshape(-1, 2)
        dst = pts_prev[good].reshape(-1, 2)
        if len(src) < 10:
            return np.eye(3, dtype=np.float64)

        H_small, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=ransac_thr)
        if H_small is None:
            return np.eye(3, dtype=np.float64)

        S = np.diag([scale_x, scale_y, 1.0])
        S_inv = np.diag([1.0 / scale_x, 1.0 / scale_y, 1.0])
        return S_inv @ H_small @ S

    @staticmethod
    def _warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply 3x3 homography H to an (M, 2) float64 array of points."""
        if len(pts) == 0:
            return pts
        ones = np.ones((len(pts), 1), dtype=np.float64)
        hom = np.hstack([pts, ones]) @ H.T
        denom = np.where(np.abs(hom[:, 2:3]) > 1e-8, hom[:, 2:3], 1.0)
        return hom[:, :2] / denom

    @staticmethod
    def _bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        area_a = max(1e-3, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1e-3, (b[2] - b[0]) * (b[3] - b[1]))
        return float(inter / (area_a + area_b - inter))

    def _merge_tile_candidates_nmm(
        self,
        raw_cands: list[dict[str, Any]],
        scale_ctx: FrameScaleContext,
    ) -> list[dict[str, Any]]:
        """Greedy Non-Maximum Merging across overlapping SAHI slices using relative center/IoU gates."""
        if len(raw_cands) <= 1:
            return raw_cands
        ordered = sorted(raw_cands, key=lambda c: c["conf"], reverse=True)
        kept: list[dict[str, Any]] = []
        for cand in ordered:
            hp = scale_ctx.player_h_at(cand["cy"])
            merge_dist = max(3.0, 0.16 * hp)
            duplicate = False
            for k in kept:
                if (
                    np.hypot(cand["cx"] - k["cx"], cand["cy"] - k["cy"]) <= merge_dist
                    or self._bbox_iou_xyxy(cand["bbox"], k["bbox"]) >= 0.38
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    def _filter_raw_ball_rows(
        self,
        ball_rows: np.ndarray,
        offset_x: int,
        offset_y: int,
        pitch_mask_small: np.ndarray,
        scale_ctx: FrameScaleContext,
        source_tag: str,
    ) -> list[dict[str, Any]]:
        """Filter raw YOLO ball detections strictly via relative player-height bounds and dynamic pitch mask."""
        accepted: list[dict[str, Any]] = []
        orig_w, orig_h = scale_ctx.frame_w, scale_ctx.frame_h

        for row in ball_rows:
            bx1, by1, bx2, by2, conf, _ = row.tolist()
            x1, y1, x2, y2 = bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y
            w = x2 - x1
            h = y2 - y1
            if w <= 1.0 or h <= 1.0:
                continue
            aspect = max(w / h, h / w)
            if aspect > 2.5:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            min_s, max_s = scale_ctx.ball_size_bounds_at(cy)
            if not (min_s <= w <= max_s and min_s <= h <= max_s):
                continue
            if not PitchColorModel.is_point_in_mask(pitch_mask_small, cx, cy, orig_w, orig_h):
                continue

            accepted.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "cx": cx,
                    "cy": cy,
                    "conf": float(conf),
                    "source": source_tag,
                }
            )
        return accepted

    def _run_sahi_pitch_slices_on_frames(
        self,
        frames_with_masks: list[tuple[int, np.ndarray, np.ndarray]],
        model: YOLO,
        scale_ctx: FrameScaleContext,
        tile_batch_size: int = 16,
    ) -> dict[int, list[dict[str, Any]]]:
        """Run batched SAHI overlapping sliced inference over the active playable pitch band."""
        if not frames_with_masks:
            return {}

        orig_w, orig_h = scale_ctx.frame_w, scale_ctx.frame_h
        # 1x3 overlapping SAHI horizontal pitch-band slices at stride-32 (high-res wide-shot ball recall, 2.7x faster)
        slice_w = max(320, int(round((0.42 * orig_w) / 32.0)) * 32)
        tile_imgsz = 640 if orig_w >= 1280 else 512

        tile_imgs: list[np.ndarray] = []
        tile_meta: list[tuple[int, int, int, np.ndarray]] = []  # (frame_idx, x0, y0, pmask)
        per_frame_raw: dict[int, list[dict[str, Any]]] = {f_idx: [] for f_idx, _, _ in frames_with_masks}

        def flush_tiles() -> None:
            if not tile_imgs:
                return
            preds = model.predict(tile_imgs, imgsz=tile_imgsz, conf=0.05, verbose=False)
            for (f_idx, x0, y0, pmask), res in zip(tile_meta, preds):
                cls_inv = {v: k for k, v in res.names.items()}
                ball_cls = cls_inv["ball"]
                boxes_np = res.boxes.cpu().numpy().data
                if len(boxes_np) > 0:
                    ball_rows = boxes_np[boxes_np[:, 5].astype(int) == ball_cls]
                    per_frame_raw[f_idx].extend(
                        self._filter_raw_ball_rows(ball_rows, x0, y0, pmask, scale_ctx, "sahi_tile")
                    )
            tile_imgs.clear()
            tile_meta.clear()

        for f_idx, frame, pmask in frames_with_masks:
            y_top, y_bot = PitchColorModel.mask_y_bounds_full(pmask, orig_h)
            band_h = max(320, y_bot - y_top)
            y_top = max(0, min(y_top, orig_h - band_h))
            slice_boxes = get_slice_bboxes(
                image_height=band_h,
                image_width=orig_w,
                slice_height=band_h,
                slice_width=min(slice_w, orig_w),
                overlap_height_ratio=0.0,
                overlap_width_ratio=0.22,
            )
            for sx1, sy1, sx2, sy2 in slice_boxes:
                ay1, ay2 = y_top + sy1, y_top + sy2
                crop = frame[ay1:ay2, sx1:sx2]
                if crop.size == 0:
                    continue
                tile_imgs.append(crop)
                tile_meta.append((f_idx, sx1, ay1, pmask))
                if len(tile_imgs) >= tile_batch_size:
                    flush_tiles()

        flush_tiles()
        return {
            f_idx: self._merge_tile_candidates_nmm(cands, scale_ctx)
            for f_idx, cands in per_frame_raw.items()
        }

    def extract_candidates_and_motion(
        self,
        video_path: str,
        model: YOLO,
        tracks: dict,
        pitch_model: PitchColorModel,
        scale_ctx: FrameScaleContext,
        batch_size: int = 20,
    ) -> dict[str, Any]:
        """
        Option A Hybrid Pass 1:
        - Computes camera-motion step homographies and dynamic pitch masks across all frames.
        - Runs full-frame adaptive context inference on intermediate frames and SAHI pitch-band
          sliced inference every `sahi_stride` frames.
        """
        orig_w, orig_h = scale_ctx.frame_w, scale_ctx.frame_h
        flow_w = self.FLOW_WORK_W
        flow_h = max(90, int(round(flow_w * orig_h / max(1, orig_w))))
        full_imgsz = 640
        cached_raw_rows = tracks.get("raw_ball_rows", [])

        raw_candidates: list[list[dict[str, Any]]] = []
        step_homographies: list[np.ndarray] = []

        prev_gray_small: np.ndarray | None = None
        frame_idx = 0

        for frame_batch in iter_video_batches(video_path, batch_size=batch_size):
            need_infer_frames = [
                f
                for offset, f in enumerate(frame_batch)
                if (frame_idx + offset) % self.sahi_stride != 0 and (frame_idx + offset) >= len(cached_raw_rows)
            ]
            non_sahi_results = (
                iter(model.predict(need_infer_frames, imgsz=full_imgsz, conf=0.05, verbose=False))
                if need_infer_frames
                else iter(())
            )

            sahi_queue: list[tuple[int, np.ndarray, np.ndarray]] = []

            for offset, frame in enumerate(frame_batch):
                curr_idx = frame_idx + offset
                pmask = pitch_model.compute_playable_pitch_mask(frame, work_w=flow_w, erode_border_frac=0.016)

                gray_small = cv2.cvtColor(
                    cv2.resize(frame, (flow_w, flow_h), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )

                person_bboxes = []
                if curr_idx < len(tracks.get("players", [])):
                    person_bboxes.extend(p["bbox"] for p in tracks["players"][curr_idx].values())
                if curr_idx < len(tracks.get("referees", [])):
                    person_bboxes.extend(r["bbox"] for r in tracks["referees"][curr_idx].values())

                if prev_gray_small is None:
                    H_curr_to_prev = np.eye(3, dtype=np.float64)
                else:
                    H_curr_to_prev = self.estimate_step_homography(
                        prev_gray_small, gray_small, person_bboxes, orig_w, orig_h
                    )
                step_homographies.append(H_curr_to_prev)
                prev_gray_small = gray_small

                if curr_idx < len(cached_raw_rows) and cached_raw_rows[curr_idx]:
                    base_cands = self._filter_raw_ball_rows(
                        np.asarray(cached_raw_rows[curr_idx], dtype=np.float64), 0, 0, pmask, scale_ctx, "full_ctx"
                    )
                elif curr_idx % self.sahi_stride != 0 and need_infer_frames:
                    res = next(non_sahi_results)
                    cls_inv = {v: k for k, v in res.names.items()}
                    ball_cls = cls_inv["ball"]
                    boxes_np = res.boxes.cpu().numpy().data
                    base_cands = (
                        self._filter_raw_ball_rows(
                            boxes_np[boxes_np[:, 5].astype(int) == ball_cls], 0, 0, pmask, scale_ctx, "full_ctx"
                        )
                        if len(boxes_np) > 0
                        else []
                    )
                else:
                    base_cands = []

                if curr_idx % self.sahi_stride == 0:
                    sahi_queue.append((curr_idx, frame, pmask))
                    raw_candidates.append(base_cands)
                else:
                    raw_candidates.append(self._merge_tile_candidates_nmm(base_cands, scale_ctx))

            if sahi_queue:
                sahi_map = self._run_sahi_pitch_slices_on_frames(sahi_queue, model, scale_ctx)
                for s_idx, cands in sahi_map.items():
                    raw_candidates[s_idx] = self._merge_tile_candidates_nmm(raw_candidates[s_idx] + cands, scale_ctx)

            frame_idx += len(frame_batch)
            if frame_idx % 150 < len(frame_batch):
                print(f"Hybrid SAHI ball candidate & motion pass: {frame_idx} frames...")

        return {
            "version": 2,
            "orig_size": (orig_w, orig_h),
            "candidates": raw_candidates,
            "step_homographies": step_homographies,
            "zoom_enriched": False,
        }

    def _build_window_homographies(
        self,
        step_homographies: list[np.ndarray],
        ref_t: int,
        start_t: int,
        end_t: int,
    ) -> dict[int, np.ndarray]:
        """Compute homographies H_{s -> ref_t} for s in [start_t, end_t]."""
        transforms: dict[int, np.ndarray] = {ref_t: np.eye(3, dtype=np.float64)}

        H_acc = np.eye(3, dtype=np.float64)
        for s in range(ref_t + 1, end_t + 1):
            H_acc = H_acc @ step_homographies[s]
            transforms[s] = H_acc

        H_acc_back = np.eye(3, dtype=np.float64)
        for s in range(ref_t - 1, start_t - 1, -1):
            try:
                H_inv = np.linalg.inv(step_homographies[s + 1])
            except np.linalg.LinAlgError:
                H_inv = np.eye(3, dtype=np.float64)
            H_acc_back = H_acc_back @ H_inv
            transforms[s] = H_acc_back

        return transforms

    def suppress_static_landmarks(
        self,
        candidates: list[list[dict[str, Any]]],
        step_homographies: list[np.ndarray],
        tracks: dict,
        scale_ctx: FrameScaleContext,
    ) -> list[list[dict[str, Any]]]:
        """Suppress pitch-stationary false positives using purely relative H_p(y) and FPS-scaled gates."""
        num_frames = len(candidates)
        filtered: list[list[dict[str, Any]]] = [[] for _ in range(num_frames)]

        window_half = scale_ctx.frames_for_seconds(3.2, minimum=25)
        static_min_span = scale_ctx.frames_for_seconds(1.3, minimum=12)
        min_stationary_span = scale_ctx.frames_for_seconds(0.16, minimum=4)

        player_feet_per_frame: list[np.ndarray] = []
        for f in range(num_frames):
            feet = []
            if f < len(tracks.get("players", [])):
                for p in tracks["players"][f].values():
                    feet.append(get_foot_position(p["bbox"]))
            player_feet_per_frame.append(np.asarray(feet, dtype=np.float64) if feet else np.empty((0, 2)))

        for t in range(num_frames):
            if not candidates[t]:
                continue

            s0 = max(0, t - window_half)
            s1 = min(num_frames - 1, t + window_half)
            H_to_t = self._build_window_homographies(step_homographies, t, s0, s1)

            win_frames = []
            win_pts = []
            for s in range(s0, s1 + 1):
                if not candidates[s]:
                    continue
                pts_s = np.array([[c["cx"], c["cy"]] for c in candidates[s]], dtype=np.float64)
                warped = self._warp_points(H_to_t[s], pts_s)
                for pt in warped:
                    win_frames.append(s)
                    win_pts.append(pt)

            win_frames_arr = np.asarray(win_frames, dtype=np.int32)
            win_pts_arr = np.asarray(win_pts, dtype=np.float64) if win_pts else np.empty((0, 2))

            for cand in candidates[t]:
                pt = np.array([cand["cx"], cand["cy"]], dtype=np.float64)
                if len(win_pts_arr) == 0:
                    filtered[t].append(cand)
                    continue

                hp_t = scale_ctx.player_h_at(cand["cy"])
                static_r = 0.34 * hp_t

                dists = np.hypot(win_pts_arr[:, 0] - pt[0], win_pts_arr[:, 1] - pt[1])
                near_frames = np.unique(win_frames_arr[dists <= static_r])
                span = int(near_frames[-1] - near_frames[0]) if len(near_frames) > 1 else 0

                is_static_clutter = False
                if len(near_frames) >= 2 and span >= min_stationary_span:
                    feet_near_count = 0
                    coexisting_elsewhere_count = 0
                    for sf in near_frames:
                        try:
                            H_t_to_sf = np.linalg.inv(H_to_t[int(sf)])
                        except np.linalg.LinAlgError:
                            H_t_to_sf = np.eye(3, dtype=np.float64)
                        pt_sf = self._warp_points(H_t_to_sf, pt[None, :])[0]
                        hp_sf = scale_ctx.player_h_at(pt_sf[1])

                        feet_sf = player_feet_per_frame[int(sf)]
                        if len(feet_sf) > 0:
                            min_foot_d = float(np.min(np.hypot(feet_sf[:, 0] - pt_sf[0], feet_sf[:, 1] - pt_sf[1])))
                            if min_foot_d <= 1.20 * hp_sf:
                                feet_near_count += 1

                        for other in candidates[int(sf)]:
                            if other["conf"] >= 0.18 and np.hypot(other["cx"] - pt_sf[0], other["cy"] - pt_sf[1]) > 1.35 * hp_sf:
                                coexisting_elsewhere_count += 1
                                break

                    foot_ratio = feet_near_count / float(len(near_frames))
                    if feet_near_count == 0:
                        is_static_clutter = True
                    elif len(near_frames) >= 5 and span >= static_min_span:
                        if foot_ratio < 0.55 or coexisting_elsewhere_count >= 2:
                            is_static_clutter = True

                if not is_static_clutter:
                    filtered[t].append(cand)

        return filtered

    def link_trajectory_viterbi(
        self,
        candidates: list[list[dict[str, Any]]],
        step_homographies: list[np.ndarray],
        scale_ctx: FrameScaleContext,
    ) -> list[dict[str, Any] | None]:
        """Link kinematically consistent ball detections using relative H_p(y) and FPS-scaled velocity gates."""
        num_frames = len(candidates)
        if num_frames == 0:
            return []

        fps_scale = 25.0 / scale_ctx.fps
        max_link_gap = scale_ctx.frames_for_seconds(2.0, minimum=20)
        max_aerial_gap = scale_ctx.frames_for_seconds(6.0, minimum=60)
        tight_island_gap = scale_ctx.frames_for_seconds(0.32, minimum=5)

        nodes: list[tuple[int, dict[str, Any]]] = []
        frame_node_indices: list[list[int]] = [[] for _ in range(num_frames)]
        for t, cands in enumerate(candidates):
            for c in sorted(cands, key=lambda item: item["conf"], reverse=True)[:6]:
                idx = len(nodes)
                nodes.append((t, c))
                frame_node_indices[t].append(idx)

        if not nodes:
            return [None] * num_frames

        n_nodes = len(nodes)
        adj_prev: list[list[tuple[int, float, np.ndarray]]] = [[] for _ in range(n_nodes)]

        for t_v in range(num_frames):
            if not frame_node_indices[t_v]:
                continue
            s_min = max(0, t_v - max_link_gap)
            if s_min == t_v:
                continue

            H_to_tv = self._build_window_homographies(step_homographies, t_v, s_min, t_v)

            for v_idx in frame_node_indices[t_v]:
                _, c_v = nodes[v_idx]
                pt_v = np.array([c_v["cx"], c_v["cy"]], dtype=np.float64)
                hp_v = scale_ctx.player_h_at(c_v["cy"])

                max_step = 1.85 * hp_v * fps_scale
                gap_speed = 1.08 * hp_v * fps_scale
                max_cap = 0.62 * scale_ctx.frame_diag

                for t_u in range(t_v - 1, s_min - 1, -1):
                    if not frame_node_indices[t_u]:
                        continue
                    dt = t_v - t_u
                    max_dist = max_step if dt == 1 else min(max_cap, max_step + gap_speed * (dt - 1))

                    pts_u = np.array(
                        [[nodes[u][1]["cx"], nodes[u][1]["cy"]] for u in frame_node_indices[t_u]],
                        dtype=np.float64,
                    )
                    pts_u_in_tv = self._warp_points(H_to_tv[t_u], pts_u)

                    for k, u_idx in enumerate(frame_node_indices[t_u]):
                        disp = pt_v - pts_u_in_tv[k]
                        dist = float(np.hypot(disp[0], disp[1]))
                        if dist <= max_dist:
                            adj_prev[v_idx].append((u_idx, dist, disp / float(dt)))

        # Pass A: Prune isolated false-positive islands (< 3 frames unless high confidence)
        parent = list(range(n_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for v_idx in range(n_nodes):
            t_v, c_v = nodes[v_idx]
            hp_v = scale_ctx.player_h_at(c_v["cy"])
            step_lim = 1.85 * hp_v * fps_scale
            tight_spd = 0.90 * hp_v * fps_scale
            for u_idx, dist, _ in adj_prev[v_idx]:
                dt = t_v - nodes[u_idx][0]
                if dt <= tight_island_gap and dist <= (step_lim + tight_spd * (dt - 1)):
                    union(u_idx, v_idx)

        comp_members: dict[int, list[int]] = {}
        for i in range(n_nodes):
            comp_members.setdefault(find(i), []).append(i)

        valid_node = np.zeros(n_nodes, dtype=bool)
        for members in comp_members.values():
            unique_t = {nodes[m][0] for m in members}
            conf_sum = sum(nodes[m][1]["conf"] for m in members)
            if len(unique_t) >= 3 or (len(unique_t) >= 2 and conf_sum >= 0.65):
                for m in members:
                    valid_node[m] = True

        if not np.any(valid_node):
            valid_node[:] = True

        # Pass B: Global Viterbi DP normalized in player-height units
        dp_score = np.full(n_nodes, -1e9, dtype=np.float64)
        dp_back = np.full(n_nodes, -1, dtype=np.int32)
        dp_vel = np.zeros((n_nodes, 2), dtype=np.float64)

        for v_idx in range(n_nodes):
            if not valid_node[v_idx]:
                continue
            t_v, c_v = nodes[v_idx]
            hp_v = max(12.0, scale_ctx.player_h_at(c_v["cy"]))
            node_reward = 1.0 + 3.0 * c_v["conf"]

            best_score = node_reward - 0.04 * fps_scale * t_v
            best_prev = -1
            best_v = np.zeros(2, dtype=np.float64)

            for u_idx, dist, vel_uv in adj_prev[v_idx]:
                if not valid_node[u_idx] or dp_score[u_idx] < -1e8:
                    continue
                dt = t_v - nodes[u_idx][0]
                accel_cost = 0.0
                if dp_back[u_idx] != -1:
                    dv_norm = float(np.hypot(*(vel_uv - dp_vel[u_idx]))) / (hp_v * fps_scale)
                    accel_cost = min(2.2, 1.05 * dv_norm)

                dist_norm = dist / hp_v
                dist_cost = 0.50 * (dist_norm / max(1.0, np.sqrt(dt * fps_scale)))
                gap_cost = 0.18 * fps_scale * (dt - 1)
                cand_score = dp_score[u_idx] + node_reward - dist_cost - gap_cost - accel_cost

                if cand_score > best_score:
                    best_score = cand_score
                    best_prev = u_idx
                    best_v = vel_uv

            if best_prev == -1 and t_v > 0:
                aerial_spd = 0.52 * hp_v * fps_scale
                for u_idx in range(v_idx - 1, -1, -1):
                    if not valid_node[u_idx] or dp_score[u_idx] < -1e8:
                        continue
                    dt = t_v - nodes[u_idx][0]
                    if dt <= max_link_gap:
                        continue
                    if dt > max_aerial_gap:
                        break
                    raw_d = float(np.hypot(c_v["cx"] - nodes[u_idx][1]["cx"], c_v["cy"] - nodes[u_idx][1]["cy"]))
                    if raw_d <= aerial_spd * dt:
                        cand_score = dp_score[u_idx] + node_reward - 0.28 * fps_scale * dt - 0.18 * (raw_d / hp_v)
                        if cand_score > best_score:
                            best_score = cand_score
                            best_prev = u_idx
                            best_v = np.zeros(2, dtype=np.float64)

            dp_score[v_idx] = best_score
            dp_back[v_idx] = best_prev
            dp_vel[v_idx] = best_v

        end_scores = np.array(
            [
                dp_score[i] - 0.04 * fps_scale * (num_frames - 1 - nodes[i][0]) if valid_node[i] else -1e9
                for i in range(n_nodes)
            ]
        )
        curr = int(np.argmax(end_scores))
        selected: list[dict[str, Any] | None] = [None] * num_frames
        while curr != -1 and dp_score[curr] > -1e8:
            t_c, cand_c = nodes[curr]
            selected[t_c] = cand_c
            curr = int(dp_back[curr])

        return selected

    def enrich_gaps_with_kalman_roi(
        self,
        video_path: str,
        model: YOLO,
        cand_data: dict[str, Any],
        selected_track: list[dict[str, Any] | None],
        pitch_model: PitchColorModel,
        scale_ctx: FrameScaleContext,
        batch_size: int = 20,
    ) -> None:
        """
        Option A Pass 2:
        On intermediate non-SAHI frames and trajectory gaps where the ball was missed or motion-blurred,
        run native 1:1 high-res ROI tile inference centered on camera-stabilized predictions!
        """
        orig_w, orig_h = scale_ctx.frame_w, scale_ctx.frame_h
        candidates = cand_data["candidates"]
        step_H = cand_data["step_homographies"]
        max_zoom_gap = scale_ctx.frames_for_seconds(2.6, minimum=30)

        crop_size = max(320, min(orig_h, int(round((0.46 * orig_h) / 32.0)) * 32))
        half = crop_size // 2

        hit_indices = [i for i, c in enumerate(selected_track) if c is not None]
        if len(hit_indices) < 2:
            cand_data["zoom_enriched"] = True
            return

        # Camera-compensated trajectory bridge across genuine multi-frame gaps (gap >= 3 frames)
        gap_targets: dict[int, tuple[float, float]] = {}
        for i0, i1 in zip(hit_indices, hit_indices[1:]):
            gap = i1 - i0 - 1
            if 3 <= gap <= max_zoom_gap:
                H_to_i1 = self._build_window_homographies(step_H, i1, i0, i1)
                c0, c1 = selected_track[i0], selected_track[i1]
                p0_in_i1 = self._warp_points(H_to_i1[i0], np.array([[c0["cx"], c0["cy"]]]))[0]
                p1_in_i1 = np.array([c1["cx"], c1["cy"]], dtype=np.float64)
                for f in range(i0 + 1, i1):
                    alpha = (f - i0) / float(i1 - i0)
                    pf_in_i1 = (1.0 - alpha) * p0_in_i1 + alpha * p1_in_i1
                    try:
                        H_i1_to_f = np.linalg.inv(H_to_i1[f])
                    except np.linalg.LinAlgError:
                        H_i1_to_f = np.eye(3, dtype=np.float64)
                    pf = self._warp_points(H_i1_to_f, pf_in_i1[None, :])[0]
                    gap_targets[f] = (float(pf[0]), float(pf[1]))

        if not gap_targets:
            cand_data["zoom_enriched"] = True
            return

        print(f"Running Kalman-guided native {crop_size}x{crop_size} ROI tile pass on {len(gap_targets)} gap frames...")
        cap = cv2.VideoCapture(video_path)
        pending_frames: list[int] = []
        pending_crops: list[np.ndarray] = []
        pending_offsets: list[tuple[int, int]] = []
        pending_masks: list[np.ndarray] = []

        def flush_batch() -> None:
            if not pending_crops:
                return
            res_list = model.predict(pending_crops, imgsz=crop_size, conf=0.05, verbose=False)
            for f_idx, (x0, y0), pmask, res in zip(pending_frames, pending_offsets, pending_masks, res_list):
                cls_inv = {v: k for k, v in res.names.items()}
                ball_cls = cls_inv["ball"]
                boxes_np = res.boxes.cpu().numpy().data
                if len(boxes_np) > 0:
                    ball_rows = boxes_np[boxes_np[:, 5].astype(int) == ball_cls]
                    new_cands = self._filter_raw_ball_rows(ball_rows, x0, y0, pmask, scale_ctx, "kalman_roi")
                    candidates[f_idx] = self._merge_tile_candidates_nmm(candidates[f_idx] + new_cands, scale_ctx)
            pending_frames.clear()
            pending_crops.clear()
            pending_offsets.clear()
            pending_masks.clear()

        f_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if f_idx in gap_targets:
                    px, py = gap_targets[f_idx]
                    x0 = int(np.clip(round(px) - half, 0, max(0, orig_w - crop_size)))
                    y0 = int(np.clip(round(py) - half, 0, max(0, orig_h - crop_size)))
                    crop = frame[y0 : y0 + crop_size, x0 : x0 + crop_size].copy()
                    pmask = pitch_model.compute_playable_pitch_mask(frame, work_w=self.FLOW_WORK_W, erode_border_frac=0.016)

                    pending_frames.append(f_idx)
                    pending_crops.append(crop)
                    pending_offsets.append((x0, y0))
                    pending_masks.append(pmask)
                    if len(pending_crops) >= batch_size:
                        flush_batch()
                f_idx += 1
            flush_batch()
        finally:
            cap.release()

        cand_data["zoom_enriched"] = True

    def kalman_rts_smooth_trajectory(
        self,
        selected_track: list[dict[str, Any] | None],
        step_homographies: list[np.ndarray],
        scale_ctx: FrameScaleContext,
    ) -> list[dict]:
        """
        Camera-motion-stabilized 4-state Constant-Velocity Kalman Filter ([x, y, vx, vy])
        followed by Rauch-Tung-Striebel (RTS) backward smoothing across 100% of frames.
        """
        num_frames = len(selected_track)
        if num_frames == 0:
            return []

        valid_indices = [i for i, item in enumerate(selected_track) if item is not None]
        if not valid_indices:
            return [{} for _ in range(num_frames)]

        first_valid = valid_indices[0]
        first_cand = selected_track[first_valid]
        hp_init = scale_ctx.player_h_at(first_cand["cy"])
        fps_scale = 25.0 / scale_ctx.fps
        drag = 0.985

        # Forward Kalman Storage
        x_pred = np.zeros((num_frames, 4), dtype=np.float64)
        P_pred = np.zeros((num_frames, 4, 4), dtype=np.float64)
        x_filt = np.zeros((num_frames, 4), dtype=np.float64)
        P_filt = np.zeros((num_frames, 4, 4), dtype=np.float64)
        F_eff = np.zeros((num_frames, 4, 4), dtype=np.float64)

        # Propagate first valid detection backward to frame 0 via camera motion for clean initialization
        p_init = np.array([first_cand["cx"], first_cand["cy"]], dtype=np.float64)
        for s in range(first_valid, 0, -1):
            p_init = self._warp_points(step_homographies[s], p_init[None, :])[0]

        x_curr = np.array([p_init[0], p_init[1], 0.0, 0.0], dtype=np.float64)
        P_curr = np.diag([(0.5 * hp_init) ** 2, (0.5 * hp_init) ** 2, (0.8 * hp_init) ** 2, (0.8 * hp_init) ** 2])
        H_obs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        for t in range(num_frames):
            if t == 0:
                F_t = np.eye(4, dtype=np.float64)
                xp = x_curr.copy()
                Pp = P_curr.copy()
            else:
                try:
                    H_prev_to_curr = np.linalg.inv(step_homographies[t])
                except np.linalg.LinAlgError:
                    H_prev_to_curr = np.eye(3, dtype=np.float64)

                # Warp previous position and velocity endpoint into frame t's camera coordinates
                pts_prev = np.array(
                    [
                        [x_curr[0], x_curr[1]],
                        [x_curr[0] + x_curr[2], x_curr[1] + x_curr[3]],
                        [x_curr[0] + 1.0, x_curr[1]],
                        [x_curr[0], x_curr[1] + 1.0],
                    ],
                    dtype=np.float64,
                )
                warped = self._warp_points(H_prev_to_curr, pts_prev)
                p_w = warped[0]
                v_w = warped[1] - p_w
                # Local 2x2 spatial Jacobian of H_prev_to_curr
                J = np.column_stack([warped[2] - p_w, warped[3] - p_w])

                F_kin = np.array(
                    [
                        [1.0, 0.0, drag, 0.0],
                        [0.0, 1.0, 0.0, drag],
                        [0.0, 0.0, drag, 0.0],
                        [0.0, 0.0, 0.0, drag],
                    ],
                    dtype=np.float64,
                )
                J4 = np.zeros((4, 4), dtype=np.float64)
                J4[:2, :2] = J
                J4[2:, 2:] = J
                F_t = F_kin @ J4

                xp = np.array([p_w[0] + drag * v_w[0], p_w[1] + drag * v_w[1], drag * v_w[0], drag * v_w[1]], dtype=np.float64)
                hp_loc = scale_ctx.player_h_at(xp[1])
                sigma_pos = 0.07 * hp_loc * fps_scale
                sigma_vel = 0.16 * hp_loc * fps_scale
                Q_t = np.diag([sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2])
                Pp = F_t @ P_curr @ F_t.T + Q_t

            F_eff[t] = F_t
            x_pred[t] = xp
            P_pred[t] = Pp

            cand = selected_track[t]
            if cand is not None:
                z = np.array([cand["cx"], cand["cy"]], dtype=np.float64)
                hp_z = scale_ctx.player_h_at(cand["cy"])
                sigma_r = (0.075 * hp_z) / np.sqrt(max(0.15, float(cand["conf"])))
                R_t = np.eye(2, dtype=np.float64) * (sigma_r**2)

                innov = z - H_obs @ xp
                S_mat = H_obs @ Pp @ H_obs.T + R_t
                K_gain = Pp @ H_obs.T @ np.linalg.inv(S_mat)
                I_KH = np.eye(4, dtype=np.float64) - K_gain @ H_obs

                x_curr = xp + K_gain @ innov
                # Joseph-form stabilized covariance update
                P_curr = I_KH @ Pp @ I_KH.T + K_gain @ R_t @ K_gain.T
            else:
                x_curr = xp
                P_curr = Pp

            x_filt[t] = x_curr
            P_filt[t] = P_curr

        # Backward Rauch-Tung-Striebel (RTS) Smoother Pass
        x_smooth = x_filt.copy()
        for t in range(num_frames - 2, -1, -1):
            try:
                P_pred_inv = np.linalg.pinv(P_pred[t + 1])
                G_t = P_filt[t] @ F_eff[t + 1].T @ P_pred_inv
                x_smooth[t] = x_filt[t] + G_t @ (x_smooth[t + 1] - x_pred[t + 1])
            except np.linalg.LinAlgError:
                pass

        # Interpolate bounding-box sizes smoothly from observed anchors (or local perspective ball scale)
        obs_w = np.array([selected_track[i]["bbox"][2] - selected_track[i]["bbox"][0] for i in valid_indices], dtype=np.float64)
        obs_h = np.array([selected_track[i]["bbox"][3] - selected_track[i]["bbox"][1] for i in valid_indices], dtype=np.float64)
        all_idx = np.arange(num_frames, dtype=np.float64)
        valid_arr = np.asarray(valid_indices, dtype=np.float64)
        widths = np.interp(all_idx, valid_arr, obs_w)
        heights = np.interp(all_idx, valid_arr, obs_h)

        output_ball_tracks: list[dict] = []
        for t in range(num_frames):
            cx = float(np.clip(x_smooth[t, 0], 0.0, scale_ctx.frame_w - 1.0))
            cy = float(np.clip(x_smooth[t, 1], 0.0, scale_ctx.frame_h - 1.0))
            bw, bh = float(widths[t]), float(heights[t])
            output_ball_tracks.append(
                {
                    1: {
                        "bbox": [cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh],
                    }
                }
            )

        return output_ball_tracks

    def track_ball(
        self,
        video_path: str,
        model: YOLO,
        tracks: dict,
        pitch_model: PitchColorModel | None = None,
        scale_ctx: FrameScaleContext | None = None,
        read_from_stub: bool = True,
        candidates_stub_path: str = "stubs/ball_candidates_stub.pkl",
    ) -> list[dict]:
        """End-to-end relative-scale SAHI + Kalman/RTS ball tracking pipeline."""
        props = get_video_properties(video_path)
        if pitch_model is None:
            pitch_model = PitchColorModel.fit_from_video(video_path, read_from_stub=False)
        if scale_ctx is None:
            scale_ctx = FrameScaleContext(props["width"], props["height"], props["fps"]).calibrate_from_tracks(tracks)

        cand_data = None
        if read_from_stub and candidates_stub_path and os.path.exists(candidates_stub_path):
            with open(candidates_stub_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and loaded.get("version") == 2 and len(loaded.get("candidates", [])) == props["frame_count"]:
                cand_data = loaded

        if cand_data is None:
            cand_data = self.extract_candidates_and_motion(
                video_path, model, tracks, pitch_model, scale_ctx
            )
            if candidates_stub_path:
                os.makedirs(os.path.dirname(candidates_stub_path) or ".", exist_ok=True)
                with open(candidates_stub_path, "wb") as f:
                    pickle.dump(cand_data, f)

        clean_cands = self.suppress_static_landmarks(
            cand_data["candidates"], cand_data["step_homographies"], tracks, scale_ctx
        )
        initial_selected = self.link_trajectory_viterbi(clean_cands, cand_data["step_homographies"], scale_ctx)

        if not cand_data.get("zoom_enriched", False):
            self.enrich_gaps_with_kalman_roi(
                video_path, model, cand_data, initial_selected, pitch_model, scale_ctx
            )
            if candidates_stub_path:
                os.makedirs(os.path.dirname(candidates_stub_path) or ".", exist_ok=True)
                with open(candidates_stub_path, "wb") as f:
                    pickle.dump(cand_data, f)

            clean_cands = self.suppress_static_landmarks(
                cand_data["candidates"], cand_data["step_homographies"], tracks, scale_ctx
            )
            final_selected = self.link_trajectory_viterbi(clean_cands, cand_data["step_homographies"], scale_ctx)
        else:
            final_selected = initial_selected

        return self.kalman_rts_smooth_trajectory(final_selected, cand_data["step_homographies"], scale_ctx)
