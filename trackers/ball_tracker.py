import os
import pickle
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from utils import get_foot_position, iter_video_batches


class BallTracker:
    """Physics- and camera-motion-aware ball tracker for monocular broadcast football footage.

    Stages:
      1. HSV playable-pitch convex-hull mask (rejects touchline spare balls & stadium clutter).
      2. Camera-motion-compensated static landmark suppression (eliminates penalty spot & corner markings).
      3. Global Viterbi / Dynamic-Programming trajectory linking + short-island pruning (blocks white-boot teleports).
      4. Targeted native 1:1 640x640 ROI crop zoom on trajectory gaps + 100%-coverage smooth interpolation.
    """

    MASK_W = 480
    MASK_H = 270

    def __init__(
        self,
        max_step_px: float = 78.0,
        gap_speed_px: float = 46.0,
        max_link_gap: int = 50,
        static_radius_px: float = 18.0,
        static_min_span: int = 35,
    ) -> None:
        self.max_step_px = max_step_px
        self.gap_speed_px = gap_speed_px
        self.max_link_gap = max_link_gap
        self.static_radius_px = static_radius_px
        self.static_min_span = static_min_span

    @classmethod
    def compute_playable_pitch_mask(cls, frame: np.ndarray) -> np.ndarray:
        """Compute a 480x270 binary mask of playable grass, eroded along stadium advertising boards."""
        small = cv2.resize(frame, (cls.MASK_W, cls.MASK_H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Broadcast grass hue range
        grass = cv2.inRange(hsv, np.array([28, 35, 35], dtype=np.uint8), np.array([92, 255, 255], dtype=np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, kernel)
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones((cls.MASK_H, cls.MASK_W), dtype=np.uint8) * 255

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

        # Pad outer image borders before eroding so only internal stadium/board boundaries shrink inward,
        # not playable grass extending off the left/right/bottom screen edges.
        pad = 10
        padded = np.zeros((cls.MASK_H + 2 * pad, cls.MASK_W + 2 * pad), dtype=np.uint8)
        hull_shifted = hull + np.array([[[pad, pad]]], dtype=np.int32)
        cv2.fillConvexPoly(padded, hull_shifted, 255)

        # Extend mask across actual image edges where hull already touches the frame boundary
        inner = padded[pad : pad + cls.MASK_H, pad : pad + cls.MASK_W]
        if np.any(inner[0, :] > 0):
            padded[:pad, pad : pad + cls.MASK_W] = inner[0:1, :]
        if np.any(inner[-1, :] > 0):
            padded[pad + cls.MASK_H :, pad : pad + cls.MASK_W] = inner[-1:, :]
        if np.any(inner[:, 0] > 0):
            padded[pad : pad + cls.MASK_H, :pad] = inner[:, 0:1]
        if np.any(inner[:, -1] > 0):
            padded[pad : pad + cls.MASK_H, pad + cls.MASK_W :] = inner[:, -1:]

        # Erode ~7px at 480x270 (~28px at 1920x1080) to trim grass apron next to LED perimeter boards
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eroded = cv2.erode(padded, erode_kernel, iterations=1)
        return eroded[pad : pad + cls.MASK_H, pad : pad + cls.MASK_W]

    @classmethod
    def is_inside_pitch(cls, mask_small: np.ndarray, cx: float, cy: float, orig_w: int, orig_h: int) -> bool:
        """Return True if full-res point (cx, cy) falls inside the downsampled playable pitch mask."""
        sx = int(np.clip(round(cx * cls.MASK_W / orig_w), 0, cls.MASK_W - 1))
        sy = int(np.clip(round(cy * cls.MASK_H / orig_h), 0, cls.MASK_H - 1))
        return bool(mask_small[sy, sx] > 0)

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
        scale_x = cls.MASK_W / orig_w
        scale_y = cls.MASK_H / orig_h

        bg_mask = np.ones((cls.MASK_H, cls.MASK_W), dtype=np.uint8) * 255
        for bbox in person_bboxes_full:
            x1 = max(0, int(bbox[0] * scale_x) - 3)
            y1 = max(0, int(bbox[1] * scale_y) - 3)
            x2 = min(cls.MASK_W, int(bbox[2] * scale_x) + 3)
            y2 = min(cls.MASK_H, int(bbox[3] * scale_y) + 3)
            bg_mask[y1:y2, x1:x2] = 0

        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray_small,
            maxCorners=350,
            qualityLevel=0.01,
            minDistance=8,
            mask=bg_mask,
        )
        if pts_prev is None or len(pts_prev) < 12:
            return np.eye(3, dtype=np.float64)

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_small,
            curr_gray_small,
            pts_prev,
            None,
            winSize=(15, 15),
            maxLevel=3,
        )
        if pts_curr is None or status is None:
            return np.eye(3, dtype=np.float64)

        good = status.ravel() == 1
        src = pts_curr[good].reshape(-1, 2)
        dst = pts_prev[good].reshape(-1, 2)
        if len(src) < 10:
            return np.eye(3, dtype=np.float64)

        H_small, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=2.0)
        if H_small is None:
            return np.eye(3, dtype=np.float64)

        # Convert homography from small (480x270) space to full (orig_w x orig_h) space: H_full = S^{-1} H_small S
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

    def extract_candidates_and_motion(
        self,
        video_path: str,
        model: YOLO,
        tracks: dict,
        batch_size: int = 20,
    ) -> dict[str, Any]:
        """Single streaming pass to extract multi-candidate ball boxes (conf >= 0.05), pitch masks, and camera motion."""
        cap_info = cv2.VideoCapture(video_path)
        orig_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()

        raw_candidates: list[list[dict[str, Any]]] = []
        step_homographies: list[np.ndarray] = []  # H[t]: maps frame t -> frame t-1 (H[0] = I)

        prev_gray_small: np.ndarray | None = None
        frame_idx = 0

        for frame_batch in iter_video_batches(video_path, batch_size=batch_size):
            results_batch = model.predict(frame_batch, imgsz=640, conf=0.05, verbose=False)

            for frame, res in zip(frame_batch, results_batch):
                cls_inv = {v: k for k, v in res.names.items()}
                ball_cls = cls_inv["ball"]

                pitch_mask = self.compute_playable_pitch_mask(frame)
                gray_small = cv2.cvtColor(
                    cv2.resize(frame, (self.MASK_W, self.MASK_H), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )

                person_bboxes = []
                if frame_idx < len(tracks.get("players", [])):
                    person_bboxes.extend(p["bbox"] for p in tracks["players"][frame_idx].values())
                if frame_idx < len(tracks.get("referees", [])):
                    person_bboxes.extend(r["bbox"] for r in tracks["referees"][frame_idx].values())

                if prev_gray_small is None:
                    H_curr_to_prev = np.eye(3, dtype=np.float64)
                else:
                    H_curr_to_prev = self.estimate_step_homography(
                        prev_gray_small, gray_small, person_bboxes, orig_w, orig_h
                    )
                step_homographies.append(H_curr_to_prev)
                prev_gray_small = gray_small

                boxes_np = res.boxes.cpu().numpy().data
                frame_cands = []
                if len(boxes_np) > 0:
                    ball_rows = boxes_np[boxes_np[:, 5].astype(int) == ball_cls]
                    for row in ball_rows:
                        x1, y1, x2, y2, conf, _ = row.tolist()
                        w = x2 - x1
                        h = y2 - y1
                        # Physical football size & aspect-ratio sanity check on 1080p broadcast
                        if w < 4 or h < 4 or w > 55 or h > 55 or max(w / max(h, 1e-3), h / max(w, 1e-3)) > 2.4:
                            continue
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        if not self.is_inside_pitch(pitch_mask, cx, cy, orig_w, orig_h):
                            continue
                        frame_cands.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "cx": cx,
                                "cy": cy,
                                "conf": float(conf),
                                "source": "full640",
                            }
                        )

                raw_candidates.append(frame_cands)
                frame_idx += 1
                if frame_idx % 150 == 0:
                    print(f"Ball candidate & camera-motion pass: {frame_idx} frames...")

        return {
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

        # Forward from ref_t + 1 .. end_t: H_{s -> ref_t} = H_{s-1 -> ref_t} @ H_{s -> s-1}
        H_acc = np.eye(3, dtype=np.float64)
        for s in range(ref_t + 1, end_t + 1):
            H_acc = H_acc @ step_homographies[s]
            transforms[s] = H_acc

        # Backward from ref_t - 1 .. start_t: H_{s -> ref_t} = H_{s+1 -> ref_t} @ inv(H_{s+1 -> s})
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
        window_half: int = 90,
    ) -> list[list[dict[str, Any]]]:
        """Suppress painted pitch markings (penalty spot, corner spot/arc) that remain fixed in pitch space."""
        num_frames = len(candidates)
        filtered: list[list[dict[str, Any]]] = [[] for _ in range(num_frames)]

        # Precompute player foot coordinates per frame
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

            # Gather all projected detections in [s0, s1] into frame t's stabilized coordinate frame
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

                dists = np.hypot(win_pts_arr[:, 0] - pt[0], win_pts_arr[:, 1] - pt[1])
                near_mask = dists <= self.static_radius_px
                near_frames = np.unique(win_frames_arr[near_mask])
                span = int(near_frames[-1] - near_frames[0]) if len(near_frames) > 1 else 0

                is_static_clutter = False
                # Case 1: Short or long stationary object in open grass with NO player nearby (Newton's 1st Law:
                # spare touchline balls, corner flags, corner arcs, and unguarded penalty spots never move on the pitch)
                if len(near_frames) >= 2 and span >= 4:
                    feet_near_count = 0
                    coexisting_elsewhere_count = 0
                    for sf in near_frames:
                        try:
                            H_t_to_sf = np.linalg.inv(H_to_t[int(sf)])
                        except np.linalg.LinAlgError:
                            H_t_to_sf = np.eye(3, dtype=np.float64)
                        pt_sf = self._warp_points(H_t_to_sf, pt[None, :])[0]

                        feet_sf = player_feet_per_frame[int(sf)]
                        if len(feet_sf) > 0:
                            min_foot_d = float(np.min(np.hypot(feet_sf[:, 0] - pt_sf[0], feet_sf[:, 1] - pt_sf[1])))
                            if min_foot_d <= 65.0:
                                feet_near_count += 1

                        for other in candidates[int(sf)]:
                            if other["conf"] >= 0.18 and np.hypot(other["cx"] - pt_sf[0], other["cy"] - pt_sf[1]) > 70.0:
                                coexisting_elsewhere_count += 1
                                break

                    foot_ratio = feet_near_count / len(near_frames)
                    if feet_near_count == 0:
                        # Completely stationary on the pitch for >= 4 frames with zero players within ~4m
                        is_static_clutter = True
                    elif len(near_frames) >= 5 and span >= self.static_min_span:
                        # Persistent pitch landmark (e.g. penalty spot when a player briefly jogs near it)
                        if foot_ratio < 0.55 or coexisting_elsewhere_count >= 2:
                            is_static_clutter = True

                if not is_static_clutter:
                    filtered[t].append(cand)

        return filtered

    def link_trajectory_viterbi(
        self,
        candidates: list[list[dict[str, Any]]],
        step_homographies: list[np.ndarray],
    ) -> list[dict[str, Any] | None]:
        """Extract kinematically consistent ball tracklets via Dynamic Programming and stitch confirmed segments."""
        num_frames = len(candidates)
        if num_frames == 0:
            return []

        # Flatten all candidate nodes: (frame_t, cand_dict)
        nodes: list[tuple[int, dict[str, Any]]] = []
        frame_node_indices: list[list[int]] = [[] for _ in range(num_frames)]
        for t, cands in enumerate(candidates):
            # Keep top-6 candidates per frame sorted by confidence
            top_cands = sorted(cands, key=lambda c: c["conf"], reverse=True)[:6]
            for c in top_cands:
                idx = len(nodes)
                nodes.append((t, c))
                frame_node_indices[t].append(idx)

        if not nodes:
            return [None] * num_frames

        # Precompute local forward homographies H_{u_t -> v_t} on demand via cached cumulative lookback
        # First, extract locally connected high-confidence tracklets using greedy/DP chaining, prune <=3-frame islands,
        # then run global Viterbi over surviving nodes so isolated white-boot spikes never hijack long gaps.
        n_nodes = len(nodes)
        adj_prev: list[list[tuple[int, float, np.ndarray]]] = [[] for _ in range(n_nodes)]

        for t_v in range(num_frames):
            if not frame_node_indices[t_v]:
                continue
            s_min = max(0, t_v - self.max_link_gap)
            if s_min == t_v:
                continue

            H_to_tv = self._build_window_homographies(step_homographies, t_v, s_min, t_v)

            for v_idx in frame_node_indices[t_v]:
                _, c_v = nodes[v_idx]
                pt_v = np.array([c_v["cx"], c_v["cy"]], dtype=np.float64)

                for t_u in range(t_v - 1, s_min - 1, -1):
                    if not frame_node_indices[t_u]:
                        continue
                    dt = t_v - t_u
                    max_dist = self.max_step_px if dt == 1 else min(650.0, self.max_step_px + self.gap_speed_px * (dt - 1))

                    pts_u = np.array([[nodes[u][1]["cx"], nodes[u][1]["cy"]] for u in frame_node_indices[t_u]], dtype=np.float64)
                    pts_u_in_tv = self._warp_points(H_to_tv[t_u], pts_u)

                    for k, u_idx in enumerate(frame_node_indices[t_u]):
                        disp = pt_v - pts_u_in_tv[k]
                        dist = float(np.hypot(disp[0], disp[1]))
                        if dist <= max_dist:
                            vel = disp / float(dt)
                            adj_prev[v_idx].append((u_idx, dist, vel))

        # Pass A: Prune isolated false-positive islands (components with < 4 detections in tight <= 8-frame links)
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
            t_v = nodes[v_idx][0]
            for u_idx, dist, _ in adj_prev[v_idx]:
                dt = t_v - nodes[u_idx][0]
                tight_limit = self.max_step_px if dt == 1 else (self.max_step_px + 38.0 * (dt - 1))
                if dt <= 8 and dist <= tight_limit:
                    union(u_idx, v_idx)

        comp_members: dict[int, list[int]] = {}
        for i in range(n_nodes):
            comp_members.setdefault(find(i), []).append(i)

        valid_node = np.zeros(n_nodes, dtype=bool)
        for members in comp_members.values():
            unique_t = {nodes[m][0] for m in members}
            conf_sum = sum(nodes[m][1]["conf"] for m in members)
            # Keep tracklets with >= 4 frames, OR >= 2 very high-confidence frames
            if len(unique_t) >= 4 or (len(unique_t) >= 2 and conf_sum >= 0.85):
                for m in members:
                    valid_node[m] = True

        if not np.any(valid_node):
            valid_node[:] = True

        # Pass B: Global Viterbi DP over valid nodes with constant-velocity smoothness reward
        dp_score = np.full(n_nodes, -1e9, dtype=np.float64)
        dp_back = np.full(n_nodes, -1, dtype=np.int32)
        dp_vel = np.zeros((n_nodes, 2), dtype=np.float64)

        for v_idx in range(n_nodes):
            if not valid_node[v_idx]:
                continue
            t_v, c_v = nodes[v_idx]
            node_reward = 1.0 + 3.0 * c_v["conf"]

            # Base score if starting track here (small penalty for starting late so continuous paths win)
            best_score = node_reward - 0.04 * t_v
            best_prev = -1
            best_v = np.zeros(2, dtype=np.float64)

            for u_idx, dist, vel_uv in adj_prev[v_idx]:
                if not valid_node[u_idx] or dp_score[u_idx] < -1e8:
                    continue
                t_u = nodes[u_idx][0]
                dt = t_v - t_u

                # Velocity consistency cost (in camera-compensated coordinates)
                prev_v = dp_vel[u_idx]
                accel_cost = 0.0
                if dp_back[u_idx] != -1:
                    dv = float(np.hypot(vel_uv[0] - prev_v[0], vel_uv[1] - prev_v[1]))
                    # Allow kicks/bounces, penalize erratic zig-zags
                    accel_cost = min(2.2, 0.025 * dv)

                dist_cost = 0.012 * (dist / max(1.0, np.sqrt(dt)))
                gap_cost = 0.18 * (dt - 1)
                cand_score = dp_score[u_idx] + node_reward - dist_cost - gap_cost - accel_cost

                if cand_score > best_score:
                    best_score = cand_score
                    best_prev = u_idx
                    best_v = vel_uv

            # Also allow bridging longer aerial clearances (> max_link_gap up to 130 frames) from earlier confirmed tracks
            if best_prev == -1 and t_v > 0:
                for u_idx in range(v_idx - 1, -1, -1):
                    if not valid_node[u_idx] or dp_score[u_idx] < -1e8:
                        continue
                    dt = t_v - nodes[u_idx][0]
                    if dt <= self.max_link_gap:
                        continue
                    if dt > 150:
                        break
                    # Image-space sanity speed check for long aerial clearances
                    raw_d = float(
                        np.hypot(c_v["cx"] - nodes[u_idx][1]["cx"], c_v["cy"] - nodes[u_idx][1]["cy"])
                    )
                    if raw_d <= 22.0 * dt:
                        cand_score = dp_score[u_idx] + node_reward - 0.28 * dt - 0.004 * raw_d
                        if cand_score > best_score:
                            best_score = cand_score
                            best_prev = u_idx
                            best_v = np.zeros(2, dtype=np.float64)

            dp_score[v_idx] = best_score
            dp_back[v_idx] = best_prev
            dp_vel[v_idx] = best_v

        # Backtrack from highest-scoring end node (penalizing early termination slightly)
        end_scores = np.array(
            [
                dp_score[i] - 0.04 * (num_frames - 1 - nodes[i][0]) if valid_node[i] else -1e9
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

    def enrich_gaps_with_roi_zoom(
        self,
        video_path: str,
        model: YOLO,
        cand_data: dict[str, Any],
        selected_track: list[dict[str, Any] | None],
        max_zoom_gap: int = 65,
        crop_size: int = 640,
        batch_size: int = 20,
    ) -> None:
        """Run native 1:1 640x640 YOLO inference along trajectory gaps to recover small/occluded balls."""
        orig_w, orig_h = cand_data["orig_size"]
        candidates = cand_data["candidates"]
        num_frames = len(selected_track)

        hit_indices = [i for i, c in enumerate(selected_track) if c is not None]
        if len(hit_indices) < 2:
            cand_data["zoom_enriched"] = True
            return

        # Map frame_idx -> predicted (cx, cy) for frames inside moderate gaps
        gap_targets: dict[int, tuple[float, float]] = {}
        for i0, i1 in zip(hit_indices, hit_indices[1:]):
            gap = i1 - i0 - 1
            if 1 <= gap <= max_zoom_gap:
                c0, c1 = selected_track[i0], selected_track[i1]
                for f in range(i0 + 1, i1):
                    alpha = (f - i0) / float(i1 - i0)
                    px = (1.0 - alpha) * c0["cx"] + alpha * c1["cx"]
                    py = (1.0 - alpha) * c0["cy"] + alpha * c1["cy"]
                    gap_targets[f] = (px, py)

        if not gap_targets:
            cand_data["zoom_enriched"] = True
            return

        print(f"Running native 1:1 {crop_size}x{crop_size} ROI zoom on {len(gap_targets)} gap frames...")
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
                if len(boxes_np) == 0:
                    continue
                for row in boxes_np[boxes_np[:, 5].astype(int) == ball_cls]:
                    bx1, by1, bx2, by2, conf, _ = row.tolist()
                    x1, y1, x2, y2 = bx1 + x0, by1 + y0, bx2 + x0, by2 + y0
                    w, h = x2 - x1, y2 - y1
                    if w < 4 or h < 4 or w > 55 or h > 55 or max(w / max(h, 1e-3), h / max(w, 1e-3)) > 2.4:
                        continue
                    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                    if not self.is_inside_pitch(pmask, cx, cy, orig_w, orig_h):
                        continue
                    # Avoid duplicating an existing candidate within 8px
                    if any(np.hypot(c["cx"] - cx, c["cy"] - cy) < 8.0 and c["conf"] >= conf for c in candidates[f_idx]):
                        continue
                    candidates[f_idx].append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "cx": cx,
                            "cy": cy,
                            "conf": float(conf),
                            "source": "zoom640",
                        }
                    )
            pending_frames.clear()
            pending_crops.clear()
            pending_offsets.clear()
            pending_masks.clear()

        half = crop_size // 2
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
                    pmask = self.compute_playable_pitch_mask(frame)

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

    @staticmethod
    def interpolate_full_coverage(selected_track: list[dict[str, Any] | None]) -> list[dict]:
        """Interpolate across 100% of frames with light Savitzky-like 3-tap box smoothing on jitter."""
        num_frames = len(selected_track)
        if num_frames == 0:
            return []

        valid_idx = []
        valid_boxes = []
        for i, item in enumerate(selected_track):
            if item is not None:
                valid_idx.append(i)
                valid_boxes.append(item["bbox"])

        if not valid_idx:
            return [{} for _ in range(num_frames)]

        valid_idx_arr = np.asarray(valid_idx, dtype=np.float64)
        valid_boxes_arr = np.asarray(valid_boxes, dtype=np.float64)
        all_idx = np.arange(num_frames, dtype=np.float64)

        interp = np.column_stack(
            [np.interp(all_idx, valid_idx_arr, valid_boxes_arr[:, col]) for col in range(4)]
        )

        # Light [0.2, 0.6, 0.2] smoothing on centers to remove 1-2px bbox quantization jitter while keeping sharp kicks
        centers = 0.5 * (interp[:, :2] + interp[:, 2:4])
        sizes = interp[:, 2:4] - interp[:, :2]
        if num_frames >= 3:
            smoothed_centers = centers.copy()
            smoothed_centers[1:-1] = 0.2 * centers[:-2] + 0.6 * centers[1:-1] + 0.2 * centers[2:]
            centers = smoothed_centers

        final_boxes = np.hstack([centers - 0.5 * sizes, centers + 0.5 * sizes])
        return [{1: {"bbox": row.tolist()}} for row in final_boxes]

    def track_ball(
        self,
        video_path: str,
        model: YOLO,
        tracks: dict,
        read_from_stub: bool = True,
        candidates_stub_path: str = "stubs/ball_candidates_stub.pkl",
    ) -> list[dict]:
        """End-to-end accurate ball tracking pipeline with candidate & camera-motion caching."""
        if read_from_stub and candidates_stub_path and os.path.exists(candidates_stub_path):
            with open(candidates_stub_path, "rb") as f:
                cand_data = pickle.load(f)
        else:
            cand_data = self.extract_candidates_and_motion(video_path, model, tracks)
            if candidates_stub_path:
                os.makedirs(os.path.dirname(candidates_stub_path) or ".", exist_ok=True)
                with open(candidates_stub_path, "wb") as f:
                    pickle.dump(cand_data, f)

        # Stage 2 & 3 (initial pass)
        clean_cands = self.suppress_static_landmarks(
            cand_data["candidates"], cand_data["step_homographies"], tracks
        )
        initial_selected = self.link_trajectory_viterbi(clean_cands, cand_data["step_homographies"])

        # Stage 4: Native 1:1 640x640 ROI zoom on gaps (cached once into ball_candidates_stub.pkl)
        if not cand_data.get("zoom_enriched", False):
            self.enrich_gaps_with_roi_zoom(video_path, model, cand_data, initial_selected)
            if candidates_stub_path:
                os.makedirs(os.path.dirname(candidates_stub_path) or ".", exist_ok=True)
                with open(candidates_stub_path, "wb") as f:
                    pickle.dump(cand_data, f)

            clean_cands = self.suppress_static_landmarks(
                cand_data["candidates"], cand_data["step_homographies"], tracks
            )
            final_selected = self.link_trajectory_viterbi(clean_cands, cand_data["step_homographies"])
        else:
            final_selected = initial_selected

        return self.interpolate_full_coverage(final_selected)
