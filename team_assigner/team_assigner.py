import os
import pickle
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from homography.pitch_model import PITCH_LENGTH, PITCH_WIDTH
from utils import iter_video_batches


class TeamAssigner:
    """
    Unsupervised K-Means (k=3) torso HSV histogram + Lab chromaticity classifier.
    Classifies per-frame sampled detections (with local 9-frame window smoothing) so
    ByteTrack opponent ID swaps when players cross paths never corrupt kit assignments!
    """

    GOALKEEPER_COLOR = (255, 191, 0)  # Distinct Sky Blue (BGR) per user preference

    def __init__(self, sample_stride: int = 2, local_smooth_frames: int = 9) -> None:
        self.sample_stride = sample_stride
        self.local_smooth_frames = local_smooth_frames
        self.team_colors: dict[int, tuple[int, int, int]] = {
            1: (245, 245, 245),
            2: (55, 55, 225),
        }
        self.team_defends_left: dict[int, bool] = {1: True, 2: False}

    @staticmethod
    def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return float(inter / (area_a + area_b - inter))

    @staticmethod
    def extract_torso_descriptor(
        frame: np.ndarray,
        bbox: list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Extract a grass-masked 16-bin HSV histogram + CIE Lab chromaticity torso descriptor."""
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        bw = x2 - x1
        if bh < 12.0 or bw < 6.0:
            return None

        ty1 = max(0, int(round(y1 + 0.17 * bh)))
        ty2 = min(fh, int(round(y1 + 0.48 * bh)))
        tx1 = max(0, int(round(x1 + 0.24 * bw)))
        tx2 = min(fw, int(round(x1 + 0.76 * bw)))
        if ty2 - ty1 < 3 or tx2 - tx1 < 3:
            return None

        crop = frame[ty1:ty2, tx1:tx2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        grass_mask = cv2.inRange(
            hsv,
            np.array([25, 28, 30], dtype=np.uint8),
            np.array([96, 255, 255], dtype=np.uint8),
        )
        fg_mask = grass_mask == 0
        if int(np.sum(fg_mask)) < 6:
            ch, cw = crop.shape[:2]
            fg_mask = np.zeros((ch, cw), dtype=bool)
            fg_mask[max(0, ch // 4) : max(1, 3 * ch // 4), max(0, cw // 4) : max(1, 3 * cw // 4)] = True

        fg_hsv = hsv[fg_mask].astype(np.float32)
        fg_bgr = crop[fg_mask].astype(np.float32)
        fg_lab = lab[fg_mask].astype(np.float32)
        if len(fg_hsv) == 0:
            return None

        h_deg_half = fg_hsv[:, 0]          # [0..179]
        s_norm = fg_hsv[:, 1] / 255.0      # [0..1]
        v_norm = fg_hsv[:, 2] / 255.0      # [0..1]

        # 16-bin normalized HSV torso histogram (Proposal §4.2)
        h_bins, _ = np.histogram(h_deg_half, bins=6, range=(0.0, 180.0), weights=s_norm)
        h_bins = h_bins / max(1.0, float(len(fg_hsv)))
        s_bins, _ = np.histogram(s_norm, bins=5, range=(0.0, 1.0))
        s_bins = s_bins / float(len(fg_hsv))
        v_bins, _ = np.histogram(v_norm, bins=5, range=(0.0, 1.0))
        v_bins = v_bins / float(len(fg_hsv))

        med_hsv = np.median(fg_hsv, axis=0)
        med_bgr = np.median(fg_bgr, axis=0)
        med_lab = np.median(fg_lab, axis=0)

        l_scaled = float(med_lab[0]) / 255.0
        a_scaled = (float(med_lab[1]) - 128.0) / 64.0
        b_scaled = (float(med_lab[2]) - 128.0) / 64.0
        s_med_scaled = float(med_hsv[1]) / 255.0
        v_med_scaled = float(med_hsv[2]) / 255.0

        descriptor = np.concatenate(
            [
                0.7 * h_bins,
                0.7 * s_bins,
                0.7 * v_bins,
                np.array(
                    [
                        1.8 * l_scaled,
                        2.6 * a_scaled,
                        1.6 * b_scaled,
                        2.0 * s_med_scaled,
                        1.8 * v_med_scaled,
                    ],
                    dtype=np.float64,
                ),
            ]
        ).astype(np.float64)

        return descriptor, med_bgr, med_hsv

    @staticmethod
    def _enhance_broadcast_kit_color(med_bgr: np.ndarray, med_hsv: np.ndarray) -> tuple[int, int, int]:
        """Convert raw shaded torso BGR into a crisp, high-contrast broadcast annotation BGR color."""
        s_val = float(med_hsv[1])
        v_val = float(med_hsv[2])
        if s_val < 70.0 and v_val > 140.0:
            return (245, 245, 245)

        bgr_pixel = np.uint8([[np.clip(med_bgr, 0, 255)]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Remove green turf bounce from red/burgundy kits so crimson stays distinct from orange/amber
        if float(med_bgr[2]) > float(med_bgr[1]) + 32.0 and float(med_bgr[2]) > float(med_bgr[0]) + 45.0:
            hsv_pixel[0, 0, 0] = min(hsv_pixel[0, 0, 0], 3.0)
        hsv_pixel[0, 0, 1] = np.clip(hsv_pixel[0, 0, 1] * 1.40, 185.0, 255.0)
        hsv_pixel[0, 0, 2] = np.clip(hsv_pixel[0, 0, 2] * 1.45, 205.0, 240.0)
        vivid_bgr = cv2.cvtColor(hsv_pixel.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        return (int(vivid_bgr[0]), int(vivid_bgr[1]), int(vivid_bgr[2]))

    def _collect_frame_color_samples(self, video_path: str, tracks: dict) -> dict[str, Any]:
        """Stream video at stride intervals and record per-(frame, track_id) torso descriptors."""
        # player_samples[tid] = list of (frame_idx, desc, bgr, hsv)
        player_samples: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
        ref_samples: list[tuple[int, int, np.ndarray, np.ndarray]] = []

        frame_idx = 0
        for batch in iter_video_batches(video_path, batch_size=20):
            for frame in batch:
                if frame_idx % self.sample_stride == 0:
                    if frame_idx < len(tracks.get("players", [])):
                        for tid, pdata in tracks["players"][frame_idx].items():
                            res = self.extract_torso_descriptor(frame, pdata["bbox"])
                            if res is not None:
                                desc, bgr, hsv = res
                                player_samples[tid].append((frame_idx, desc, bgr, hsv))

                    if frame_idx < len(tracks.get("referees", [])):
                        for rtid, rdata in tracks["referees"][frame_idx].items():
                            res = self.extract_torso_descriptor(frame, rdata["bbox"])
                            if res is not None:
                                desc, _, hsv = res
                                ref_samples.append((frame_idx, rtid, desc, hsv))
                frame_idx += 1

        return {
            "version": 3,
            "player_samples": dict(player_samples),
            "ref_samples": ref_samples,
        }

    @staticmethod
    def _is_goalkeeper_observation(
        role_hint: str,
        pos_xy: list[float] | None,
        bgr: np.ndarray,
        hsv: np.ndarray,
    ) -> bool:
        """Identify goalkeeper observations via YOLO role or distinct dark/cyan jersey inside the 6-yard box."""
        if role_hint == "goalkeeper":
            return True
        if pos_xy is None:
            return False
        mx, my = float(pos_xy[0]), float(pos_xy[1])
        in_goal_area = (mx < 7.0 or mx > (PITCH_LENGTH - 7.0)) and (24.5 <= my <= 43.5)
        if not in_goal_area:
            return False
        # Non-white (V < 140) and non-vivid-red (R - B < 32) kit inside the 6-yard goal area
        return float(hsv[2]) < 140.0 and (float(bgr[2]) - float(bgr[0])) < 32.0

    def assign_teams(
        self,
        video_path: str,
        tracks: dict,
        read_from_stub: bool = True,
        stub_path: str = "stubs/team_assigner_stub.pkl",
    ) -> dict[str, Any]:
        """Classify Team 1, Team 2, Goalkeepers, and Match Officials via per-crop K-Means (k=3)."""
        cache = None
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and loaded.get("version") == 3:
                cache = loaded

        if cache is None:
            cache = self._collect_frame_color_samples(video_path, tracks)
            if stub_path:
                os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
                with open(stub_path, "wb") as f:
                    pickle.dump(cache, f)

        player_samples: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = cache["player_samples"]
        ref_samples: list[tuple[int, int, np.ndarray, np.ndarray]] = cache["ref_samples"]

        num_frames = len(tracks.get("players", []))
        track_frames: dict[int, int] = defaultdict(int)
        track_on_pitch_votes: dict[int, list[bool]] = defaultdict(list)

        for f in range(num_frames):
            for tid, pdata in tracks["players"][f].items():
                track_frames[tid] += 1
                unc = pdata.get("position_unclamped") or pdata.get("position_transformed")
                if unc is not None:
                    in_bounds = -0.8 <= unc[0] <= PITCH_LENGTH + 0.8 and -0.8 <= unc[1] <= PITCH_WIDTH + 0.8
                    track_on_pitch_votes[tid].append(bool(in_bounds))

        track_is_on_pitch: dict[int, bool] = {
            tid: (float(np.mean(votes)) >= 0.45 if votes else True)
            for tid, votes in track_on_pitch_votes.items()
        }

        # 1. Build per-crop KMeans(n_clusters=3) training matrix on pure individual crops (no ID-swap mixing!)
        train_descs: list[np.ndarray] = []
        train_bgrs: list[np.ndarray] = []
        train_hsvs: list[np.ndarray] = []
        train_xs: list[float] = []
        train_kinds: list[str] = []

        # Subsample every 3rd collected sample (~every 6th frame) for fast K-Means fitting
        for tid, obs_list in player_samples.items():
            if not track_is_on_pitch.get(tid, True):
                continue
            for idx_s, (f_obs, desc, bgr, hsv) in enumerate(obs_list):
                if idx_s % 3 != 0 or f_obs >= num_frames or tid not in tracks["players"][f_obs]:
                    continue
                pdata = tracks["players"][f_obs][tid]
                pos = pdata.get("position_unclamped") or pdata.get("position_transformed")
                if self._is_goalkeeper_observation(pdata.get("role", "player"), pos, bgr, hsv):
                    continue
                train_descs.append(desc)
                train_bgrs.append(bgr)
                train_hsvs.append(hsv)
                train_xs.append(float(pos[0]) if pos is not None else 52.5)
                train_kinds.append("player")

        # Include official torso samples weighted proportionally (~30% of player sample count)
        n_player_train = max(1, len(train_descs))
        ref_repeat = max(1, int(round((0.30 * n_player_train) / max(1, len(ref_samples)))))
        for _, _, rdesc, rhsv in ref_samples:
            for _ in range(ref_repeat):
                train_descs.append(rdesc)
                train_bgrs.append(np.zeros(3, dtype=np.float32))
                train_hsvs.append(rhsv)
                train_xs.append(52.5)
                train_kinds.append("referee")

        X_train = np.asarray(train_descs, dtype=np.float64)
        with threadpool_limits(limits=1):
            kmeans = KMeans(n_clusters=3, n_init=15, random_state=42)
            train_labels = kmeans.fit_predict(X_train)
        centers = kmeans.cluster_centers_

        # Identify Match Official cluster via referee sample majority
        ref_cluster_counts = np.zeros(3, dtype=np.int32)
        for kind, lbl in zip(train_kinds, train_labels):
            if kind == "referee":
                ref_cluster_counts[int(lbl)] += 1
        official_cluster = (
            int(np.argmax(ref_cluster_counts))
            if np.max(ref_cluster_counts) > 0
            else int(np.argmin(centers[:, -1]))
        )

        kit_clusters = [c for c in range(3) if c != official_cluster]

        # Order Team 1 (defends left goal X=0) vs Team 2 (defends right goal X=105) by mean X_m
        cl_xs: dict[int, list[float]] = {c: [] for c in kit_clusters}
        cl_bgrs: dict[int, list[np.ndarray]] = {c: [] for c in kit_clusters}
        cl_hsvs: dict[int, list[np.ndarray]] = {c: [] for c in kit_clusters}

        for kind, lbl, x_val, bgr, hsv in zip(train_kinds, train_labels, train_xs, train_bgrs, train_hsvs):
            if kind == "player" and int(lbl) in kit_clusters:
                cl_xs[int(lbl)].append(x_val)
                cl_bgrs[int(lbl)].append(bgr)
                cl_hsvs[int(lbl)].append(hsv)

        mx0 = float(np.mean(cl_xs[kit_clusters[0]])) if cl_xs[kit_clusters[0]] else 35.0
        mx1 = float(np.mean(cl_xs[kit_clusters[1]])) if cl_xs[kit_clusters[1]] else 70.0
        if mx0 <= mx1:
            team1_cluster, team2_cluster = kit_clusters[0], kit_clusters[1]
        else:
            team1_cluster, team2_cluster = kit_clusters[1], kit_clusters[0]

        self.team_defends_left = {1: True, 2: False}
        for team_id, cl_idx in ((1, team1_cluster), (2, team2_cluster)):
            if cl_bgrs[cl_idx] and cl_hsvs[cl_idx]:
                self.team_colors[team_id] = self._enhance_broadcast_kit_color(
                    np.median(cl_bgrs[cl_idx], axis=0), np.median(cl_hsvs[cl_idx], axis=0)
                )

        c_team1 = centers[team1_cluster]
        c_team2 = centers[team2_cluster]
        c_off = centers[official_cluster]

        # 2. Classify every sampled observation per track_id, then smooth locally (+/- 9 frames)
        #    so 1-frame shadow flicker is smoothed while ByteTrack opponent ID swaps adapt cleanly!
        per_track_obs_roles: dict[int, list[tuple[int, str, int]]] = {}
        for tid, obs_list in player_samples.items():
            obs_decisions: list[tuple[int, str, int]] = []
            for f_obs, desc, bgr, hsv in obs_list:
                if f_obs >= num_frames or tid not in tracks["players"][f_obs]:
                    continue
                pdata = tracks["players"][f_obs][tid]
                pos = pdata.get("position_unclamped") or pdata.get("position_transformed")
                mx = float(pos[0]) if pos is not None else 52.5

                if self._is_goalkeeper_observation(pdata.get("role", "player"), pos, bgr, hsv):
                    gk_team = 1 if mx < (0.5 * PITCH_LENGTH) else 2
                    obs_decisions.append((f_obs, "goalkeeper", gk_team))
                    continue

                d1 = float(np.linalg.norm(desc - c_team1))
                d2 = float(np.linalg.norm(desc - c_team2))
                doff = float(np.linalg.norm(desc - c_off))

                # Stray mid-pitch official check (dark black kit on short tracklet)
                if doff < min(d1, d2) and float(hsv[2]) < 88.0 and track_frames[tid] <= 20 and 10.0 <= mx <= 95.0:
                    obs_decisions.append((f_obs, "referee", 0))
                elif d1 <= d2:
                    obs_decisions.append((f_obs, "player", 1))
                else:
                    obs_decisions.append((f_obs, "player", 2))

            if obs_decisions:
                per_track_obs_roles[tid] = obs_decisions

        # Interpolate local smoothed (role, team_id) to every frame of each track_id
        resolved_per_frame: dict[int, dict[int, tuple[str, int]]] = defaultdict(dict)
        half_w = self.local_smooth_frames

        for f in range(num_frames):
            for tid, pdata in tracks["players"][f].items():
                obs_dec = per_track_obs_roles.get(tid)
                if not obs_dec:
                    role_def = pdata.get("role", "player")
                    pos = pdata.get("position_transformed")
                    t_def = 1 if (pos is None or pos[0] < 52.5) else 2
                    resolved_per_frame[f][tid] = (role_def, t_def if role_def != "goalkeeper" else t_def)
                    continue

                # Gather observations within +- half_w frames (or nearest 3 observations)
                nearby = [item for item in obs_dec if abs(item[0] - f) <= half_w]
                if not nearby:
                    nearby = sorted(obs_dec, key=lambda item: abs(item[0] - f))[:3]

                gk_votes = sum(1 for _, r, _ in nearby if r == "goalkeeper")
                ref_votes = sum(1 for _, r, _ in nearby if r == "referee")
                t1_votes = sum(1 for _, r, t in nearby if r == "player" and t == 1)
                t2_votes = sum(1 for _, r, t in nearby if r == "player" and t == 2)

                if gk_votes > max(ref_votes, t1_votes + t2_votes) // 2 and gk_votes >= 1 and pdata.get("role") == "goalkeeper":
                    pos = pdata.get("position_transformed")
                    gk_t = 1 if (pos is None or pos[0] < 52.5) else 2
                    resolved_per_frame[f][tid] = ("goalkeeper", gk_t)
                elif gk_votes >= max(ref_votes, t1_votes, t2_votes) and gk_votes > 0:
                    pos = pdata.get("position_transformed")
                    gk_t = 1 if (pos is None or pos[0] < 52.5) else 2
                    resolved_per_frame[f][tid] = ("goalkeeper", gk_t)
                elif ref_votes > max(t1_votes, t2_votes):
                    resolved_per_frame[f][tid] = ("referee", 0)
                elif t1_votes >= t2_votes:
                    resolved_per_frame[f][tid] = ("player", 1)
                else:
                    resolved_per_frame[f][tid] = ("player", 2)

        # 3. Apply per-frame roles, reclassify stray officials, deduplicate split boxes, and attach colors
        for f in range(num_frames):
            frame_players = tracks["players"][f]
            if not frame_players:
                continue

            for tid in list(frame_players.keys()):
                role, _ = resolved_per_frame[f].get(tid, ("player", 1))
                if role == "referee":
                    pdata = frame_players.pop(tid)
                    tracks["referees"][f][10000 + tid] = {
                        "bbox": pdata["bbox"],
                        "position_transformed": pdata.get("position_transformed"),
                        "position_unclamped": pdata.get("position_unclamped"),
                        "on_pitch": pdata.get("on_pitch", True),
                    }

            sorted_tids = sorted(frame_players.keys(), key=lambda t: track_frames[t], reverse=True)
            kept_tids: list[int] = []
            for tid in sorted_tids:
                box_curr = frame_players[tid]["bbox"]
                pt_curr = frame_players[tid].get("position_unclamped") or frame_players[tid].get("position_transformed")
                role_c, team_c = resolved_per_frame[f].get(tid, ("player", 1))
                is_dup = False
                for ktid in kept_tids:
                    role_k, team_k = resolved_per_frame[f].get(ktid, ("player", 1))
                    iou = self._bbox_iou(box_curr, frame_players[ktid]["bbox"])
                    same_team_or_gk = (team_c == team_k) or (role_c == "goalkeeper" and role_k == "goalkeeper")
                    pt_kept = frame_players[ktid].get("position_unclamped") or frame_players[ktid].get("position_transformed")
                    foot_dist = (
                        float(np.hypot(pt_curr[0] - pt_kept[0], pt_curr[1] - pt_kept[1]))
                        if (pt_curr is not None and pt_kept is not None)
                        else 99.0
                    )
                    same_gk_goal = (
                        role_c == "goalkeeper"
                        and role_k == "goalkeeper"
                        and foot_dist < 4.5
                    )
                    if iou >= 0.55 or same_gk_goal or (same_team_or_gk and (iou >= 0.35 or foot_dist < 0.48)):
                        is_dup = True
                        break
                if is_dup:
                    del frame_players[tid]
                else:
                    kept_tids.append(tid)

            for tid, pdata in frame_players.items():
                role, team_id = resolved_per_frame[f].get(tid, ("player", 1))
                pdata["role"] = role
                pdata["team_id"] = team_id
                pdata["team_color"] = (
                    self.GOALKEEPER_COLOR if role == "goalkeeper" else self.team_colors[team_id]
                )
                frame_in_bounds = pdata.get("on_pitch", True)
                pdata["on_pitch"] = bool(track_is_on_pitch.get(tid, True) and frame_in_bounds)

        return {
            "team_colors": self.team_colors,
            "goalkeeper_color": self.GOALKEEPER_COLOR,
            "team_defends_left": self.team_defends_left,
        }
