import os
import pickle
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from homography.pitch_model import PITCH_LENGTH, PITCH_WIDTH
from utils import FrameScaleContext, PitchColorModel, get_video_properties, iter_video_batches


class TeamAssigner:
    """
    Zero-hardcoded-visuals team & goalkeeper classifier:
      - Strips turf background from torso crops using the video's learned CIE Lab PitchColorModel.
      - Clusters outfield jerseys dynamically via KMeans(n_clusters=2) in Lab + circular HSV space.
      - Detects goalkeepers via YOLO role or dynamic cluster-distance anomaly inside either goal area,
        coloring goalkeepers with their team's outfield marker color.
      - Selects a high-contrast recognizable referee marker color (Yellow / Black / Cyan / Magenta)
        maximally separated from both teams' colors in perceptual CIE Lab space.
    """

    def __init__(self, sample_stride: int = 2) -> None:
        self.sample_stride = max(1, int(sample_stride))
        self.team_colors: dict[int, tuple[int, int, int]] = {
            1: (240, 240, 240),
            2: (60, 60, 215),
        }
        self.referee_color: tuple[int, int, int] = (0, 235, 255)
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
    def _bgr_to_cie_lab(bgr: tuple[int, int, int] | np.ndarray) -> np.ndarray:
        """Convert a single BGR triplet to standard CIE Lab (L in [0..100], a* in [-128..127], b* in [-128..127])."""
        px = np.uint8([[np.clip(bgr, 0, 255)]])
        lab_cv = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
        return np.array([lab_cv[0] * (100.0 / 255.0), lab_cv[1] - 128.0, lab_cv[2] - 128.0], dtype=np.float64)

    @classmethod
    def compute_contrasting_referee_color(
        cls,
        team_colors: dict[int, tuple[int, int, int]],
        turf_bgr: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """
        Choose an unmistakable referee marker color (preferring classic Yellow or Black, with
        Cyan/Magenta fallback) that has strong CIE Lab perceptual contrast against BOTH teams.
        """
        lab_t1 = cls._bgr_to_cie_lab(team_colors[1])
        lab_t2 = cls._bgr_to_cie_lab(team_colors[2])
        lab_turf = cls._bgr_to_cie_lab(turf_bgr)

        # Ordered broadcast-recognizable official marker palette (BGR)
        palette = [
            ("yellow", (0, 235, 255)),
            ("black", (18, 18, 18)),
            ("cyan", (255, 220, 0)),
            ("magenta", (220, 40, 235)),
        ]

        def is_distinct_from_teams(cand_bgr: tuple[int, int, int], min_de: float) -> bool:
            lc = cls._bgr_to_cie_lab(cand_bgr)
            cc = float(np.hypot(lc[1], lc[2]))
            hc = float(np.rad2deg(np.arctan2(lc[2], lc[1])))
            for lt in (lab_t1, lab_t2):
                if float(np.linalg.norm(lc - lt)) < min_de:
                    return False
                ct = float(np.hypot(lt[1], lt[2]))
                # If both candidate and team kit are genuinely chromatic (ct >= 34), require distinct hue (>= 45 deg)
                if cc >= 34.0 and ct >= 34.0:
                    ht = float(np.rad2deg(np.arctan2(lt[2], lt[1])))
                    dh = abs((hc - ht + 180.0) % 360.0 - 180.0)
                    if dh < 35.0:
                        return False
            return True

        # 1. Prefer Yellow whenever neither team wears a yellow/gold/amber kit
        if is_distinct_from_teams(palette[0][1], 44.0):
            return palette[0][1]

        # 2. If a team wears Yellow (e.g. Dortmund), prefer Black whenever neither team is very dark
        if is_distinct_from_teams(palette[1][1], 36.0):
            return palette[1][1]

        # 3. General max-min perceptual contrast fallback across all candidates
        best_col = palette[2][1]
        best_score = -1.0
        for _, col in palette:
            lc = cls._bgr_to_cie_lab(col)
            score = min(
                float(np.linalg.norm(lc - lab_t1)),
                float(np.linalg.norm(lc - lab_t2)),
                0.75 * float(np.linalg.norm(lc - lab_turf)),
            )
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    @staticmethod
    def _normalize_cluster_display_bgr(bgr_samples: list[np.ndarray]) -> tuple[int, int, int]:
        """
        Compute a crisp, hue-faithful broadcast BGR color from a jersey cluster's medoid sample
        using smooth CIE Lab chroma/lightness normalization (zero hardcoded color rules!).
        """
        arr = np.asarray(bgr_samples, dtype=np.float32)
        lab_arr = cv2.cvtColor(arr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
        med_lab = np.median(lab_arr, axis=0)
        medoid_idx = int(np.argmin(np.linalg.norm(lab_arr - med_lab, axis=1)))
        base_lab = lab_arr[medoid_idx].copy()

        l_cv = base_lab[0]
        ca = base_lab[1] - 128.0
        cb = base_lab[2] - 128.0
        chroma = float(np.hypot(ca, cb))

        # Preserve exact hue angle while gently boosting visibility for broadcast overlays
        boost = 1.28 if chroma >= 14.0 else 1.05
        ca_new = np.clip(128.0 + ca * boost, 0.0, 255.0)
        cb_new = np.clip(128.0 + cb * boost, 0.0, 255.0)
        l_new = np.clip(l_cv * 1.12 + 14.0, 45.0, 246.0)

        lab_px = np.uint8([[[l_new, ca_new, cb_new]]])
        bgr_out = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0, 0]
        return (int(bgr_out[0]), int(bgr_out[1]), int(bgr_out[2]))

    @staticmethod
    def extract_torso_descriptor(
        frame: np.ndarray,
        bbox: list[float],
        pitch_model: PitchColorModel,
        scale_ctx: FrameScaleContext,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract a grass-free CIE Lab + circular HSV histogram torso descriptor using relative geometry."""
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bh = float(y2 - y1)
        bw = float(x2 - x1)
        min_bh = 0.25 * scale_ctx.player_h_at(y2)
        if bh < min_bh or bw < 0.22 * min_bh:
            return None

        # Upper-jersey crop [16%..46% height] focuses strictly on the shirt (avoiding shorts!)
        ty1 = max(0, int(round(y1 + 0.16 * bh)))
        ty2 = min(fh, int(round(y1 + 0.46 * bh)))
        tx1 = max(0, int(round(x1 + 0.24 * bw)))
        tx2 = min(fw, int(round(x1 + 0.76 * bw)))
        if ty2 - ty1 < 3 or tx2 - tx1 < 3:
            return None

        crop = frame[ty1:ty2, tx1:tx2]
        fg_mask = ~pitch_model.is_turf_mask(crop)
        min_fg = max(4, int(round(0.12 * crop.shape[0] * crop.shape[1])))
        if int(np.sum(fg_mask)) < min_fg:
            ch, cw = crop.shape[:2]
            fg_mask = np.zeros((ch, cw), dtype=bool)
            fg_mask[max(0, ch // 4) : max(1, 3 * ch // 4), max(0, cw // 4) : max(1, 3 * cw // 4)] = True

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        fg_bgr = crop[fg_mask].astype(np.float32)
        fg_hsv = hsv[fg_mask].astype(np.float32)
        fg_lab = lab[fg_mask].astype(np.float32)
        if len(fg_bgr) == 0:
            return None

        h_rad = fg_hsv[:, 0] * (2.0 * np.pi / 180.0)
        s_norm = fg_hsv[:, 1] / 255.0
        v_norm = fg_hsv[:, 2] / 255.0

        h_bins, _ = np.histogram(fg_hsv[:, 0], bins=8, range=(0.0, 180.0), weights=s_norm)
        h_bins = h_bins / max(1.0, float(len(fg_hsv)))
        s_bins, _ = np.histogram(s_norm, bins=4, range=(0.0, 1.0))
        s_bins = s_bins / float(len(fg_hsv))
        v_bins, _ = np.histogram(v_norm, bins=4, range=(0.0, 1.0))
        v_bins = v_bins / float(len(fg_hsv))

        med_lab = np.median(fg_lab, axis=0)
        # Representative display BGR: if jersey has chromatic primary color (e.g. Red/White or Yellow/Black),
        # summarize the primary chromatic body pixels rather than blending with white sleeves or black trim!
        pixel_chroma = np.hypot(fg_lab[:, 1] - 128.0, fg_lab[:, 2] - 128.0)
        if float(np.percentile(pixel_chroma, 70.0)) >= 20.0:
            dom_mask = pixel_chroma >= np.percentile(pixel_chroma, 55.0)
            med_bgr = np.median(fg_bgr[dom_mask], axis=0)
        else:
            med_bgr = np.median(fg_bgr, axis=0)

        l_s = float(med_lab[0]) / 255.0
        a_s = (float(med_lab[1]) - 128.0) / 64.0
        b_s = (float(med_lab[2]) - 128.0) / 64.0
        cos_h = float(np.mean(s_norm * np.cos(h_rad)))
        sin_h = float(np.mean(s_norm * np.sin(h_rad)))

        descriptor = np.concatenate(
            [
                0.75 * h_bins,
                0.75 * s_bins,
                0.75 * v_bins,
                np.array([1.7 * l_s, 2.5 * a_s, 2.3 * b_s, 1.6 * cos_h, 1.6 * sin_h], dtype=np.float64),
            ]
        ).astype(np.float64)

        return descriptor, med_bgr

    def _collect_frame_color_samples(
        self,
        video_path: str,
        tracks: dict,
        pitch_model: PitchColorModel,
        scale_ctx: FrameScaleContext,
    ) -> dict[str, Any]:
        """Stream video at stride intervals and record grass-masked per-(frame, track_id) torso descriptors."""
        player_samples: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)

        frame_idx = 0
        for batch in iter_video_batches(video_path, batch_size=20):
            for frame in batch:
                if frame_idx % self.sample_stride == 0 and frame_idx < len(tracks.get("players", [])):
                    for tid, pdata in tracks["players"][frame_idx].items():
                        res = self.extract_torso_descriptor(frame, pdata["bbox"], pitch_model, scale_ctx)
                        if res is not None:
                            desc, bgr = res
                            player_samples[tid].append((frame_idx, desc, bgr))
                frame_idx += 1

        return {
            "version": 5,
            "player_samples": dict(player_samples),
        }

    def assign_teams(
        self,
        video_path: str,
        tracks: dict,
        pitch_model: PitchColorModel | None = None,
        scale_ctx: FrameScaleContext | None = None,
        read_from_stub: bool = True,
        stub_path: str = "stubs/team_assigner_stub.pkl",
    ) -> dict[str, Any]:
        """Cluster outfield jerseys dynamically per video, resolve goalkeepers, and compute contrast referee color."""
        props = get_video_properties(video_path)
        if pitch_model is None:
            pitch_model = PitchColorModel.fit_from_video(video_path, read_from_stub=False)
        if scale_ctx is None:
            scale_ctx = FrameScaleContext(props["width"], props["height"], props["fps"]).calibrate_from_tracks(tracks)

        cache = None
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and loaded.get("version") == 5:
                cache = loaded

        if cache is None:
            cache = self._collect_frame_color_samples(video_path, tracks, pitch_model, scale_ctx)
            if stub_path:
                os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
                with open(stub_path, "wb") as f:
                    pickle.dump(cache, f)

        player_samples: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = cache["player_samples"]
        num_frames = len(tracks.get("players", []))

        track_frames: dict[int, int] = defaultdict(int)
        track_on_pitch_votes: dict[int, list[bool]] = defaultdict(list)
        for f in range(num_frames):
            for tid, pdata in tracks["players"][f].items():
                track_frames[tid] += 1
                if pdata.get("in_visual_pitch", True):
                    track_on_pitch_votes[tid].append(bool(pdata.get("on_pitch", True)))
                else:
                    track_on_pitch_votes[tid].append(False)

        track_is_on_pitch: dict[int, bool] = {
            tid: (float(np.mean(votes)) >= 0.40 if votes else True)
            for tid, votes in track_on_pitch_votes.items()
        }

        # 1. Fit unsupervised KMeans(n_clusters=2) strictly on non-goal-mouth outfield player crops
        train_descs: list[np.ndarray] = []
        train_bgrs: list[np.ndarray] = []
        train_xs: list[float] = []

        for tid, obs_list in player_samples.items():
            if not track_is_on_pitch.get(tid, True):
                continue
            for idx_s, (f_obs, desc, bgr) in enumerate(obs_list):
                if idx_s % 2 != 0 or f_obs >= num_frames or tid not in tracks["players"][f_obs]:
                    continue
                pdata = tracks["players"][f_obs][tid]
                if pdata.get("role") == "goalkeeper" or not pdata.get("in_visual_pitch", True):
                    continue
                pos = pdata.get("position_unclamped") or pdata.get("position_transformed")
                if pos is not None:
                    mx, my = float(pos[0]), float(pos[1])
                    # Exclude 6-yard goal mouth from initial 2-team kit fit so unique GK kits don't skew centroids
                    if (mx < 7.0 or mx > (PITCH_LENGTH - 7.0)) and (24.0 <= my <= 44.0):
                        continue
                train_descs.append(desc)
                train_bgrs.append(bgr)
                train_xs.append(float(pos[0]) if pos is not None else 52.5)

        if len(train_descs) < 4:
            # Fallback using all available samples
            for tid, obs_list in player_samples.items():
                for f_obs, desc, bgr in obs_list:
                    train_descs.append(desc)
                    train_bgrs.append(bgr)
                    train_xs.append(52.5)

        X_train = np.asarray(train_descs, dtype=np.float64)
        with threadpool_limits(limits=1):
            kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
            train_labels = kmeans.fit_predict(X_train)
        centers = kmeans.cluster_centers_

        # Dynamic anomaly threshold for non-outfield goalkeeper jerseys (92nd percentile of outfield distances)
        dists_to_assigned = np.linalg.norm(X_train - centers[train_labels], axis=1)
        gk_anomaly_thresh = float(np.percentile(dists_to_assigned, 92.0)) * 1.12

        # Order Team 1 (defends left goal X=0) vs Team 2 (defends right goal X=105) by mean X_m
        cl_xs = {0: [], 1: []}
        cl_bgrs = {0: [], 1: []}
        for lbl, x_val, bgr in zip(train_labels, train_xs, train_bgrs):
            cl_xs[int(lbl)].append(x_val)
            cl_bgrs[int(lbl)].append(bgr)

        mx0 = float(np.mean(cl_xs[0])) if cl_xs[0] else 35.0
        mx1 = float(np.mean(cl_xs[1])) if cl_xs[1] else 70.0
        if mx0 <= mx1:
            team1_cl, team2_cl = 0, 1
        else:
            team1_cl, team2_cl = 1, 0

        self.team_defends_left = {1: True, 2: False}
        for team_id, cl_idx in ((1, team1_cl), (2, team2_cl)):
            if cl_bgrs[cl_idx]:
                self.team_colors[team_id] = self._normalize_cluster_display_bgr(cl_bgrs[cl_idx])

        self.referee_color = self.compute_contrasting_referee_color(self.team_colors, pitch_model.mean_bgr)

        c_team1 = centers[team1_cl]
        c_team2 = centers[team2_cl]

        # 2. Classify each sampled observation, using dynamic anomaly distance inside goal areas for GKs
        per_track_obs_roles: dict[int, list[tuple[int, str, int]]] = {}
        for tid, obs_list in player_samples.items():
            obs_decisions: list[tuple[int, str, int]] = []
            for f_obs, desc, _ in obs_list:
                if f_obs >= num_frames or tid not in tracks["players"][f_obs]:
                    continue
                pdata = tracks["players"][f_obs][tid]
                pos = pdata.get("position_unclamped") or pdata.get("position_transformed")
                mx = float(pos[0]) if pos is not None else 52.5
                my = float(pos[1]) if pos is not None else 34.0

                d1 = float(np.linalg.norm(desc - c_team1))
                d2 = float(np.linalg.norm(desc - c_team2))
                in_goal_area = pos is not None and (mx < 8.5 or mx > (PITCH_LENGTH - 8.5)) and (23.5 <= my <= 44.5)

                is_gk = pdata.get("role") == "goalkeeper" or (in_goal_area and min(d1, d2) > gk_anomaly_thresh)
                if is_gk:
                    gk_team = 1 if mx < (0.5 * PITCH_LENGTH) else 2
                    obs_decisions.append((f_obs, "goalkeeper", gk_team))
                elif d1 <= d2:
                    obs_decisions.append((f_obs, "player", 1))
                else:
                    obs_decisions.append((f_obs, "player", 2))

            if obs_decisions:
                per_track_obs_roles[tid] = obs_decisions

        # Local FPS-scaled temporal smoothing (~0.36s window) adapts cleanly across ByteTrack opponent crossings
        half_w = scale_ctx.frames_for_seconds(0.36, minimum=5)
        resolved_per_frame: dict[int, dict[int, tuple[str, int]]] = defaultdict(dict)

        for f in range(num_frames):
            for tid, pdata in tracks["players"][f].items():
                obs_dec = per_track_obs_roles.get(tid)
                if not obs_dec:
                    role_def = pdata.get("role", "player")
                    pos = pdata.get("position_transformed")
                    t_def = 1 if (pos is None or pos[0] < 52.5) else 2
                    resolved_per_frame[f][tid] = (role_def, t_def)
                    continue

                nearby = [item for item in obs_dec if abs(item[0] - f) <= half_w]
                if not nearby:
                    nearby = sorted(obs_dec, key=lambda item: abs(item[0] - f))[:3]

                gk_votes = sum(1 for _, r, _ in nearby if r == "goalkeeper")
                t1_votes = sum(1 for _, r, t in nearby if r == "player" and t == 1)
                t2_votes = sum(1 for _, r, t in nearby if r == "player" and t == 2)

                if gk_votes >= max(t1_votes, t2_votes) and gk_votes > 0:
                    pos = pdata.get("position_transformed")
                    gk_t = 1 if (pos is None or pos[0] < 52.5) else 2
                    resolved_per_frame[f][tid] = ("goalkeeper", gk_t)
                elif t1_votes >= t2_votes:
                    resolved_per_frame[f][tid] = ("player", 1)
                else:
                    resolved_per_frame[f][tid] = ("player", 2)

        # 3. Deduplicate same-frame split boxes and attach team colors (GK shares team's outfield color!)
        for f in range(num_frames):
            frame_players = tracks["players"][f]
            if not frame_players:
                continue

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
                    same_gk_goal = role_c == "goalkeeper" and role_k == "goalkeeper" and foot_dist < 4.5
                    if iou >= 0.55 or same_gk_goal or (same_team_or_gk and (iou >= 0.35 or foot_dist < 0.45)):
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
                # Goalkeeper uses the exact same marker color as his team's outfield players (per user preference)
                pdata["team_color"] = self.team_colors[team_id]
                frame_in_bounds = pdata.get("on_pitch", True)
                pdata["on_pitch"] = bool(track_is_on_pitch.get(tid, True) and frame_in_bounds and pdata.get("in_visual_pitch", True))

        return {
            "team_colors": self.team_colors,
            "referee_color": self.referee_color,
            "team_defends_left": self.team_defends_left,
        }
