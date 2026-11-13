import os
import pickle
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from homography.pitch_model import PITCH_LENGTH, PITCH_WIDTH


# Canonical 10-outfield-player football formations
FORMATIONS_3_LINE: list[tuple[int, int, int]] = [
    (4, 4, 2),
    (4, 3, 3),
    (4, 5, 1),
    (3, 5, 2),
    (3, 4, 3),
    (5, 4, 1),
    (5, 3, 2),
]

FORMATIONS_4_LINE: list[tuple[int, int, int, int]] = [
    (4, 2, 3, 1),
    (4, 1, 4, 1),
]


class FormationAnalyzer:
    """Dual Spatial K-Means formation classifier for both On-Ball (Attacking) and Off-Ball (Defending) teams."""

    def __init__(
        self,
        coord_smooth_window: int = 35,
        ghost_ttl_frames: int = 50,
        formation_smooth_window: int = 35,
        min_4line_band_gap_m: float = 5.2,
    ) -> None:
        self.coord_smooth_window = coord_smooth_window
        self.ghost_ttl_frames = ghost_ttl_frames
        self.formation_smooth_window = formation_smooth_window
        self.min_4line_band_gap_m = min_4line_band_gap_m

    @staticmethod
    def _normalize_tactical_coords(
        x_m: float,
        y_m: float,
        defends_left: bool,
    ) -> tuple[float, float]:
        """Map pitch (X_m, Y_m) to team-centric (u_depth, v_width) where u=0 is own goal and u=105 is opponent goal."""
        if defends_left:
            return float(x_m), float(y_m)
        return float(PITCH_LENGTH - x_m), float(PITCH_WIDTH - y_m)

    @staticmethod
    def _camera_fov_depth_span(
        H_img_to_pitch: np.ndarray,
        defends_left: bool,
        frame_w: int = 1920,
        frame_h: int = 1080,
    ) -> tuple[float, float]:
        """Compute the visible [u_min, u_max] tactical depth range covered by the broadcast camera at frame t."""
        top_y = 0.18 * frame_h
        sample_img = np.array(
            [
                [0.0, top_y, 1.0],
                [0.5 * frame_w, top_y, 1.0],
                [float(frame_w), top_y, 1.0],
                [float(frame_w), float(frame_h), 1.0],
                [0.5 * frame_w, float(frame_h), 1.0],
                [0.0, float(frame_h), 1.0],
            ],
            dtype=np.float64,
        )
        hom = sample_img @ H_img_to_pitch.T
        valid = hom[:, 2] > 1e-4
        if not np.any(valid):
            return 0.0, PITCH_LENGTH
        xs = np.clip(hom[valid, 0] / hom[valid, 2], 0.0, PITCH_LENGTH)
        us = xs if defends_left else (PITCH_LENGTH - xs)
        return float(np.min(us)), float(np.max(us))

    def _build_smoothed_team_trajectories(
        self,
        tracks: dict,
        team_defends_left: dict[int, bool],
    ) -> dict[int, dict[int, dict[int, tuple[float, float]]]]:
        """Smooth each on-pitch outfield player's (u_depth, v_width) within contiguous same-team segments."""
        num_frames = len(tracks.get("players", []))
        raw_series: dict[int, dict[int, list[tuple[int, float, float]]]] = {
            1: defaultdict(list),
            2: defaultdict(list),
        }

        for f in range(num_frames):
            for tid, pdata in tracks["players"][f].items():
                if pdata.get("role") != "player" or not pdata.get("on_pitch", True):
                    continue
                team_id = pdata.get("team_id")
                pt = pdata.get("position_transformed")
                if team_id not in (1, 2) or pt is None:
                    continue
                u, v = self._normalize_tactical_coords(pt[0], pt[1], team_defends_left.get(team_id, True))
                raw_series[team_id][tid].append((f, u, v))

        half = self.coord_smooth_window // 2
        sigma = max(1.0, self.coord_smooth_window / 4.0)
        smoothed: dict[int, dict[int, dict[int, tuple[float, float]]]] = {1: {}, 2: {}}

        for team_id in (1, 2):
            for tid, obs in raw_series[team_id].items():
                # Split into contiguous segments if gap > 10 frames (prevents smoothing across ID swaps)
                segments: list[list[tuple[int, float, float]]] = []
                for item in obs:
                    if not segments or (item[0] - segments[-1][-1][0]) > 10:
                        segments.append([item])
                    else:
                        segments[-1].append(item)

                tid_map: dict[int, tuple[float, float]] = {}
                for seg in segments:
                    frames_arr = np.array([o[0] for o in seg], dtype=np.int32)
                    uv_arr = np.array([[o[1], o[2]] for o in seg], dtype=np.float64)
                    for f_curr in frames_arr:
                        dt = frames_arr - f_curr
                        mask = np.abs(dt) <= half
                        weights = np.exp(-0.5 * (dt[mask] / sigma) ** 2)
                        weights /= np.sum(weights)
                        sm_uv = np.sum(uv_arr[mask] * weights[:, None], axis=0)
                        tid_map[int(f_curr)] = (float(sm_uv[0]), float(sm_uv[1]))
                smoothed[team_id][tid] = tid_map

        return smoothed

    @staticmethod
    def _compensate_counts_to_10(
        counts: list[int],
        u_min_fov: float,
        u_max_fov: float,
    ) -> list[int]:
        """Adjust visible outfield line counts to sum to 10 using camera frustum clipping priors."""
        adj = list(counts)
        total = sum(adj)
        if total == 10:
            return adj

        if total < 10:
            missing = 10 - total
            back_clipped = max(0.0, u_min_fov - 4.0)
            front_clipped = max(0.0, (PITCH_LENGTH - 8.0) - u_max_fov)

            for _ in range(missing):
                if back_clipped >= front_clipped + 6.0:
                    if adj[0] < 4:
                        adj[0] += 1
                    else:
                        adj[1] += 1
                elif front_clipped >= back_clipped + 6.0:
                    if adj[-1] < 3:
                        adj[-1] += 1
                    else:
                        adj[-2] += 1
                else:
                    if adj[0] < 3:
                        adj[0] += 1
                    elif adj[-1] < 1:
                        adj[-1] += 1
                    else:
                        adj[len(adj) // 2] += 1
        else:
            while sum(adj) > 10:
                idx = int(np.argmax(adj))
                adj[idx] -= 1

        return adj

    @staticmethod
    def _match_canonical_3line(counts_3: list[int], phase: str) -> tuple[int, int, int]:
        """Match 3-band counts to the nearest canonical 10-man (DEF, MID, FWD) formation."""
        cum_obs = np.array([counts_3[0], counts_3[0] + counts_3[1]], dtype=np.float64)
        best_tmpl = FORMATIONS_3_LINE[0]
        best_cost = 1e9

        for tmpl in FORMATIONS_3_LINE:
            cum_t = np.array([tmpl[0], tmpl[0] + tmpl[1]], dtype=np.float64)
            cost = float(np.sum(np.abs(cum_obs - cum_t))) + 0.5 * float(
                sum(abs(a - b) for a, b in zip(counts_3, tmpl))
            )
            if phase == "DEFENDING" and tmpl in ((4, 5, 1), (5, 4, 1), (4, 4, 2)):
                cost -= 0.05
            elif phase == "ATTACKING" and tmpl in ((4, 3, 3), (3, 4, 3), (4, 4, 2)):
                cost -= 0.05

            if cost < best_cost:
                best_cost = cost
                best_tmpl = tmpl

        return best_tmpl

    def _classify_team_frame_formation(
        self,
        active_players: list[tuple[int, float, float, bool]],
        u_min_fov: float,
        u_max_fov: float,
        phase: str,
    ) -> tuple[str, dict[int, tuple[str, int]], list[int]]:
        """
        Run spatial K-Means on active outfield players [(tid, u_depth, v_width, is_visible)]
        and classify into 3-line or 4-line (4-2-3-1 / 4-1-4-1) formation.
        """
        n_active = len(active_players)
        if n_active < 4:
            default_fmt = "4-3-3" if phase == "ATTACKING" else "4-4-2"
            return default_fmt, {p[0]: ("MID", 1) for p in active_players if p[3]}, [n_active]

        X_spatial = np.array([[p[1], 0.12 * p[2]] for p in active_players], dtype=np.float64)
        u_vals = X_spatial[:, 0]

        # 1. Standard 3-Line KMeans (DEF, MID, FWD)
        km3 = KMeans(n_clusters=3, n_init=4, max_iter=50, random_state=42)
        raw_lbl3 = km3.fit_predict(X_spatial)
        order3 = np.argsort(km3.cluster_centers_[:, 0])
        rank_map3 = {int(orig): rank for rank, orig in enumerate(order3)}
        sorted_lbl3 = np.array([rank_map3[int(l)] for l in raw_lbl3], dtype=np.int32)

        raw_counts3 = [int(np.sum(sorted_lbl3 == r)) for r in range(3)]
        adj_counts3 = self._compensate_counts_to_10(raw_counts3, u_min_fov, u_max_fov)
        best_3 = self._match_canonical_3line(adj_counts3, phase)

        # 2. Evaluate 4-Line split (4-2-3-1 / 4-1-4-1) when midfield/forward depth bands separate cleanly
        use_4_line = False
        chosen_4: tuple[int, int, int, int] | None = None
        sorted_lbl4: np.ndarray | None = None

        if n_active >= 7:
            km4 = KMeans(n_clusters=4, n_init=4, max_iter=50, random_state=42)
            raw_lbl4 = km4.fit_predict(X_spatial)
            order4 = np.argsort(km4.cluster_centers_[:, 0])
            centers4_u = km4.cluster_centers_[order4, 0]
            rank_map4 = {int(orig): rank for rank, orig in enumerate(order4)}
            cand_lbl4 = np.array([rank_map4[int(l)] for l in raw_lbl4], dtype=np.int32)

            band_gaps = np.diff(centers4_u)
            dm_std = float(np.std(u_vals[cand_lbl4 == 1])) if np.sum(cand_lbl4 == 1) > 1 else 0.0
            am_std = float(np.std(u_vals[cand_lbl4 == 2])) if np.sum(cand_lbl4 == 2) > 1 else 0.0

            if float(np.min(band_gaps)) >= self.min_4line_band_gap_m and dm_std <= 3.4 and am_std <= 3.4:
                raw_counts4 = [int(np.sum(cand_lbl4 == r)) for r in range(4)]
                adj_counts4 = self._compensate_counts_to_10(raw_counts4, u_min_fov, u_max_fov)
                cum4_obs = np.cumsum(adj_counts4)[:-1]

                best_4_cost = 1e9
                best_4_tmpl = FORMATIONS_4_LINE[0]
                for tmpl4 in FORMATIONS_4_LINE:
                    cum4_t = np.cumsum(tmpl4)[:-1]
                    cost4 = float(np.sum(np.abs(cum4_obs - cum4_t))) + 0.5 * float(
                        sum(abs(a - b) for a, b in zip(adj_counts4, tmpl4))
                    )
                    if cost4 < best_4_cost:
                        best_4_cost = cost4
                        best_4_tmpl = tmpl4

                if best_4_cost <= 1.5 and adj_counts4[0] == 4 and adj_counts4[3] == 1:
                    use_4_line = True
                    chosen_4 = best_4_tmpl
                    sorted_lbl4 = cand_lbl4

        player_line_assignments: dict[int, tuple[str, int]] = {}
        if use_4_line and chosen_4 is not None and sorted_lbl4 is not None:
            fmt_str = "-".join(str(x) for x in chosen_4)
            names_4 = ("DEF", "DM", "AM", "FWD")
            for idx, (tid, _, _, is_vis) in enumerate(active_players):
                if is_vis:
                    b_idx = int(sorted_lbl4[idx])
                    player_line_assignments[tid] = (names_4[b_idx], b_idx)
            return fmt_str, player_line_assignments, list(chosen_4)

        fmt_str = "-".join(str(x) for x in best_3)
        names_3 = ("DEF", "MID", "FWD")
        for idx, (tid, _, _, is_vis) in enumerate(active_players):
            if is_vis:
                b_idx = int(sorted_lbl3[idx])
                player_line_assignments[tid] = (names_3[b_idx], b_idx)
        return fmt_str, player_line_assignments, list(best_3)

    @staticmethod
    def _smooth_formation_strings(raw_formations: list[str], window: int) -> list[str]:
        """Apply centered majority-vote mode smoothing on formation labels to eliminate frame jitter."""
        n = len(raw_formations)
        if n == 0:
            return []
        half = window // 2
        smoothed: list[str] = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            counts = Counter(raw_formations[lo:hi])
            prev_lbl = smoothed[-1] if smoothed else raw_formations[i]
            best_lbl = max(counts.keys(), key=lambda k: (counts[k], 1 if k == prev_lbl else 0))
            smoothed.append(best_lbl)
        return smoothed

    def analyze_formations(
        self,
        tracks: dict,
        homographies: list[dict[str, np.ndarray]],
        possession_states: list[dict[str, Any]],
        team_defends_left: dict[int, bool],
        read_from_stub: bool = True,
        stub_path: str = "stubs/formations_stub.pkl",
    ) -> list[dict[str, Any]]:
        """
        Compute per-frame formations for BOTH the On-Ball (Attacking) and Off-Ball (Defending) teams,
        annotate per-player tactical lines on `tracks`, and cache results to `stubs/formations_stub.pkl`.
        """
        num_frames = min(len(tracks.get("players", [])), len(homographies), len(possession_states))

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get("version") == 2 and len(cached.get("frame_formations", [])) == num_frames:
                per_frame_lines = cached.get("per_frame_player_lines", [])
                for f_idx in range(num_frames):
                    line_map = per_frame_lines[f_idx] if f_idx < len(per_frame_lines) else {}
                    for tid, pdata in tracks["players"][f_idx].items():
                        if pdata.get("role") == "goalkeeper":
                            pdata["tactical_line"] = "GK"
                            pdata["tactical_band_idx"] = -1
                        elif tid in line_map:
                            pdata["tactical_line"], pdata["tactical_band_idx"] = line_map[tid]
                return cached["frame_formations"]

        smoothed_trajs = self._build_smoothed_team_trajectories(tracks, team_defends_left)

        last_seen: dict[int, dict[int, tuple[int, float, float]]] = {1: {}, 2: {}}
        raw_team_fmt: dict[int, list[str]] = {1: [], 2: []}
        visible_counts: dict[int, list[int]] = {1: [], 2: []}
        active_counts: dict[int, list[int]] = {1: [], 2: []}
        per_frame_player_lines: list[dict[int, tuple[str, int]]] = []

        with threadpool_limits(limits=1):
            for f in range(num_frames):
                H_i2p = homographies[f]["H_img_to_pitch"]
                poss = possession_states[f]
                on_ball_team = poss["on_ball_team"]
                frame_line_map: dict[int, tuple[str, int]] = {}

                # Evict any ghost IDs that are currently visible on the opposite team (due to ByteTrack ID swap)
                for tid, pdata in tracks["players"][f].items():
                    t_curr = pdata.get("team_id")
                    if t_curr in (1, 2):
                        other_t = 2 if t_curr == 1 else 1
                        last_seen[other_t].pop(tid, None)

                for team_id in (1, 2):
                    defends_left = team_defends_left.get(team_id, team_id == 1)
                    u_min_fov, u_max_fov = self._camera_fov_depth_span(H_i2p, defends_left)
                    phase = "ATTACKING" if team_id == on_ball_team else "DEFENDING"

                    vis_tids: list[int] = []
                    active_list: list[tuple[int, float, float, bool]] = []

                    for tid, pdata in tracks["players"][f].items():
                        if pdata.get("team_id") != team_id:
                            continue
                        if pdata.get("role") == "goalkeeper":
                            pdata["tactical_line"] = "GK"
                            pdata["tactical_band_idx"] = -1
                            continue
                        if not pdata.get("on_pitch", True):
                            continue
                        uv = smoothed_trajs[team_id].get(tid, {}).get(f)
                        if uv is None:
                            continue
                        vis_tids.append(tid)
                        last_seen[team_id][tid] = (f, uv[0], uv[1])
                        active_list.append((tid, uv[0], uv[1], True))

                    if len(active_list) < 10:
                        ghost_candidates = []
                        for g_tid, (last_f, gu, gv) in list(last_seen[team_id].items()):
                            if g_tid in vis_tids:
                                continue
                            age = f - last_f
                            if age <= self.ghost_ttl_frames:
                                min_d = min(
                                    (np.hypot(gu - vu, gv - vv) for (_, vu, vv, _) in active_list),
                                    default=99.0,
                                )
                                if min_d >= 3.2:
                                    ghost_candidates.append((age, g_tid, gu, gv))
                            else:
                                del last_seen[team_id][g_tid]

                        ghost_candidates.sort(key=lambda item: item[0])
                        for _, g_tid, gu, gv in ghost_candidates[: max(0, 10 - len(active_list))]:
                            active_list.append((g_tid, gu, gv, False))

                    if len(active_list) > 10:
                        active_list = active_list[:10]

                    fmt_str, t_line_map, _ = self._classify_team_frame_formation(
                        active_list, u_min_fov, u_max_fov, phase
                    )
                    raw_team_fmt[team_id].append(fmt_str)
                    visible_counts[team_id].append(len(vis_tids))
                    active_counts[team_id].append(len(active_list))

                    for tid, (line_name, band_idx) in t_line_map.items():
                        tracks["players"][f][tid]["tactical_line"] = line_name
                        tracks["players"][f][tid]["tactical_band_idx"] = band_idx
                        frame_line_map[tid] = (line_name, band_idx)

                per_frame_player_lines.append(frame_line_map)

        sm_fmt_1 = self._smooth_formation_strings(raw_team_fmt[1], self.formation_smooth_window)
        sm_fmt_2 = self._smooth_formation_strings(raw_team_fmt[2], self.formation_smooth_window)

        frame_formations: list[dict[str, Any]] = []
        for f in range(num_frames):
            poss = possession_states[f]
            on_team = poss["on_ball_team"]
            off_team = poss["off_ball_team"]
            team_info = {
                1: {
                    "formation": sm_fmt_1[f],
                    "raw_formation": raw_team_fmt[1][f],
                    "phase": "ATTACKING" if on_team == 1 else "DEFENDING",
                    "visible_outfield": visible_counts[1][f],
                    "active_outfield": active_counts[1][f],
                    "possession_pct": poss["possession_pct"][1],
                },
                2: {
                    "formation": sm_fmt_2[f],
                    "raw_formation": raw_team_fmt[2][f],
                    "phase": "ATTACKING" if on_team == 2 else "DEFENDING",
                    "visible_outfield": visible_counts[2][f],
                    "active_outfield": active_counts[2][f],
                    "possession_pct": poss["possession_pct"][2],
                },
            }
            frame_formations.append(
                {
                    "on_ball_team": on_team,
                    "off_ball_team": off_team,
                    "ball_carrier_id": poss["ball_carrier_id"],
                    "attacking_formation": team_info[on_team]["formation"],
                    "defending_formation": team_info[off_team]["formation"],
                    "teams": team_info,
                }
            )

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(
                    {
                        "version": 2,
                        "frame_formations": frame_formations,
                        "per_frame_player_lines": per_frame_player_lines,
                    },
                    f,
                )

        return frame_formations
