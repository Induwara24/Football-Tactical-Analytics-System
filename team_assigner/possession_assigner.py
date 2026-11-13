from typing import Any
import numpy as np

from utils import get_center_of_bbox, get_foot_position


class PossessionAssigner:
    """Assigns live ball carrier and hysteresis-smoothed On-Ball (Attacking) vs Off-Ball (Defending) team roles."""

    def __init__(
        self,
        max_control_dist_m: float = 2.1,
        max_control_dist_px: float = 68.0,
        confirm_window: int = 14,
        confirm_min_frames: int = 5,
        mode_smooth_window: int = 19,
    ) -> None:
        self.max_control_dist_m = max_control_dist_m
        self.max_control_dist_px = max_control_dist_px
        self.confirm_window = confirm_window
        self.confirm_min_frames = confirm_min_frames
        self.mode_smooth_window = mode_smooth_window

    def assign_ball_possession(self, tracks: dict) -> list[dict[str, Any]]:
        """Annotate `has_ball` on players and return frame-indexed on-ball/off-ball possession states."""
        num_frames = len(tracks.get("players", []))
        if num_frames == 0:
            return []

        raw_carrier_ids: list[int | None] = [None] * num_frames
        raw_touch_teams: list[int | None] = [None] * num_frames

        for f in range(num_frames):
            # Reset has_ball flag
            for pdata in tracks["players"][f].values():
                pdata["has_ball"] = False

            ball_entry = tracks.get("ball", [])[f].get(1) if f < len(tracks.get("ball", [])) else None
            if not ball_entry:
                continue

            ball_pt = ball_entry.get("position_transformed")
            ball_bbox = ball_entry.get("bbox")
            if ball_pt is None or ball_bbox is None:
                continue

            bx_center, _ = get_center_of_bbox(ball_bbox)
            by_bottom = float(ball_bbox[3])

            best_score = 1e9
            best_tid: int | None = None
            best_team: int | None = None

            for tid, pdata in tracks["players"][f].items():
                if not pdata.get("on_pitch", True):
                    continue
                pt = pdata.get("position_transformed")
                if pt is None:
                    continue

                fx, fy = get_foot_position(pdata["bbox"])
                dist_m = float(np.hypot(pt[0] - ball_pt[0], pt[1] - ball_pt[1]))
                dist_px = float(np.hypot(fx - bx_center, fy - by_bottom))

                if dist_m <= self.max_control_dist_m and dist_px <= self.max_control_dist_px:
                    # Combined metric + pixel proximity score handles close 1v1 duels cleanly
                    score = dist_m + 0.015 * dist_px
                    if score < best_score:
                        best_score = score
                        best_tid = tid
                        best_team = pdata.get("team_id")

            if best_tid is not None and best_team in (1, 2):
                tracks["players"][f][best_tid]["has_ball"] = True
                raw_carrier_ids[f] = best_tid
                raw_touch_teams[f] = int(best_team)

        # Initial team with first confirmed touch (defaults to Team 1)
        initial_team = next((t for t in raw_touch_teams if t in (1, 2)), 1)

        # Stage 1: Causal confirmation filter (carries possession across passes, switches on confirmed turnovers)
        confirmed_team = np.full(num_frames, initial_team, dtype=np.int32)
        active_team = initial_team
        for f in range(num_frames):
            lo = max(0, f - self.confirm_window + 1)
            window_touches = [t for t in raw_touch_teams[lo : f + 1] if t in (1, 2)]
            other_team = 2 if active_team == 1 else 1
            other_count = sum(1 for t in window_touches if t == other_team)
            same_count = sum(1 for t in window_touches if t == active_team)
            if other_count >= self.confirm_min_frames and other_count > same_count:
                active_team = other_team
            confirmed_team[f] = active_team

        # Stage 2: Centered majority mode filter to eliminate brief boundary oscillation
        half = self.mode_smooth_window // 2
        smoothed_team = confirmed_team.copy()
        for f in range(num_frames):
            lo = max(0, f - half)
            hi = min(num_frames, f + half + 1)
            slice_teams = confirmed_team[lo:hi]
            c1 = int(np.sum(slice_teams == 1))
            c2 = int(np.sum(slice_teams == 2))
            smoothed_team[f] = 1 if c1 >= c2 else 2

        # Compute cumulative possession percentages and assemble output records
        team1_frames = 0
        team2_frames = 0
        possession_states: list[dict[str, Any]] = []

        for f in range(num_frames):
            on_ball = int(smoothed_team[f])
            off_ball = 2 if on_ball == 1 else 1
            if on_ball == 1:
                team1_frames += 1
            else:
                team2_frames += 1
            total = team1_frames + team2_frames

            possession_states.append(
                {
                    "on_ball_team": on_ball,
                    "off_ball_team": off_ball,
                    "ball_carrier_id": raw_carrier_ids[f],
                    "possession_pct": {
                        1: round(100.0 * team1_frames / total, 1),
                        2: round(100.0 * team2_frames / total, 1),
                    },
                }
            )

        return possession_states

