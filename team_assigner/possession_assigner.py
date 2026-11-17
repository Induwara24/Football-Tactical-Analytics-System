from typing import Any
import numpy as np

from utils import FrameScaleContext, get_center_of_bbox, get_foot_position


class PossessionAssigner:
    """
    Relative-geometry & FPS-adaptive ball possession tracker.
    Scales pixel control proximity dynamically from each player's perspective bounding-box height
    and scales turnover confirmation / mode windows proportionally to video FPS.
    """

    def __init__(
        self,
        max_control_dist_m: float = 2.2,
        max_control_player_h_mult: float = 1.25,
        confirm_seconds: float = 0.56,
        confirm_min_seconds: float = 0.20,
        mode_smooth_seconds: float = 0.76,
    ) -> None:
        self.max_control_dist_m = max_control_dist_m
        self.max_control_player_h_mult = max_control_player_h_mult
        self.confirm_seconds = confirm_seconds
        self.confirm_min_seconds = confirm_min_seconds
        self.mode_smooth_seconds = mode_smooth_seconds

    def assign_ball_possession(
        self,
        tracks: dict,
        scale_ctx: FrameScaleContext | None = None,
    ) -> list[dict[str, Any]]:
        """Annotate `has_ball` on players and return frame-indexed on-ball/off-ball possession states."""
        num_frames = len(tracks.get("players", []))
        if num_frames == 0:
            return []

        confirm_win = scale_ctx.frames_for_seconds(self.confirm_seconds, minimum=6) if scale_ctx else 14
        confirm_min = scale_ctx.frames_for_seconds(self.confirm_min_seconds, minimum=3) if scale_ctx else 5
        mode_win = scale_ctx.frames_for_seconds(self.mode_smooth_seconds, minimum=7) if scale_ctx else 19
        if mode_win % 2 == 0:
            mode_win += 1

        raw_carrier_ids: list[int | None] = [None] * num_frames
        raw_touch_teams: list[int | None] = [None] * num_frames

        for f in range(num_frames):
            for pdata in tracks["players"][f].values():
                pdata["has_ball"] = False

            ball_entry = tracks.get("ball", [])[f].get(1) if f < len(tracks.get("ball", [])) else None
            if not ball_entry:
                continue

            ball_pt = ball_entry.get("position_transformed")
            ball_bbox = ball_entry.get("bbox")
            if ball_bbox is None:
                continue

            bx_center, by_center = get_center_of_bbox(ball_bbox)
            by_bottom = float(ball_bbox[3])

            best_score = 1e9
            best_tid: int | None = None
            best_team: int | None = None

            for tid, pdata in tracks["players"][f].items():
                if not pdata.get("in_visual_pitch", True):
                    continue
                pb = pdata["bbox"]
                fx, fy = get_foot_position(pb)
                bh_box = float(pb[3] - pb[1])
                player_h = max(
                    bh_box,
                    scale_ctx.player_h_at(fy) if scale_ctx is not None else 50.0,
                )
                max_px_gate = self.max_control_player_h_mult * player_h
                # Measure pixel distance to player's lower-leg segment [y1 + 0.55*bh .. y2] for bouncing balls
                leg_y = float(np.clip(by_bottom, pb[1] + 0.55 * bh_box, pb[3]))
                dist_px = min(
                    float(np.hypot(fx - bx_center, fy - by_bottom)),
                    float(np.hypot(fx - bx_center, leg_y - by_center)),
                )
                if dist_px > max_px_gate:
                    continue

                pt = pdata.get("position_transformed")
                if pt is not None and ball_pt is not None:
                    dist_m = float(np.hypot(pt[0] - ball_pt[0], pt[1] - ball_pt[1]))
                    if dist_m > self.max_control_dist_m:
                        continue
                    score = dist_m + 0.75 * (dist_px / max(1.0, player_h))
                else:
                    score = 2.0 * (dist_px / max(1.0, player_h))

                if score < best_score:
                    best_score = score
                    best_tid = tid
                    best_team = pdata.get("team_id")

            if best_tid is not None and best_team in (1, 2):
                tracks["players"][f][best_tid]["has_ball"] = True
                raw_carrier_ids[f] = best_tid
                raw_touch_teams[f] = int(best_team)

        initial_team = next((t for t in raw_touch_teams if t in (1, 2)), 1)

        confirmed_team = np.full(num_frames, initial_team, dtype=np.int32)
        active_team = initial_team
        for f in range(num_frames):
            lo = max(0, f - confirm_win + 1)
            window_touches = [t for t in raw_touch_teams[lo : f + 1] if t in (1, 2)]
            other_team = 2 if active_team == 1 else 1
            other_count = sum(1 for t in window_touches if t == other_team)
            same_count = sum(1 for t in window_touches if t == active_team)
            if other_count >= confirm_min and other_count > same_count:
                active_team = other_team
            confirmed_team[f] = active_team

        half = mode_win // 2
        smoothed_team = confirmed_team.copy()
        for f in range(num_frames):
            lo = max(0, f - half)
            hi = min(num_frames, f + half + 1)
            slice_teams = confirmed_team[lo:hi]
            c1 = int(np.sum(slice_teams == 1))
            c2 = int(np.sum(slice_teams == 2))
            smoothed_team[f] = 1 if c1 >= c2 else 2

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
