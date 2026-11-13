from typing import Any
import cv2
import numpy as np


class FormationDrawer:
    """Renders a sleek broadcast Top-Left Tactical HUD displaying live Attacking & Defending formations."""

    def __init__(
        self,
        x0: int = 24,
        y0: int = 22,
        alpha: float = 0.84,
        team_names: dict[int, str] | None = None,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.alpha = alpha
        self.team_names = team_names or {1: "TEAM 1", 2: "TEAM 2"}

    def draw_formation_hud(
        self,
        frame: np.ndarray,
        frame_num: int,
        formations: list[dict[str, Any]],
        team_colors: dict[int, tuple[int, int, int]],
    ) -> np.ndarray:
        """Overlay the live On-Ball (Attacking) & Off-Ball (Defending) tactical formation panel."""
        if frame_num >= len(formations):
            return frame

        fstate = formations[frame_num]
        teams_data = fstate["teams"]

        card_w = 456
        card_h = 114
        x0, y0 = self.x0, self.y0

        fh, fw = frame.shape[:2]
        if x0 + card_w > fw or y0 + card_h > fh:
            return frame

        roi = frame[y0 : y0 + card_h, x0 : x0 + card_w]
        panel = np.full((card_h, card_w, 3), (24, 28, 34), dtype=np.uint8)

        # Header strip
        cv2.rectangle(panel, (0, 0), (card_w, 26), (36, 42, 52), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            panel,
            "LIVE TACTICAL FORMATIONS",
            (12, 18),
            font,
            0.46,
            (195, 205, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "SHAPE    POSS",
            (322, 18),
            font,
            0.42,
            (165, 175, 185),
            1,
            cv2.LINE_AA,
        )

        # Render Team 1 (Row 0) and Team 2 (Row 1) in consistent visual positions
        for idx, team_id in enumerate((1, 2)):
            tdata = teams_data[team_id]
            is_attacking = tdata["phase"] == "ATTACKING"
            fmt_str = tdata["formation"]
            poss_pct = tdata["possession_pct"]
            kit_col = team_colors.get(team_id, (230, 230, 230))

            ry0 = 30 + idx * 41
            ry1 = ry0 + 36

            # Subtle row background highlight when On-Ball
            row_bg = (34, 46, 40) if is_attacking else (28, 32, 40)
            cv2.rectangle(panel, (6, ry0), (card_w - 6, ry1), row_bg, cv2.FILLED)

            # Kit color vertical swatch bar (with dark outline for white kits)
            cv2.rectangle(panel, (10, ry0 + 5), (18, ry1 - 5), kit_col, cv2.FILLED)
            cv2.rectangle(panel, (10, ry0 + 5), (18, ry1 - 5), (15, 15, 15), 1)

            # Team title
            t_name = self.team_names.get(team_id, f"TEAM {team_id}")
            cv2.putText(
                panel,
                t_name,
                (27, ry0 + 23),
                font,
                0.54,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            # Dynamic Tactical Phase Badge: ON-BALL (ATTACK) vs OFF-BALL (DEFEND)
            pill_x0, pill_y0 = 114, ry0 + 6
            pill_x1, pill_y1 = 292, ry1 - 6
            if is_attacking:
                pill_bg = (48, 128, 68)       # Vibrant emerald (BGR)
                pill_border = (95, 215, 120)
                phase_txt = "ON-BALL : ATTACKING"
                txt_col = (245, 255, 245)
            else:
                pill_bg = (68, 56, 46)        # Cool slate-amber (BGR)
                pill_border = (135, 118, 98)
                phase_txt = "OFF-BALL : DEFENDING"
                txt_col = (220, 220, 225)

            cv2.rectangle(panel, (pill_x0, pill_y0), (pill_x1, pill_y1), pill_bg, cv2.FILLED)
            cv2.rectangle(panel, (pill_x0, pill_y0), (pill_x1, pill_y1), pill_border, 1)
            cv2.putText(
                panel,
                phase_txt,
                (pill_x0 + 9, ry0 + 22),
                font,
                0.42,
                txt_col,
                1,
                cv2.LINE_AA,
            )

            # Live Formation string (e.g., 4-3-3 / 4-2-3-1 / 4-5-1)
            cv2.putText(
                panel,
                fmt_str,
                (306, ry0 + 24),
                font,
                0.60,
                (255, 255, 255) if is_attacking else (215, 225, 235),
                2,
                cv2.LINE_AA,
            )

            # Possession %
            poss_str = f"{int(round(poss_pct)):2d}%"
            cv2.putText(
                panel,
                poss_str,
                (404, ry0 + 23),
                font,
                0.50,
                (195, 210, 220),
                1,
                cv2.LINE_AA,
            )

        # Outer border
        cv2.rectangle(panel, (0, 0), (card_w - 1, card_h - 1), (110, 120, 130), 1)
        cv2.addWeighted(panel, self.alpha, roi, 1.0 - self.alpha, 0.0, dst=roi)
        return frame

