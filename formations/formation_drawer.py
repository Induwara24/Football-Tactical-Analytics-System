from typing import Any
import cv2
import numpy as np


class FormationDrawer:
    """Resolution-scaled broadcast Top-Left Tactical HUD displaying live Attacking & Defending formations."""

    def __init__(
        self,
        alpha: float = 0.84,
        team_names: dict[int, str] | None = None,
    ) -> None:
        self.alpha = alpha
        self.team_names = team_names or {1: "TEAM 1", 2: "TEAM 2"}

    def draw_formation_hud(
        self,
        frame: np.ndarray,
        frame_num: int,
        formations: list[dict[str, Any]],
        team_colors: dict[int, tuple[int, int, int]],
    ) -> np.ndarray:
        """Overlay the resolution-scaled On-Ball (Attacking) & Off-Ball (Defending) tactical formation card."""
        if frame_num >= len(formations):
            return frame

        fh, fw = frame.shape[:2]
        ui = fh / 1080.0
        s = lambda v: max(1, int(round(v * ui)))

        fstate = formations[frame_num]
        teams_data = fstate["teams"]

        card_w = s(456)
        card_h = s(114)
        x0, y0 = s(24), s(22)

        if x0 + card_w > fw or y0 + card_h > fh:
            return frame

        roi = frame[y0 : y0 + card_h, x0 : x0 + card_w]
        panel = np.full((card_h, card_w, 3), (24, 28, 34), dtype=np.uint8)

        cv2.rectangle(panel, (0, 0), (card_w, s(26)), (36, 42, 52), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        lw_thin = max(1, int(round(1.0 * ui)))
        lw_bold = max(1, int(round(2.0 * ui)))

        cv2.putText(
            panel,
            "LIVE TACTICAL FORMATIONS",
            (s(12), s(18)),
            font,
            0.46 * ui,
            (195, 205, 215),
            lw_thin,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "SHAPE    POSS",
            (s(322), s(18)),
            font,
            0.42 * ui,
            (165, 175, 185),
            lw_thin,
            cv2.LINE_AA,
        )

        for idx, team_id in enumerate((1, 2)):
            tdata = teams_data[team_id]
            is_attacking = tdata["phase"] == "ATTACKING"
            fmt_str = tdata["formation"]
            poss_pct = tdata["possession_pct"]
            kit_col = team_colors.get(team_id, (230, 230, 230))

            ry0 = s(30 + idx * 41)
            ry1 = ry0 + s(36)

            row_bg = (34, 46, 40) if is_attacking else (28, 32, 40)
            cv2.rectangle(panel, (s(6), ry0), (card_w - s(6), ry1), row_bg, cv2.FILLED)

            cv2.rectangle(panel, (s(10), ry0 + s(5)), (s(18), ry1 - s(5)), kit_col, cv2.FILLED)
            cv2.rectangle(panel, (s(10), ry0 + s(5)), (s(18), ry1 - s(5)), (15, 15, 15), lw_thin)

            t_name = self.team_names.get(team_id, f"TEAM {team_id}")
            cv2.putText(
                panel,
                t_name,
                (s(27), ry0 + s(23)),
                font,
                0.54 * ui,
                (245, 245, 245),
                lw_bold,
                cv2.LINE_AA,
            )

            pill_x0, pill_y0 = s(114), ry0 + s(6)
            pill_x1, pill_y1 = s(292), ry1 - s(6)
            if is_attacking:
                pill_bg = (48, 128, 68)
                pill_border = (95, 215, 120)
                phase_txt = "ON-BALL : ATTACKING"
                txt_col = (245, 255, 245)
            else:
                pill_bg = (68, 56, 46)
                pill_border = (135, 118, 98)
                phase_txt = "OFF-BALL : DEFENDING"
                txt_col = (220, 220, 225)

            cv2.rectangle(panel, (pill_x0, pill_y0), (pill_x1, pill_y1), pill_bg, cv2.FILLED)
            cv2.rectangle(panel, (pill_x0, pill_y0), (pill_x1, pill_y1), pill_border, lw_thin)
            cv2.putText(
                panel,
                phase_txt,
                (pill_x0 + s(9), ry0 + s(22)),
                font,
                0.42 * ui,
                txt_col,
                lw_thin,
                cv2.LINE_AA,
            )

            cv2.putText(
                panel,
                fmt_str,
                (s(306), ry0 + s(24)),
                font,
                0.60 * ui,
                (255, 255, 255) if is_attacking else (215, 225, 235),
                lw_bold,
                cv2.LINE_AA,
            )

            poss_str = f"{int(round(poss_pct)):2d}%"
            cv2.putText(
                panel,
                poss_str,
                (s(404), ry0 + s(23)),
                font,
                0.50 * ui,
                (195, 210, 220),
                lw_thin,
                cv2.LINE_AA,
            )

        cv2.rectangle(panel, (0, 0), (card_w - 1, card_h - 1), (110, 120, 130), lw_thin)
        cv2.addWeighted(panel, self.alpha, roi, 1.0 - self.alpha, 0.0, dst=roi)
        return frame
