import os
from typing import Any
import cv2
import numpy as np

from .pitch_model import PITCH_LENGTH, PITCH_WIDTH, build_fifa_template_points


class TacticalViewDrawer:
    """Resolution-adaptive broadcast 2D Tactical Pitch Radar (Mini-Map) and registration debug renderer."""

    DEFAULT_PLAYER_COLOR = (235, 235, 235)
    DEFAULT_REFEREE_COLOR = (0, 235, 255)
    BALL_COLOR = (0, 255, 0)

    def __init__(
        self,
        width_fraction: float = 0.186,
        alpha: float = 0.84,
        position: str = "bottom_center",
    ) -> None:
        self.width_fraction = width_fraction
        self.alpha = alpha
        self.position = position
        self.debug_template_pts = build_fifa_template_points()

        # Cache pre-rendered pitch canvas keyed by (frame_w, frame_h)
        self._cached_dims: tuple[int, int] | None = None
        self._px_per_meter: float = 3.4
        self._padding: int = 14
        self._canvas_w: int = 385
        self._canvas_h: int = 259
        self._base_canvas: np.ndarray | None = None

    def _ensure_canvas_for_frame(self, fw: int, fh: int) -> None:
        if self._cached_dims == (fw, fh) and self._base_canvas is not None:
            return
        ui = fh / 1080.0
        inner_w = max(160, int(round(self.width_fraction * fw)))
        self._px_per_meter = inner_w / PITCH_LENGTH
        inner_h = int(round(PITCH_WIDTH * self._px_per_meter))
        self._padding = max(6, int(round(14 * ui)))
        self._canvas_w = inner_w + 2 * self._padding
        self._canvas_h = inner_h + 2 * self._padding
        self._base_canvas = self._build_pitch_canvas(ui)
        self._cached_dims = (fw, fh)

    def _m2px(self, x_m: float, y_m: float) -> tuple[int, int]:
        px = int(round(self._padding + x_m * self._px_per_meter))
        py = int(round(self._padding + y_m * self._px_per_meter))
        return px, py

    def _build_pitch_canvas(self, ui: float) -> np.ndarray:
        canvas = np.full((self._canvas_h, self._canvas_w, 3), (24, 58, 28), dtype=np.uint8)
        lw = max(1, int(round(1.0 * ui)))

        num_stripes = 10
        for i in range(num_stripes):
            x0, y0 = self._m2px(i * (PITCH_LENGTH / num_stripes), 0.0)
            x1, y1 = self._m2px((i + 1) * (PITCH_LENGTH / num_stripes), PITCH_WIDTH)
            shade = (36, 98, 44) if i % 2 == 0 else (31, 86, 38)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), shade, cv2.FILLED)

        line_col = (235, 235, 235)
        cv2.rectangle(canvas, self._m2px(0.0, 0.0), self._m2px(PITCH_LENGTH, PITCH_WIDTH), line_col, lw, cv2.LINE_AA)
        cv2.line(canvas, self._m2px(52.5, 0.0), self._m2px(52.5, PITCH_WIDTH), line_col, lw, cv2.LINE_AA)

        center_px = self._m2px(52.5, 34.0)
        r_circle = max(4, int(round(9.15 * self._px_per_meter)))
        spot_r = max(1, int(round(2.0 * ui)))
        cv2.circle(canvas, center_px, r_circle, line_col, lw, cv2.LINE_AA)
        cv2.circle(canvas, center_px, spot_r, line_col, -1, cv2.LINE_AA)

        cv2.rectangle(canvas, self._m2px(0.0, 13.84), self._m2px(16.5, 54.16), line_col, lw, cv2.LINE_AA)
        cv2.rectangle(canvas, self._m2px(0.0, 24.84), self._m2px(5.5, 43.16), line_col, lw, cv2.LINE_AA)
        cv2.circle(canvas, self._m2px(11.0, 34.0), spot_r, line_col, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, self._m2px(11.0, 34.0), (r_circle, r_circle), 0.0, -53.0, 53.0, line_col, lw, cv2.LINE_AA)

        cv2.rectangle(canvas, self._m2px(88.5, 13.84), self._m2px(105.0, 54.16), line_col, lw, cv2.LINE_AA)
        cv2.rectangle(canvas, self._m2px(99.5, 24.84), self._m2px(105.0, 43.16), line_col, lw, cv2.LINE_AA)
        cv2.circle(canvas, self._m2px(94.0, 34.0), spot_r, line_col, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, self._m2px(94.0, 34.0), (r_circle, r_circle), 0.0, 127.0, 233.0, line_col, lw, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (self._canvas_w - 1, self._canvas_h - 1), (180, 180, 180), lw)
        return canvas

    def _draw_camera_fov_on_radar(
        self,
        radar: np.ndarray,
        H_img_to_pitch: np.ndarray,
        frame_w: int,
        frame_h: int,
        lw: int,
    ) -> None:
        top_y = int(0.18 * frame_h)
        boundary_img = np.array(
            [
                [0.0, float(top_y)],
                [0.5 * frame_w, float(top_y)],
                [float(frame_w), float(top_y)],
                [float(frame_w), float(frame_h)],
                [0.5 * frame_w, float(frame_h)],
                [0.0, float(frame_h)],
            ],
            dtype=np.float64,
        )
        ones = np.ones((len(boundary_img), 1), dtype=np.float64)
        hom = np.hstack([boundary_img, ones]) @ H_img_to_pitch.T
        if np.any(hom[:, 2] <= 1e-4):
            return

        pts_m = hom[:, :2] / hom[:, 2:3]
        pts_m[:, 0] = np.clip(pts_m[:, 0], -3.0, PITCH_LENGTH + 3.0)
        pts_m[:, 1] = np.clip(pts_m[:, 0 * 0 + 1], -3.0, PITCH_WIDTH + 3.0)
        poly_px = np.array([self._m2px(xm, ym) for xm, ym in pts_m], dtype=np.int32)

        fov_layer = radar.copy()
        cv2.fillPoly(fov_layer, [poly_px], (190, 230, 195))
        cv2.addWeighted(fov_layer, 0.16, radar, 0.84, 0.0, dst=radar)
        cv2.polylines(radar, [poly_px], isClosed=True, color=(210, 240, 210), thickness=lw, lineType=cv2.LINE_AA)

    def draw_frame_radar(
        self,
        frame: np.ndarray,
        frame_num: int,
        tracks: dict,
        homographies: list[dict[str, Any]] | None = None,
        referee_color: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """Render the resolution-scaled semi-transparent 2D Tactical Radar onto a broadcast frame."""
        fh, fw = frame.shape[:2]
        ui = fh / 1080.0
        lw = max(1, int(round(1.0 * ui)))
        self._ensure_canvas_for_frame(fw, fh)
        radar = self._base_canvas.copy()
        ref_col = referee_color if referee_color is not None else self.DEFAULT_REFEREE_COLOR

        if homographies is not None and frame_num < len(homographies):
            self._draw_camera_fov_on_radar(radar, homographies[frame_num]["H_img_to_pitch"], fw, fh, lw)

        # Draw very subtle structural formation line connectors per team (alpha = 0.28)
        player_dict = tracks.get("players", [])[frame_num] if frame_num < len(tracks.get("players", [])) else {}
        bands_by_team: dict[tuple[int, int], list[tuple[float, tuple[int, int], tuple[int, int, int]]]] = {}
        for pdata in player_dict.values():
            if not pdata.get("on_pitch", True) or pdata.get("role") != "player":
                continue
            pt = pdata.get("position_transformed")
            band_idx = pdata.get("tactical_band_idx")
            team_id = pdata.get("team_id")
            if pt is None or band_idx is None or band_idx < 0 or team_id not in (1, 2):
                continue
            col = pdata.get("team_color", self.DEFAULT_PLAYER_COLOR)
            bands_by_team.setdefault((int(team_id), int(band_idx)), []).append((pt[1], self._m2px(pt[0], pt[1]), col))

        if bands_by_team:
            line_layer = radar.copy()
            for _, members in bands_by_team.items():
                if len(members) >= 2:
                    members_sorted = sorted(members, key=lambda m: m[0])
                    poly_pts = np.array([m[1] for m in members_sorted], dtype=np.int32)
                    cv2.polylines(
                        line_layer,
                        [poly_pts],
                        isClosed=False,
                        color=members_sorted[0][2],
                        thickness=lw,
                        lineType=cv2.LINE_AA,
                    )
            cv2.addWeighted(line_layer, 0.28, radar, 0.72, 0.0, dst=radar)

        # Draw Referees
        r_ref_out = max(2, int(round(4.0 * ui)))
        r_ref_in = max(1, int(round(3.0 * ui)))
        ref_lum = 0.114 * ref_col[0] + 0.587 * ref_col[1] + 0.299 * ref_col[2]
        ref_rim = (235, 235, 235) if ref_lum < 55.0 else (0, 0, 0)

        ref_dict = tracks.get("referees", [])[frame_num] if frame_num < len(tracks.get("referees", [])) else {}
        for rdata in ref_dict.values():
            if not rdata.get("on_pitch", True):
                continue
            pt = rdata.get("position_transformed")
            if pt is not None:
                px, py = self._m2px(pt[0], pt[1])
                cv2.circle(radar, (px, py), r_ref_out, ref_rim, -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), r_ref_in, ref_col, -1, cv2.LINE_AA)

        # Draw Players & Goalkeepers (GK uses same marker color as team's outfield players)
        r_p_out = max(3, int(round(5.0 * ui)))
        r_p_in = max(2, int(round(4.0 * ui)))
        r_carrier = max(4, int(round(7.0 * ui)))

        for pdata in player_dict.values():
            if not pdata.get("on_pitch", True):
                continue
            pt = pdata.get("position_transformed")
            if pt is None:
                continue
            col = pdata.get("team_color", self.DEFAULT_PLAYER_COLOR)
            px, py = self._m2px(pt[0], pt[1])
            if pdata.get("has_ball", False):
                cv2.circle(radar, (px, py), r_carrier, (0, 185, 255), lw, cv2.LINE_AA)
            cv2.circle(radar, (px, py), r_p_out, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), r_p_in, col, -1, cv2.LINE_AA)

        # Draw Ball on top with crisp white halo
        r_b_out = max(3, int(round(5.0 * ui)))
        r_b_in = max(2, int(round(3.0 * ui)))
        ball_dict = tracks.get("ball", [])[frame_num] if frame_num < len(tracks.get("ball", [])) else {}
        for bdata in ball_dict.values():
            pt = bdata.get("position_transformed")
            if pt is not None:
                px, py = self._m2px(pt[0], pt[1])
                cv2.circle(radar, (px, py), r_b_out, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), r_b_in, self.BALL_COLOR, -1, cv2.LINE_AA)

        margin = max(8, int(round(18 * ui)))
        if self.position == "top_left":
            x0, y0 = margin, margin
        elif self.position == "bottom_right":
            x0, y0 = fw - self._canvas_w - margin, fh - self._canvas_h - margin
        else:
            x0 = (fw - self._canvas_w) // 2
            y0 = fh - self._canvas_h - margin

        if x0 >= 0 and y0 >= 0 and x0 + self._canvas_w <= fw and y0 + self._canvas_h <= fh:
            roi = frame[y0 : y0 + self._canvas_h, x0 : x0 + self._canvas_w]
            cv2.addWeighted(radar, self.alpha, roi, 1.0 - self.alpha, 0.0, dst=roi)
        return frame

    def export_registration_debug_frames(
        self,
        video_path: str,
        homographies: list[dict[str, Any]],
        frame_indices: list[int],
        output_dir: str = "output_videos",
        prefix: str = "m2_debug_registration",
    ) -> list[str]:
        """Export standalone debug frames showing the projected FIFA pitch wireframe on raw broadcast frames."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        saved_paths: list[str] = []

        ones = np.ones((len(self.debug_template_pts), 1), dtype=np.float64)
        template_hom = np.hstack([self.debug_template_pts, ones])

        try:
            for idx in frame_indices:
                if idx >= len(homographies):
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                H_p2i = homographies[idx]["H_pitch_to_img"]
                proj = template_hom @ H_p2i.T
                valid = proj[:, 2] > 1e-3
                px = (proj[valid, 0] / proj[valid, 2]).astype(np.int32)
                py = (proj[valid, 1] / proj[valid, 2]).astype(np.int32)

                h, w = frame.shape[:2]
                dot_r = max(2, int(round(3.0 * h / 1080.0)))
                for x, y in zip(px, py):
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (int(x), int(y)), dot_r, (255, 255, 0), -1, cv2.LINE_AA)

                out_path = os.path.join(output_dir, f"{prefix}_{idx}.jpg")
                cv2.imwrite(out_path, frame)
                saved_paths.append(out_path)
        finally:
            cap.release()

        return saved_paths
