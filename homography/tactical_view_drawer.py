import os
import cv2
import numpy as np

from .pitch_model import PITCH_LENGTH, PITCH_WIDTH, build_fifa_template_points


class TacticalViewDrawer:
    """Broadcast-style 2D Tactical Pitch Radar (Mini-Map) and separate debug wireframe renderer."""

    DEFAULT_PLAYER_COLOR = (0, 0, 255)        # Red (BGR) until M3 TeamAssigner assigns kit colors
    DEFAULT_GOALKEEPER_COLOR = (255, 191, 0)  # Sky Blue (BGR)
    REFEREE_COLOR = (0, 255, 255)             # Yellow (BGR)
    BALL_COLOR = (0, 255, 0)                  # Green (BGR)

    def __init__(
        self,
        px_per_meter: float = 3.4,
        padding: int = 14,
        alpha: float = 0.84,
        position: str = "bottom_center",
    ) -> None:
        self.px_per_meter = px_per_meter
        self.padding = padding
        self.alpha = alpha
        self.position = position

        self.inner_w = int(round(PITCH_LENGTH * px_per_meter))
        self.inner_h = int(round(PITCH_WIDTH * px_per_meter))
        self.canvas_w = self.inner_w + 2 * padding
        self.canvas_h = self.inner_h + 2 * padding

        self.base_canvas = self._build_pitch_canvas()
        self.debug_template_pts = build_fifa_template_points()

    def _m2px(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Convert metric pitch coordinates (X_m, Y_m) to radar canvas pixel coordinates."""
        px = int(round(self.padding + x_m * self.px_per_meter))
        py = int(round(self.padding + y_m * self.px_per_meter))
        return px, py

    def _build_pitch_canvas(self) -> np.ndarray:
        """Pre-render a striped emerald FIFA 105m x 68m tactical pitch canvas."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), (24, 58, 28), dtype=np.uint8)

        # Alternating subtle grass stripes across 10 vertical zones
        num_stripes = 10
        for i in range(num_stripes):
            x0, y0 = self._m2px(i * (PITCH_LENGTH / num_stripes), 0.0)
            x1, y1 = self._m2px((i + 1) * (PITCH_LENGTH / num_stripes), PITCH_WIDTH)
            shade = (36, 98, 44) if i % 2 == 0 else (31, 86, 38)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), shade, cv2.FILLED)

        line_col = (235, 235, 235)
        lw = 1

        # Outer boundary & halfway line
        cv2.rectangle(canvas, self._m2px(0.0, 0.0), self._m2px(PITCH_LENGTH, PITCH_WIDTH), line_col, lw, cv2.LINE_AA)
        cv2.line(canvas, self._m2px(52.5, 0.0), self._m2px(52.5, PITCH_WIDTH), line_col, lw, cv2.LINE_AA)

        # Center circle & spot
        center_px = self._m2px(52.5, 34.0)
        r_circle = int(round(9.15 * self.px_per_meter))
        cv2.circle(canvas, center_px, r_circle, line_col, lw, cv2.LINE_AA)
        cv2.circle(canvas, center_px, 2, line_col, -1, cv2.LINE_AA)

        # Left & Right 18-yard penalty boxes + 6-yard goal areas + penalty spots & arcs
        cv2.rectangle(canvas, self._m2px(0.0, 13.84), self._m2px(16.5, 54.16), line_col, lw, cv2.LINE_AA)
        cv2.rectangle(canvas, self._m2px(0.0, 24.84), self._m2px(5.5, 43.16), line_col, lw, cv2.LINE_AA)
        cv2.circle(canvas, self._m2px(11.0, 34.0), 2, line_col, -1, cv2.LINE_AA)
        cv2.ellipse(
            canvas,
            self._m2px(11.0, 34.0),
            (r_circle, r_circle),
            0.0,
            -53.0,
            53.0,
            line_col,
            lw,
            cv2.LINE_AA,
        )

        cv2.rectangle(canvas, self._m2px(88.5, 13.84), self._m2px(105.0, 54.16), line_col, lw, cv2.LINE_AA)
        cv2.rectangle(canvas, self._m2px(99.5, 24.84), self._m2px(105.0, 43.16), line_col, lw, cv2.LINE_AA)
        cv2.circle(canvas, self._m2px(94.0, 34.0), 2, line_col, -1, cv2.LINE_AA)
        cv2.ellipse(
            canvas,
            self._m2px(94.0, 34.0),
            (r_circle, r_circle),
            0.0,
            127.0,
            233.0,
            line_col,
            lw,
            cv2.LINE_AA,
        )

        # Crisp outer border frame
        cv2.rectangle(canvas, (0, 0), (self.canvas_w - 1, self.canvas_h - 1), (180, 180, 180), 1)
        return canvas

    def _draw_camera_fov_on_radar(
        self,
        radar: np.ndarray,
        H_img_to_pitch: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Project visible screen boundary onto the 2D radar as a subtle translucent frustum."""
        # Sample boundary points below the stadium horizon (y from ~160..frame_h)
        top_y = int(0.16 * frame_h)
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
        pts_m[:, 1] = np.clip(pts_m[:, 1], -3.0, PITCH_WIDTH + 3.0)
        poly_px = np.array([self._m2px(xm, ym) for xm, ym in pts_m], dtype=np.int32)

        fov_layer = radar.copy()
        cv2.fillPoly(fov_layer, [poly_px], (190, 230, 195))
        cv2.addWeighted(fov_layer, 0.16, radar, 0.84, 0.0, dst=radar)
        cv2.polylines(radar, [poly_px], isClosed=True, color=(210, 240, 210), thickness=1, lineType=cv2.LINE_AA)

    def draw_frame_radar(
        self,
        frame: np.ndarray,
        frame_num: int,
        tracks: dict,
        homographies: list[dict[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        """Render the semi-transparent 2D Tactical Radar onto a single broadcast frame."""
        fh, fw = frame.shape[:2]
        radar = self.base_canvas.copy()

        if homographies is not None and frame_num < len(homographies):
            self._draw_camera_fov_on_radar(radar, homographies[frame_num]["H_img_to_pitch"], fw, fh)

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
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
            cv2.addWeighted(line_layer, 0.28, radar, 0.72, 0.0, dst=radar)

        # Draw Referees
        ref_dict = tracks.get("referees", [])[frame_num] if frame_num < len(tracks.get("referees", [])) else {}
        for rdata in ref_dict.values():
            pt = rdata.get("position_transformed")
            if pt is not None:
                px, py = self._m2px(pt[0], pt[1])
                cv2.circle(radar, (px, py), 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), 3, self.REFEREE_COLOR, -1, cv2.LINE_AA)

        # Draw Players & Goalkeepers
        for pdata in player_dict.values():
            if not pdata.get("on_pitch", True):
                continue
            pt = pdata.get("position_transformed")
            if pt is None:
                continue
            default_col = (
                self.DEFAULT_GOALKEEPER_COLOR
                if pdata.get("role") == "goalkeeper"
                else self.DEFAULT_PLAYER_COLOR
            )
            col = pdata.get("team_color", default_col)
            px, py = self._m2px(pt[0], pt[1])
            if pdata.get("has_ball", False):
                cv2.circle(radar, (px, py), 7, (0, 185, 255), 1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), 5, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), 4, col, -1, cv2.LINE_AA)

        # Draw Ball on top with crisp white halo
        ball_dict = tracks.get("ball", [])[frame_num] if frame_num < len(tracks.get("ball", [])) else {}
        for bdata in ball_dict.values():
            pt = bdata.get("position_transformed")
            if pt is not None:
                px, py = self._m2px(pt[0], pt[1])
                cv2.circle(radar, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), 3, self.BALL_COLOR, -1, cv2.LINE_AA)

        # Placement on broadcast frame (default: bottom_center)
        margin = 18
        if self.position == "top_left":
            x0, y0 = margin, margin
        elif self.position == "bottom_right":
            x0, y0 = fw - self.canvas_w - margin, fh - self.canvas_h - margin
        else:
            x0 = (fw - self.canvas_w) // 2
            y0 = fh - self.canvas_h - margin

        roi = frame[y0 : y0 + self.canvas_h, x0 : x0 + self.canvas_w]
        cv2.addWeighted(radar, self.alpha, roi, 1.0 - self.alpha, 0.0, dst=roi)
        return frame

    def export_registration_debug_frames(
        self,
        video_path: str,
        homographies: list[dict[str, np.ndarray]],
        frame_indices: list[int],
        output_dir: str = "output_videos",
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
                for x, y in zip(px, py):
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1, cv2.LINE_AA)

                out_path = os.path.join(output_dir, f"m2_debug_registration_{idx}.jpg")
                cv2.imwrite(out_path, frame)
                saved_paths.append(out_path)
        finally:
            cap.release()

        return saved_paths

