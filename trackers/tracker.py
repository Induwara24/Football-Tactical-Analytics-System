import os
import pickle
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from utils import (
    FrameScaleContext,
    PitchColorModel,
    ensure_ultralytics_config_dir,
    get_bbox_width,
    get_center_of_bbox,
    get_foot_position,
    get_video_properties,
    iter_video_batches,
    stream_annotated_video,
)

ensure_ultralytics_config_dir()
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from .ball_tracker import BallTracker


def _make_bytetrack_args(
    track_high_thresh: float = 0.22,
    track_low_thresh: float = 0.08,
    track_buffer: int = 30,
) -> SimpleNamespace:
    return SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=float(track_high_thresh),
        track_low_thresh=float(track_low_thresh),
        new_track_thresh=float(track_high_thresh),
        track_buffer=int(track_buffer),
        match_thresh=0.8,
        fuse_score=True,
    )


class Tracker:
    """Scale-adaptive YOLO detector + ByteTrack multi-object tracker with resolution-scaled broadcast overlays."""

    DEFAULT_PLAYER_COLOR = (235, 235, 235)
    DEFAULT_REFEREE_COLOR = (0, 235, 255)  # Dynamically overridden per video via TeamAssigner contrast color
    BALL_COLOR = (0, 255, 0)               # Green pointer for Ball (per user preference)

    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.tracker = BYTETracker(_make_bytetrack_args())
        self.ball_tracker = BallTracker()
        self.referee_color: tuple[int, int, int] = self.DEFAULT_REFEREE_COLOR

    @staticmethod
    def _select_person_imgsz(width: int) -> int:
        """Choose CPU/GPU-efficient stride-32 person inference resolution (SAHI handles small ball tiles)."""
        if width >= 1600:
            return 736
        return 640

    def detect_frames(self, frames: list[np.ndarray], batch_size: int = 16) -> list:
        """Run batched scale-adaptive YOLO detection across an in-memory list of video frames."""
        if not frames:
            return []
        imgsz = self._select_person_imgsz(frames[0].shape[1])
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], imgsz=imgsz, conf=0.08, verbose=False)
            detections.extend(detections_batch)
        return detections

    def _iter_detection_batches(
        self,
        video_source: str | list[np.ndarray],
        batch_size: int = 16,
    ):
        """Yield YOLO Results batches from either a video file path (streaming) or a frame list."""
        if isinstance(video_source, str):
            props = get_video_properties(video_source)
            imgsz = self._select_person_imgsz(props["width"])
            for frame_batch in iter_video_batches(video_source, batch_size=batch_size):
                yield self.model.predict(frame_batch, imgsz=imgsz, conf=0.08, verbose=False)
        else:
            if not video_source:
                return
            imgsz = self._select_person_imgsz(video_source[0].shape[1])
            for i in range(0, len(video_source), batch_size):
                yield self.model.predict(video_source[i : i + batch_size], imgsz=imgsz, conf=0.08, verbose=False)

    def get_object_tracks(
        self,
        video_source: str | list[np.ndarray],
        pitch_model: PitchColorModel | None = None,
        read_from_stub: bool = False,
        stub_path: str | None = None,
        batch_size: int = 16,
    ) -> dict:
        """Return frame-indexed player/goalkeeper, referee, and ball tracks, calibrated dynamically per video."""
        expected_frames = None
        fps = 25.0
        if isinstance(video_source, str):
            props = get_video_properties(video_source)
            expected_frames = props["frame_count"]
            fps = props["fps"]
        else:
            expected_frames = len(video_source)

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and len(cached.get("players", [])) == expected_frames:
                return cached

        # Configure ByteTrack buffer dynamically from video FPS (~1.2s occlusion memory)
        track_buffer = max(20, int(round(1.2 * fps)))
        self.tracker = BYTETracker(_make_bytetrack_args(track_high_thresh=0.20, track_low_thresh=0.08, track_buffer=track_buffer))

        tracks: dict[str, list] = {
            "players": [],
            "referees": [],
            "ball": [],
            "raw_ball_rows": [],
        }

        total_appearances: dict[int, int] = defaultdict(int)
        gk_appearances: dict[int, int] = defaultdict(int)

        frame_num = 0
        for detections_batch in self._iter_detection_batches(video_source, batch_size=batch_size):
            for detection in detections_batch:
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                gk_cls_id = cls_names_inv["goalkeeper"]
                player_cls_id = cls_names_inv["player"]
                referee_cls_id = cls_names_inv["referee"]
                ball_cls_id = cls_names_inv["ball"]

                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                tracks["raw_ball_rows"].append([])

                boxes_np = detection.boxes.cpu().numpy().data.copy()
                if len(boxes_np) > 0:
                    orig_cls = boxes_np[:, 5].astype(int).copy()

                    ball_mask = orig_cls == ball_cls_id
                    if np.any(ball_mask):
                        ball_candidates = boxes_np[ball_mask]
                        tracks["raw_ball_rows"][frame_num] = ball_candidates.tolist()
                        best_idx = int(np.argmax(ball_candidates[:, 4]))
                        tracks["ball"][frame_num][1] = {"bbox": ball_candidates[best_idx, :4].tolist()}

                    person_mask = orig_cls != ball_cls_id
                    if np.any(person_mask):
                        person_boxes_np = boxes_np[person_mask].copy()
                        person_orig_cls = orig_cls[person_mask]
                        person_boxes_np[person_orig_cls == gk_cls_id, 5] = float(player_cls_id)
                        tracked_rows = self.tracker.update(Boxes(person_boxes_np, detection.orig_shape), None)
                    else:
                        person_orig_cls = np.empty(0, dtype=int)
                        tracked_rows = self.tracker.update(
                            Boxes(np.empty((0, 6), dtype=np.float32), detection.orig_shape), None
                        )

                    for row in tracked_rows:
                        bbox = row[:4].tolist()
                        track_id = int(row[4])
                        cls_id = int(row[6])
                        det_idx = int(row[7])

                        if cls_id == player_cls_id:
                            total_appearances[track_id] += 1
                            is_gk_frame = 0 <= det_idx < len(person_orig_cls) and person_orig_cls[det_idx] == gk_cls_id
                            if is_gk_frame:
                                gk_appearances[track_id] += 1

                            tracks["players"][frame_num][track_id] = {
                                "bbox": bbox,
                                "role": "goalkeeper" if is_gk_frame else "player",
                                "in_visual_pitch": True,
                            }
                        elif cls_id == referee_cls_id:
                            tracks["referees"][frame_num][track_id] = {
                                "bbox": bbox,
                                "in_visual_pitch": True,
                            }
                else:
                    self.tracker.update(Boxes(np.empty((0, 6), dtype=np.float32), detection.orig_shape), None)

                frame_num += 1
                if frame_num % 100 == 0:
                    print(f"Processed {frame_num} frames for detection & tracking...")

        gk_track_ids = {
            tid
            for tid, total in total_appearances.items()
            if total > 0 and (gk_appearances[tid] / total) >= 0.25
        }
        compact_id_map: dict[int, int] = {}
        for idx, frame_players in enumerate(tracks["players"]):
            remapped_players = {}
            for tid, pdata in frame_players.items():
                if tid not in compact_id_map:
                    compact_id_map[tid] = len(compact_id_map) + 1
                pdata["role"] = "goalkeeper" if tid in gk_track_ids else "player"
                remapped_players[compact_id_map[tid]] = pdata
            tracks["players"][idx] = remapped_players

        # Tag visual pitch membership using a generously un-eroded pitch hull so stands/crowd false detections
        # are excluded while every player on or near the touchlines is guaranteed visible (`in_visual_pitch = True`)!
        if isinstance(video_source, str) and pitch_model is not None:
            self.tag_visual_pitch_membership(video_source, tracks, pitch_model)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    @staticmethod
    def tag_visual_pitch_membership(
        video_path: str,
        tracks: dict,
        pitch_model: PitchColorModel,
        stride: int = 5,
    ) -> None:
        """Mark `in_visual_pitch` using dilated visual turf hull (never hides valid pitch players!)."""
        props = get_video_properties(video_path)
        fw, fh = props["width"], props["height"]
        cap = cv2.VideoCapture(video_path)
        n_frames = len(tracks.get("players", []))
        cached_mask: np.ndarray | None = None

        try:
            for f in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if f % stride == 0 or cached_mask is None:
                    raw_hull = pitch_model.compute_playable_pitch_mask(frame, work_w=480, erode_border_frac=0.0)
                    # Dilate slightly (~2% of width) so touchline/apron players & linesmen always stay visible
                    kd = max(5, int(round(0.022 * 480)) | 1)
                    cached_mask = cv2.dilate(raw_hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))

                for group in ("players", "referees"):
                    for odata in tracks[group][f].values():
                        fx, fy = get_foot_position(odata["bbox"])
                        odata["in_visual_pitch"] = PitchColorModel.is_point_in_mask(cached_mask, fx, fy, fw, fh)
        finally:
            cap.release()

    def refine_ball_tracks(
        self,
        video_path: str,
        tracks: dict,
        pitch_model: PitchColorModel | None = None,
        scale_ctx: FrameScaleContext | None = None,
        read_from_stub: bool = True,
        ball_stub_path: str = "stubs/ball_candidates_stub.pkl",
    ) -> list[dict]:
        """Run Hybrid SAHI sliced + Camera-Stabilized Kalman/RTS ball tracking."""
        return self.ball_tracker.track_ball(
            video_path=video_path,
            model=self.model,
            tracks=tracks,
            pitch_model=pitch_model,
            scale_ctx=scale_ctx,
            read_from_stub=read_from_stub,
            candidates_stub_path=ball_stub_path,
        )

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int],
        track_id: int | None = None,
        has_ball: bool = False,
        is_referee: bool = False,
    ) -> np.ndarray:
        """Draw resolution-scaled open-top half-ellipse beneath a bounding box with optional ID badge."""
        ui = frame.shape[0] / 1080.0
        y2 = int(round(bbox[3]))
        x_center, _ = get_center_of_bbox(bbox)
        width = max(int(round(6 * ui)), get_bbox_width(bbox))
        thickness = max(1, int(round(2.0 * ui)))

        if has_ball:
            pad_x = max(3, int(round(6 * ui)))
            pad_y = max(2, int(round(4 * ui)))
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width) + pad_x, max(2, int(0.35 * width)) + pad_y),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=(0, 185, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        # If referee marker is very dark (e.g., Black on yellow-kit matches), draw a subtle light outer halo for crispness
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        if is_referee and lum < 55.0:
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width) + 1, max(1, int(0.35 * width)) + 1),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=(235, 235, 235),
                thickness=thickness + 2,
                lineType=cv2.LINE_AA,
            )

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), max(1, int(0.35 * width))),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        if track_id is not None:
            text = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.58 * ui
            font_thickness = max(1, int(round(2.0 * ui)))
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            rect_w = max(int(round(38 * ui)), text_w + int(round(12 * ui)))
            rect_h = max(12, int(round(20 * ui)))
            y_off = int(round(15 * ui))

            x1_rect = int(x_center - rect_w // 2)
            x2_rect = int(x_center + rect_w // 2)
            y1_rect = int((y2 - rect_h // 2) + y_off)
            y2_rect = int((y2 + rect_h // 2) + y_off)

            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), (20, 20, 20), max(1, int(round(ui))))

            text_color = (0, 0, 0) if lum >= 145.0 else (255, 255, 255)
            x_text = int(x_center - text_w / 2)
            y_text = int((y1_rect + y2_rect) / 2 + text_h / 2)
            cv2.putText(
                frame,
                text,
                (x_text, y_text),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_triangle(
        self,
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Draw a resolution-scaled filled downward-pointing pointer triangle above the ball."""
        ui = frame.shape[0] / 1080.0
        y = int(round(bbox[1]))
        x, _ = get_center_of_bbox(bbox)
        dx = max(5, int(round(10 * ui)))
        dy = max(10, int(round(20 * ui)))

        triangle_points = np.array(
            [
                [x, y],
                [x - dx, y - dy],
                [x + dx, y - dy],
            ],
            dtype=np.int32,
        )

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), max(1, int(round(2 * ui))))
        return frame

    def draw_frame_annotations(
        self,
        frame: np.ndarray,
        frame_num: int,
        tracks: dict,
        referee_color: tuple[int, int, int] | None = None,
        in_place: bool = True,
    ) -> np.ndarray:
        """Render broadcast annotations on a single frame using visual pitch visibility (never metric clipping!)."""
        target_frame = frame if in_place else frame.copy()
        ref_col = referee_color if referee_color is not None else self.referee_color

        player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
        referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
        ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}

        # Draw Players & Goalkeepers (uses `in_visual_pitch`, NOT metric `on_pitch`!)
        for track_id, player in player_dict.items():
            if not player.get("in_visual_pitch", True):
                continue
            color = player.get("team_color", self.DEFAULT_PLAYER_COLOR)
            self.draw_ellipse(
                target_frame,
                player["bbox"],
                color,
                track_id,
                has_ball=bool(player.get("has_ball", False)),
                is_referee=False,
            )

        # Draw Referees (contrasting referee color, no ID badge)
        for _, referee in referee_dict.items():
            if not referee.get("in_visual_pitch", True):
                continue
            self.draw_ellipse(target_frame, referee["bbox"], ref_col, is_referee=True)

        # Draw Ball (green pointer triangle)
        for _, ball in ball_dict.items():
            self.draw_triangle(target_frame, ball["bbox"], self.BALL_COLOR)

        return target_frame
