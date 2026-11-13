import os
import pickle
from collections import defaultdict
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from utils import get_bbox_width, get_center_of_bbox, iter_video_batches, stream_annotated_video
from .ball_tracker import BallTracker


DEFAULT_BYTETRACK_ARGS = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


class Tracker:
    """Batched YOLO detector + ByteTrack multi-object tracker with broadcast annotation helpers."""

    DEFAULT_PLAYER_COLOR = (0, 0, 255)        # Red (BGR) until M3 TeamAssigner assigns kit colors
    DEFAULT_GOALKEEPER_COLOR = (255, 191, 0)  # Sky Blue (BGR) distinct M1 goalkeeper color
    REFEREE_COLOR = (0, 255, 255)             # Yellow (BGR)
    BALL_COLOR = (0, 255, 0)                  # Green (BGR)

    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.tracker = BYTETracker(DEFAULT_BYTETRACK_ARGS)
        self.ball_tracker = BallTracker()

    def detect_frames(self, frames: list[np.ndarray], batch_size: int = 20) -> list:
        """Run batched YOLO detection across an in-memory list of video frames."""
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.1, verbose=False)
            detections.extend(detections_batch)
        return detections

    def _iter_detection_batches(
        self,
        video_source: str | list[np.ndarray],
        batch_size: int = 20,
    ):
        """Yield YOLO Results batches from either a video file path (streaming) or a frame list."""
        if isinstance(video_source, str):
            for frame_batch in iter_video_batches(video_source, batch_size=batch_size):
                yield self.model.predict(frame_batch, conf=0.1, verbose=False)
        else:
            for i in range(0, len(video_source), batch_size):
                yield self.model.predict(video_source[i : i + batch_size], conf=0.1, verbose=False)

    def get_object_tracks(
        self,
        video_source: str | list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None,
        batch_size: int = 20,
    ) -> dict:
        """Return frame-indexed player/goalkeeper, referee, and ball tracks, using stub cache if available."""
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        self.tracker.reset()

        tracks: dict[str, list[dict]] = {
            "players": [],
            "referees": [],
            "ball": [],
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

                boxes_np = detection.boxes.cpu().numpy().data.copy()
                if len(boxes_np) > 0:
                    orig_cls = boxes_np[:, 5].astype(int).copy()

                    # Keep highest-confidence ball detection per frame
                    ball_mask = orig_cls == ball_cls_id
                    if np.any(ball_mask):
                        ball_candidates = boxes_np[ball_mask]
                        best_idx = int(np.argmax(ball_candidates[:, 4]))
                        tracks["ball"][frame_num][1] = {"bbox": ball_candidates[best_idx, :4].tolist()}

                    # Unify goalkeeper -> player and exclude ball prior to ByteTrack so IDs stay clean
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
                            }
                        elif cls_id == referee_cls_id:
                            tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                else:
                    self.tracker.update(Boxes(np.empty((0, 6), dtype=np.float32), detection.orig_shape), None)

                frame_num += 1
                if frame_num % 100 == 0:
                    print(f"Processed {frame_num} frames for detection & tracking...")

        # Stabilize goalkeeper role and assign compact sequential track IDs (1..N) in order of appearance
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

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def refine_ball_tracks(
        self,
        video_path: str,
        tracks: dict,
        read_from_stub: bool = True,
        ball_stub_path: str = "stubs/ball_candidates_stub.pkl",
    ) -> list[dict]:
        """Run 4-stage pitch-masked, static-suppressed, Viterbi-linked, ROI-zoomed ball tracking."""
        return self.ball_tracker.track_ball(
            video_path=video_path,
            model=self.model,
            tracks=tracks,
            read_from_stub=read_from_stub,
            candidates_stub_path=ball_stub_path,
        )

    def interpolate_ball_positions(self, ball_positions: list[dict]) -> list[dict]:
        """Linearly interpolate missing ball bounding boxes across frames using vectorized NumPy."""
        num_frames = len(ball_positions)
        if num_frames == 0:
            return ball_positions

        valid_indices = []
        valid_bboxes = []
        for idx, frame_ball in enumerate(ball_positions):
            bbox = frame_ball.get(1, {}).get("bbox")
            if bbox is not None and len(bbox) == 4:
                valid_indices.append(idx)
                valid_bboxes.append(bbox)

        if not valid_indices:
            return ball_positions

        valid_indices_arr = np.asarray(valid_indices, dtype=np.float64)
        valid_bboxes_arr = np.asarray(valid_bboxes, dtype=np.float64)
        all_indices = np.arange(num_frames, dtype=np.float64)

        interp_bboxes = np.column_stack(
            [np.interp(all_indices, valid_indices_arr, valid_bboxes_arr[:, col]) for col in range(4)]
        )

        return [{1: {"bbox": row.tolist()}} for row in interp_bboxes]

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int],
        track_id: int | None = None,
        has_ball: bool = False,
    ) -> np.ndarray:
        """Draw an open-top half-ellipse beneath a bounding box with an optional track-ID rectangle badge."""
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = max(1, get_bbox_width(bbox))

        if has_ball:
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width) + 6, max(2, int(0.35 * width)) + 4),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=(0, 185, 255),
                thickness=2,
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
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        if track_id is not None:
            text = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            rectangle_width = max(40, text_w + 12)
            rectangle_height = 20
            x1_rect = int(x_center - rectangle_width // 2)
            x2_rect = int(x_center + rectangle_width // 2)
            y1_rect = int((y2 - rectangle_height // 2) + 15)
            y2_rect = int((y2 + rectangle_height // 2) + 15)

            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED,
            )
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                (20, 20, 20),
                1,
            )

            # High-contrast text color (white on dark crimson badges, black on bright white/sky-blue badges)
            luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_color = (0, 0, 0) if luminance >= 145.0 else (255, 255, 255)

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
        """Draw a filled downward-pointing pointer triangle directly above an object's bounding box."""
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ],
            dtype=np.int32,
        )

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_frame_annotations(
        self,
        frame: np.ndarray,
        frame_num: int,
        tracks: dict,
        in_place: bool = True,
    ) -> np.ndarray:
        """Render broadcast annotations on a single frame for players, goalkeepers, referees, and the ball."""
        target_frame = frame if in_place else frame.copy()

        player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
        referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
        ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}

        # Draw Players & Goalkeepers (half-ellipse + track_id square badge)
        for track_id, player in player_dict.items():
            if not player.get("on_pitch", True):
                continue
            default_color = (
                self.DEFAULT_GOALKEEPER_COLOR
                if player.get("role") == "goalkeeper"
                else self.DEFAULT_PLAYER_COLOR
            )
            color = player.get("team_color", default_color)
            self.draw_ellipse(
                target_frame,
                player["bbox"],
                color,
                track_id,
                has_ball=bool(player.get("has_ball", False)),
            )

        # Draw Referees (yellow half-ellipse only, no ID badge)
        for _, referee in referee_dict.items():
            self.draw_ellipse(target_frame, referee["bbox"], self.REFEREE_COLOR)

        # Draw Ball (green pointer triangle)
        for _, ball in ball_dict.items():
            self.draw_triangle(target_frame, ball["bbox"], self.BALL_COLOR)

        return target_frame

    def draw_annotations(
        self,
        video_frames: list[np.ndarray],
        tracks: dict,
        in_place: bool = True,
    ) -> list[np.ndarray]:
        """Render broadcast annotations across an in-memory sequence of video frames."""
        output_video_frames = video_frames if in_place else []
        for frame_num, frame in enumerate(video_frames):
            annotated = self.draw_frame_annotations(frame, frame_num, tracks, in_place=in_place)
            if not in_place:
                output_video_frames.append(annotated)
        return output_video_frames

    def render_video(
        self,
        input_video_path: str,
        output_video_path: str,
        tracks: dict,
        fps: float | None = None,
        sample_frame_paths: dict[int, str] | None = None,
    ) -> int:
        """Stream input video frame-by-frame, draw broadcast annotations, and write output video."""
        return stream_annotated_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            annotate_frame_fn=lambda frame, idx: self.draw_frame_annotations(frame, idx, tracks, in_place=True),
            fps=fps,
            sample_frame_paths=sample_frame_paths,
        )
