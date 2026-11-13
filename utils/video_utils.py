import os
from collections.abc import Callable, Iterator

import cv2
import numpy as np


def get_video_properties(video_path: str) -> dict:
    """Return basic metadata (fps, width, height, frame_count) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")
    props = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return props


def iter_video_batches(video_path: str, batch_size: int = 20) -> Iterator[list[np.ndarray]]:
    """Stream a video in fixed-size frame batches to keep 1080p memory usage low."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    batch: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        cap.release()


def read_video(video_path: str, max_frames: int | None = None) -> list[np.ndarray]:
    """Read frames from a video file into a list of BGR numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")
    frames = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def save_video(output_video_frames: list[np.ndarray], output_video_path: str, fps: float = 25.0) -> None:
    """Write an in-memory sequence of BGR frames to disk as an MP4/AVI video."""
    if not output_video_frames:
        raise ValueError("output_video_frames is empty; nothing to save.")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    height, width = output_video_frames[0].shape[:2]
    ext = os.path.splitext(output_video_path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if ext == ".avi" else "mp4v"))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    try:
        for frame in output_video_frames:
            out.write(frame)
    finally:
        out.release()


def stream_annotated_video(
    input_video_path: str,
    output_video_path: str,
    annotate_frame_fn: Callable[[np.ndarray, int], np.ndarray],
    fps: float | None = None,
    sample_frame_paths: dict[int, str] | None = None,
) -> int:
    """Stream frames from input_video_path through annotate_frame_fn directly to output_video_path."""
    props = get_video_properties(input_video_path)
    target_fps = fps if fps is not None else props["fps"]
    width, height = props["width"], props["height"]

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_video_path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if ext == ".avi" else "mp4v"))

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {input_video_path}")

    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = annotate_frame_fn(frame, frame_idx)
            out.write(annotated)
            if sample_frame_paths and frame_idx in sample_frame_paths:
                sample_path = sample_frame_paths[frame_idx]
                os.makedirs(os.path.dirname(sample_path) or ".", exist_ok=True)
                cv2.imwrite(sample_path, annotated)
            frame_idx += 1
    finally:
        cap.release()
        out.release()

    return frame_idx
