import argparse
import os
import pickle
from typing import Any

from formations import FormationAnalyzer, FormationDrawer
from homography import HomographyEstimator, TacticalViewDrawer
from team_assigner import PossessionAssigner, TeamAssigner
from trackers import Tracker
from utils import (
    FrameScaleContext,
    PitchColorModel,
    get_video_properties,
    get_video_stub_dir,
    stream_annotated_video,
)


def run_pipeline(
    input_path: str,
    output_path: str | None = None,
    model_path: str = "models/best.pt",
    stub_dir: str | None = None,
    read_from_stub: bool = True,
    sample_frame_paths: dict[int, str] | None = None,
    render_video: bool = True,
) -> dict[str, Any]:
    """
    Execute the generalized M1+M2+M3 football tactical analytics pipeline on any input video.
    Automatically namespaces stubs per video signature so caches never collide across clips.
    """
    props = get_video_properties(input_path)
    resolved_stub_dir = stub_dir or get_video_stub_dir(input_path)

    pitch_stub = os.path.join(resolved_stub_dir, "pitch_model.pkl")
    track_stub = os.path.join(resolved_stub_dir, "tracks.pkl")
    ball_stub = os.path.join(resolved_stub_dir, "ball_candidates.pkl")
    hom_stub = os.path.join(resolved_stub_dir, "homography.pkl")
    team_stub = os.path.join(resolved_stub_dir, "team_assigner.pkl")
    fmt_stub = os.path.join(resolved_stub_dir, "formations.pkl")

    # 0. Dynamic Video Calibration: Unsupervised CIE Lab Pitch Turf Model
    pitch_model = PitchColorModel.fit_from_video(
        input_path,
        read_from_stub=read_from_stub,
        stub_path=pitch_stub,
    )

    # 1. Milestone M1: Scale-Adaptive Person Tracking + Hybrid SAHI & Kalman/RTS Ball Tracking
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(
        input_path,
        pitch_model=pitch_model,
        read_from_stub=read_from_stub,
        stub_path=track_stub,
    )
    # Ensure `in_visual_pitch` is populated even when loading older track stubs
    if tracks["players"] and any("in_visual_pitch" not in p for p in tracks["players"][0].values()):
        tracker.tag_visual_pitch_membership(input_path, tracks, pitch_model)

    scale_ctx = FrameScaleContext(props["width"], props["height"], props["fps"]).calibrate_from_tracks(tracks)

    tracks["ball"] = tracker.refine_ball_tracks(
        video_path=input_path,
        tracks=tracks,
        pitch_model=pitch_model,
        scale_ctx=scale_ctx,
        read_from_stub=read_from_stub,
        ball_stub_path=ball_stub,
    )

    step_homographies = None
    if os.path.exists(ball_stub):
        with open(ball_stub, "rb") as f:
            step_homographies = pickle.load(f).get("step_homographies")

    # 2. Milestone M2: Camera-Agnostic Dynamic Field Registration & 2D Pitch Projection
    homography_estimator = HomographyEstimator()
    homographies = homography_estimator.estimate_video_homographies(
        video_path=input_path,
        tracks=tracks,
        pitch_model=pitch_model,
        scale_ctx=scale_ctx,
        step_homographies=step_homographies,
        read_from_stub=read_from_stub,
        stub_path=hom_stub,
    )
    homography_estimator.add_transformed_positions_to_tracks(tracks, homographies, scale_ctx=scale_ctx)

    # 3. Milestone M3: Dynamic Jersey K-Means, Relative Possession & Dual Formations
    team_assigner = TeamAssigner()
    team_meta = team_assigner.assign_teams(
        video_path=input_path,
        tracks=tracks,
        pitch_model=pitch_model,
        scale_ctx=scale_ctx,
        read_from_stub=read_from_stub,
        stub_path=team_stub,
    )
    tracker.referee_color = team_meta["referee_color"]

    possession_assigner = PossessionAssigner()
    possession_states = possession_assigner.assign_ball_possession(tracks, scale_ctx=scale_ctx)

    formation_analyzer = FormationAnalyzer()
    formations = formation_analyzer.analyze_formations(
        tracks=tracks,
        homographies=homographies,
        possession_states=possession_states,
        team_defends_left=team_meta["team_defends_left"],
        scale_ctx=scale_ctx,
        read_from_stub=read_from_stub,
        stub_path=fmt_stub,
    )

    rendered_count = 0
    if render_video and output_path:
        tactical_drawer = TacticalViewDrawer(position="bottom_center")
        formation_drawer = FormationDrawer()
        ref_col = team_meta["referee_color"]

        def annotate_frame(frame, frame_idx: int):
            tracker.draw_frame_annotations(frame, frame_idx, tracks, referee_color=ref_col, in_place=True)
            tactical_drawer.draw_frame_radar(frame, frame_idx, tracks, homographies=homographies, referee_color=ref_col)
            formation_drawer.draw_formation_hud(frame, frame_idx, formations, team_meta["team_colors"])
            return frame

        rendered_count = stream_annotated_video(
            input_video_path=input_path,
            output_video_path=output_path,
            annotate_frame_fn=annotate_frame,
            fps=props["fps"],
            sample_frame_paths=sample_frame_paths,
        )
        print(f"Saved annotated video ({rendered_count} frames @ {props['fps']:.1f} FPS) to {output_path}")

    return {
        "props": props,
        "stub_dir": resolved_stub_dir,
        "pitch_model": pitch_model,
        "scale_ctx": scale_ctx,
        "tracks": tracks,
        "homographies": homographies,
        "team_meta": team_meta,
        "possession_states": possession_states,
        "formations": formations,
        "rendered_count": rendered_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized Football Tactical Analytics Pipeline")
    parser.add_argument("--input", default="input_videos/A1606b0e6_0.mp4", help="Path to input broadcast video")
    parser.add_argument("--output", default=None, help="Path to output annotated MP4 video")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO checkpoint")
    parser.add_argument("--stub-dir", default=None, help="Optional explicit stub directory")
    parser.add_argument("--no-stub", action="store_true", help="Recompute all stages ignoring cached stubs")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or (
        "output_videos/output_video.mp4"
        if stem == "A1606b0e6_0"
        else f"output_videos/{stem}_annotated.mp4"
    )

    props = get_video_properties(args.input)
    n_f = props["frame_count"]
    sample_frames = {
        0: f"output_videos/{stem}_sample_0.jpg",
        n_f // 3: f"output_videos/{stem}_sample_{n_f // 3}.jpg",
        (2 * n_f) // 3: f"output_videos/{stem}_sample_{(2 * n_f) // 3}.jpg",
        max(0, n_f - 25): f"output_videos/{stem}_sample_{max(0, n_f - 25)}.jpg",
    }

    run_pipeline(
        input_path=args.input,
        output_path=output_path,
        model_path=args.model,
        stub_dir=args.stub_dir,
        read_from_stub=not args.no_stub,
        sample_frame_paths=sample_frames,
        render_video=True,
    )


if __name__ == "__main__":
    main()
