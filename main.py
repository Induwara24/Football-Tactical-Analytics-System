import os
import pickle

from formations import FormationAnalyzer, FormationDrawer
from homography import HomographyEstimator, TacticalViewDrawer
from team_assigner import PossessionAssigner, TeamAssigner
from trackers import Tracker
from utils import get_video_properties, stream_annotated_video


def main() -> None:
    input_path = "input_videos/A1606b0e6_0.mp4"
    output_path = "output_videos/output_video.mp4"
    track_stub_path = "stubs/track_stubs.pkl"
    ball_stub_path = "stubs/ball_candidates_stub.pkl"
    homography_stub_path = "stubs/homography_stub.pkl"
    team_stub_path = "stubs/team_assigner_stub.pkl"
    formations_stub_path = "stubs/formations_stub.pkl"

    props = get_video_properties(input_path)

    # Milestone M1: Multi-Object Tracking + 4-Stage Accurate Ball Tracking
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        input_path,
        read_from_stub=True,
        stub_path=track_stub_path,
    )
    tracks["ball"] = tracker.refine_ball_tracks(
        video_path=input_path,
        tracks=tracks,
        read_from_stub=True,
        ball_stub_path=ball_stub_path,
    )

    step_homographies = None
    if os.path.exists(ball_stub_path):
        with open(ball_stub_path, "rb") as f:
            step_homographies = pickle.load(f).get("step_homographies")

    # Milestone M2: Dynamic Field Registration & 2D Tactical Pitch Projection
    homography_estimator = HomographyEstimator()
    homographies = homography_estimator.estimate_video_homographies(
        video_path=input_path,
        tracks=tracks,
        step_homographies=step_homographies,
        read_from_stub=True,
        stub_path=homography_stub_path,
    )
    homography_estimator.add_transformed_positions_to_tracks(tracks, homographies)

    # Milestone M3: Team Affiliation K-Means, Possession State & Dual Attacking/Defensive Formations
    team_assigner = TeamAssigner()
    team_meta = team_assigner.assign_teams(
        video_path=input_path,
        tracks=tracks,
        read_from_stub=True,
        stub_path=team_stub_path,
    )

    possession_assigner = PossessionAssigner()
    possession_states = possession_assigner.assign_ball_possession(tracks)

    formation_analyzer = FormationAnalyzer()
    formations = formation_analyzer.analyze_formations(
        tracks=tracks,
        homographies=homographies,
        possession_states=possession_states,
        team_defends_left=team_meta["team_defends_left"],
        read_from_stub=True,
        stub_path=formations_stub_path,
    )

    tactical_drawer = TacticalViewDrawer(position="bottom_center")
    formation_drawer = FormationDrawer()

    sample_frames = {
        0: "output_videos/m3_sample_frame_0.jpg",
        200: "output_videos/m3_sample_frame_200.jpg",
        450: "output_videos/m3_sample_frame_450.jpg",
        700: "output_videos/m3_sample_frame_700.jpg",
    }

    def annotate_frame(frame, frame_idx: int):
        tracker.draw_frame_annotations(frame, frame_idx, tracks, in_place=True)
        tactical_drawer.draw_frame_radar(frame, frame_idx, tracks, homographies=homographies)
        formation_drawer.draw_formation_hud(frame, frame_idx, formations, team_meta["team_colors"])
        return frame

    rendered_count = stream_annotated_video(
        input_video_path=input_path,
        output_video_path=output_path,
        annotate_frame_fn=annotate_frame,
        fps=props["fps"],
        sample_frame_paths=sample_frames,
    )
    print(f"Saved M1+M2+M3 annotated video ({rendered_count} frames @ {props['fps']} FPS) to {output_path}")


if __name__ == "__main__":
    main()
