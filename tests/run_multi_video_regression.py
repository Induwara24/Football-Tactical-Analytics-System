import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from homography.pitch_model import build_fifa_template_points
from main import run_pipeline
from team_assigner import TeamAssigner
from trackers import Tracker
from homography import TacticalViewDrawer
from formations import FormationDrawer


def audit_source_constraints() -> None:
    """Verify Zero Hardcoded Visuals (no cv2.inRange) and SAHI + Kalman presence across core modules."""
    checked_files = [
        "utils/calibration.py",
        "trackers/tracker.py",
        "trackers/ball_tracker.py",
        "homography/homography_estimator.py",
        "team_assigner/team_assigner.py",
        "team_assigner/possession_assigner.py",
        "formations/formation_analyzer.py",
    ]
    for rel in checked_files:
        full_p = os.path.join(PROJECT_ROOT, rel)
        with open(full_p, "r", encoding="utf-8") as f:
            src = f.read()
        assert "cv2.inRange(" not in src, f"Constraint 1 Violation: hardcoded cv2.inRange found in {rel}"

    with open(os.path.join(PROJECT_ROOT, "trackers/ball_tracker.py"), "r", encoding="utf-8") as f:
        ball_src = f.read()
    assert "get_slice_bboxes" in ball_src, "Constraint 3 Violation: SAHI slicing missing in ball_tracker.py"
    assert "kalman_rts_smooth_trajectory" in ball_src, "Constraint 3 Violation: Kalman/RTS smoother missing in ball_tracker.py"
    print("[PASS] Static Constraint Audit: 0 hardcoded cv2.inRange calls; SAHI + Kalman/RTS verified.")


def render_summary_sheet(
    video_path: str,
    result: dict,
    out_img_path: str,
) -> None:
    """Export a 2x2 diagnostic grid: 3 annotated tactical frames + 1 FIFA pitch-line wireframe debug frame."""
    tracks = result["tracks"]
    homographies = result["homographies"]
    formations = result["formations"]
    team_meta = result["team_meta"]
    n_f = result["props"]["frame_count"]

    tracker = Tracker("models/best.pt")
    tactical_drawer = TacticalViewDrawer(position="bottom_center")
    formation_drawer = FormationDrawer()
    ref_col = team_meta["referee_color"]

    tmpl = build_fifa_template_points()
    tmpl_hom = np.hstack([tmpl, np.ones((len(tmpl), 1), dtype=np.float64)])

    idxs = [0, n_f // 3, (2 * n_f) // 3]
    mid_idx = n_f // 2
    cap = cv2.VideoCapture(video_path)
    panels = []

    try:
        for f_idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            tracker.draw_frame_annotations(frame, f_idx, tracks, referee_color=ref_col, in_place=True)
            tactical_drawer.draw_frame_radar(frame, f_idx, tracks, homographies=homographies, referee_color=ref_col)
            formation_drawer.draw_formation_hud(frame, f_idx, formations, team_meta["team_colors"])
            panels.append(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))

        # 4th panel: raw broadcast frame + cyan FIFA pitch-line registration wireframe at mid_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, dbg = cap.read()
        if ret and mid_idx < len(homographies):
            H_p2i = homographies[mid_idx]["H_pitch_to_img"]
            conf = homographies[mid_idx].get("confidence", 0.0)
            proj = tmpl_hom @ H_p2i.T
            valid = proj[:, 2] > 1e-3
            px = (proj[valid, 0] / proj[valid, 2]).astype(np.int32)
            py = (proj[valid, 1] / proj[valid, 2]).astype(np.int32)
            for x, y in zip(px, py):
                if 0 <= x < dbg.shape[1] and 0 <= y < dbg.shape[0]:
                    cv2.circle(dbg, (int(x), int(y)), 3, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                dbg,
                f"HOMOGRAPHY LOCK (Frame {mid_idx}, score={conf:.1f})",
                (28, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(cv2.resize(dbg, (960, 540), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()

    if len(panels) == 4:
        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        os.makedirs(os.path.dirname(out_img_path) or ".", exist_ok=True)
        cv2.imwrite(out_img_path, grid)


def evaluate_video(video_path: str, out_dir: str, render_mp4: bool = False) -> dict:
    stem = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")
    t0 = time.perf_counter()
    mp4_out = os.path.join(out_dir, f"{stem}_annotated.mp4") if render_mp4 else None

    res = run_pipeline(
        input_path=video_path,
        output_path=mp4_out,
        read_from_stub=True,
        render_video=render_mp4,
    )
    elapsed = time.perf_counter() - t0

    tracks = res["tracks"]
    homographies = res["homographies"]
    formations = res["formations"]
    team_meta = res["team_meta"]
    scale_ctx = res["scale_ctx"]
    n_f = res["props"]["frame_count"]

    # 1. Visual Player Visibility
    vis_players = [
        sum(1 for p in tracks["players"][f].values() if p.get("in_visual_pitch", True))
        for f in range(n_f)
    ]
    min_vis, med_vis = int(min(vis_players)), float(np.median(vis_players))

    # 2. Ball Trajectory Coverage & Relative Smoothness
    ball_cov = sum(1 for f in range(n_f) if 1 in tracks["ball"][f]) / float(max(1, n_f))
    ball_centers = np.array(
        [
            [
                0.5 * (tracks["ball"][f][1]["bbox"][0] + tracks["ball"][f][1]["bbox"][2]),
                0.5 * (tracks["ball"][f][1]["bbox"][1] + tracks["ball"][f][1]["bbox"][3]),
            ]
            for f in range(n_f)
            if 1 in tracks["ball"][f]
        ],
        dtype=np.float64,
    )
    ball_steps_hp = (
        np.hypot(np.diff(ball_centers[:, 0]), np.diff(ball_centers[:, 1])) / max(1.0, scale_ctx.median_player_h)
        if len(ball_centers) > 1
        else np.zeros(1)
    )

    # 3. Homography Invertibility & Registration Score
    hom_scores = [float(h.get("confidence", 0.0)) for h in homographies]
    for i, h in enumerate(homographies):
        prod = h["H_pitch_to_img"] @ h["H_img_to_pitch"]
        prod /= prod[2, 2]
        assert np.max(np.abs(prod - np.eye(3))) < 1e-4, f"Non-invertible H at {stem} frame {i}"

    # 4. Team & Referee Perceptual Contrast (CIE Lab Delta-E)
    c1, c2 = team_meta["team_colors"][1], team_meta["team_colors"][2]
    cref = team_meta["referee_color"]
    lab1 = TeamAssigner._bgr_to_cie_lab(c1)
    lab2 = TeamAssigner._bgr_to_cie_lab(c2)
    labr = TeamAssigner._bgr_to_cie_lab(cref)
    de_teams = float(np.linalg.norm(lab1 - lab2))
    de_ref = float(min(np.linalg.norm(labr - lab1), np.linalg.norm(labr - lab2)))

    # 5. Dual Formation Validity
    for f, st in enumerate(formations):
        for t_id in (1, 2):
            fmt = st["teams"][t_id]["formation"]
            assert sum(int(x) for x in fmt.split("-")) == 10, f"Invalid 10-man formation {fmt} in {stem}"

    summary_img = os.path.join(out_dir, f"{stem}_summary.jpg")
    render_summary_sheet(video_path, res, summary_img)

    metrics = {
        "video": os.path.basename(video_path),
        "frames": n_f,
        "fps": res["props"]["fps"],
        "med_player_h_px": round(scale_ctx.median_player_h, 1),
        "min_vis_players": min_vis,
        "med_vis_players": round(med_vis, 1),
        "ball_coverage": round(100.0 * ball_cov, 1),
        "p95_ball_step_hp": round(float(np.percentile(ball_steps_hp, 95)), 3),
        "med_homog_score": round(float(np.median(hom_scores)), 1),
        "team1_bgr": c1,
        "team2_bgr": c2,
        "ref_bgr": cref,
        "de_teams": round(de_teams, 1),
        "de_ref_min": round(de_ref, 1),
        "elapsed_s": round(elapsed, 1),
        "summary_img": summary_img,
    }

    assert min_vis >= 8 and med_vis >= 14.0, f"Low player visibility on {stem}: {metrics}"
    assert ball_cov >= 0.99, f"Incomplete ball coverage on {stem}: {metrics}"
    assert de_teams >= 22.0, f"Poor team color separation on {stem}: {metrics}"
    assert de_ref >= 30.0, f"Poor referee contrast on {stem}: {metrics}"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Video Regression Validation Suite")
    parser.add_argument("--video", default=None, help="Optional single video path to validate")
    parser.add_argument("--include-inputs", action="store_true", default=True, help="Also test A1606b0e6_0 & C35bd9041_0")
    parser.add_argument("--render-mp4", action="store_true", help="Also render full MP4s into output_videos/regression/")
    args = parser.parse_args()

    audit_source_constraints()

    if args.video:
        videos = [os.path.abspath(args.video)]
    else:
        videos = sorted(glob.glob(os.path.join(PROJECT_ROOT, "test_videos", "*.mp4")))
        if args.include_inputs:
            for extra in ("input_videos/A1606b0e6_0.mp4", "input_videos/C35bd9041_0.mp4"):
                full_e = os.path.join(PROJECT_ROOT, extra)
                if os.path.exists(full_e):
                    videos.append(full_e)

    out_dir = os.path.join(PROJECT_ROOT, "output_videos", "regression")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRunning multi-video regression across {len(videos)} videos...")
    all_metrics = []
    for vpath in videos:
        print(f"\n--- Validating {os.path.basename(vpath)} ---")
        m = evaluate_video(vpath, out_dir, render_mp4=args.render_mp4)
        all_metrics.append(m)
        print("  Result:", m)

    print("\n================ ALL MULTI-VIDEO REGRESSION CHECKS PASSED ================")
    for m in all_metrics:
        print(
            f"  {m['video']:20s} | {m['frames']:3d}f @ {m['fps']:4.1f}fps | H_p={m['med_player_h_px']:4.1f}px | "
            f"vis_p={m['med_vis_players']:4.1f} (min {m['min_vis_players']:2d}) | ball={m['ball_coverage']:5.1f}% | "
            f"H_score={m['med_homog_score']:5.1f} | dE_teams={m['de_teams']:4.1f}, dE_ref={m['de_ref_min']:4.1f}"
        )


if __name__ == "__main__":
    main()
