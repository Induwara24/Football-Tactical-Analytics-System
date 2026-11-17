# Football Tactical Analytics & Parallax-Corrected Offside Detection

Monocular broadcast football computer-vision pipeline built for resource-constrained environments (single GPU / local dev). Reference `Project Proposal_ Resource-Efficient Football Tactical Analytics and Parallax-Corrected Offside Detection.pdf` for full architectural context.

## Environment & Execution
- **Python / Pip**: Always run commands using the project virtual environment:
  - `.\.venv\Scripts\python.exe`
  - `.\.venv\Scripts\pip.exe`
- **Assets & Checkpoints**:
  - Fine-tuned detector: `models/best.pt` — classes `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}` (`models/last.pt` and `yolov8x.pt` also present).
  - Sample broadcast clip: `input_videos/A1606b0e6_0.mp4` (1920x1080, 25 FPS, 750 frames).
  - Outputs & caches: write rendered videos/debug frames to `output_videos/` and cached inference/tracking stubs to `stubs/`. Do not overwrite raw files in `input_videos/` or trained weights in `models/`.

## Architecture & Milestones
Keep `main.py` as a clean pipeline orchestrator and isolate logic into focused modules (`utils/`, `trackers/`, `homography/`, `team_assigner/`, `formations/`, `offside/`):

### Phase 1: Dynamic Field Registration & Formation Clustering
- **M1 — Video I/O & Tracking**: Video read/write helpers (`utils/video_utils.py`), YOLO detection (`models/best.pt`), and persistent multi-object tracking (ByteTrack / BoT-SORT), treating goalkeepers consistently with players where needed and interpolating missing ball detections.
- **M2 — Dynamic Homography**: Pitch masking (HSV), line/keypoint extraction (Canny + Hough), and per-frame 3x3 homography (`cv2.getPerspectiveTransform` / `findHomography`) projecting player foot (`bottom-center`) coordinates onto a 2D tactical pitch.
- **M3 — Team Affiliation & Formations**: K-Means on player torso HSV colour features for team assignment, followed by spatial K-Means on smoothed defending-team 2D coordinates to classify defensive lines/formations (e.g., 4-4-2, 4-3-3, 4-5-1).
- **M4 — Phase 1 Validation**: End-to-end formation tracking accuracy review on broadcast clips.

### Phase 2: Parallax-Corrected Offside Engine
- **M5 — Skeletal Pose**: YOLO-Pose (17 keypoints) to extract furthest-forward playable body parts (head, shoulder, knee, foot).
- **M6 — Parallax Correction**: Estimate vertical vanishing point, project elevated keypoints vertically down to the grass image plane, and apply the Phase 1 homography strictly to the ground-projected pixel.
- **M7 — Pass Spotting**: Lightweight 1D-CNN on ball trajectory vectors (SoccerNet) to pinpoint exact pass initiation frames.
- **M8 — Offside Inference & AR Overlay**: Compare parallax-corrected attacker vs. second-last defender coordinates at the pass frame and render the perspective-correct AR offside line via OpenCV.

## Coding & Workflow Guidelines
- **Zero Hardcoded Visuals**: Never hardcode RGB/HSV color ranges for pitch masking, grass removal, goalkeeper kits, or team jerseys. Dynamically learn the pitch turf distribution and cluster team/goalkeeper/referee visuals per video.
- **Relative Geometry Only**: Never use fixed pixel constants for player filtering, ball bounds, tracking gates, morphology kernels, or UI overlays. Scale all spatial thresholds dynamically from frame dimensions (`width`, `height`, `diag`), video `fps`, and the local/median detected player bounding box height (`H_p(y)`).
- **Small-Object Ball Strategy**: Use pitch-bounded tiled/sliced inference (`SAHI`) combined with camera-motion-stabilized Kalman filtering and RTS trajectory smoothing so small or motion-blurred balls are tracked reliably in wide shots.
- **Multi-Video Regression Testing**: Validate pipeline changes across all sample clips in `test_videos/` (plus `input_videos/A1606b0e6_0.mp4`) via `tests/run_multi_video_regression.py` to guarantee global generalization rather than single-clip overfitting.
- **Cache Heavy Inference**: Always support per-video namespaced `read_from_stub` / `stub_path` caching (`.pkl` in `stubs/<video_stem>/`) for YOLO detections, object tracks, homographies, and pose keypoints so downstream tuning runs in seconds without cross-video cache collisions.
- **Resource Efficiency**: Batch frame/tile inference, prefer vectorized NumPy/OpenCV operations over Python pixel loops, and avoid holding unnecessary copies of full 1080p frame arrays in memory.
- **Visual Verification**: When adding or tuning CV stages (tracking boxes, pitch masks, Canny lines, 2D radar, formations, AR lines), save sample annotated frames across diverse clips and inspect them visually to confirm geometric and colour accuracy.

