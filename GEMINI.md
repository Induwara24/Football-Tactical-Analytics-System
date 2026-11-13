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
- **Cache Heavy Inference**: Always support `read_from_stub` / `stub_path` caching (`.pkl` in `stubs/`) for YOLO detections, object tracks, and pose keypoints so OpenCV/K-Means/visualization tuning runs in seconds without re-running model inference across full videos.
- **Resource Efficiency**: Batch frame inference (e.g., batch size 16–20), prefer vectorized NumPy/OpenCV operations over Python pixel loops, and avoid holding unnecessary copies of full 1080p frame arrays in memory.
- **Visual Verification**: When adding or tuning CV stages (tracking boxes, HSV crops, Canny lines, 2D radar, AR lines), save sample annotated frames and inspect them visually to confirm geometric and colour accuracy.

