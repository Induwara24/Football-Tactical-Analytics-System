# Football-Tactical-Analytics-System

Monocular broadcast football computer-vision pipeline for tactical analytics and parallax-corrected offside detection, built for resource-constrained environments (single GPU / local development).

## Overview

The pipeline processes broadcast football footage (`1920x1080`, `25 FPS`) in `main.py` across modular stages:

### Phase 1: Dynamic Field Registration & Formation Clustering
- **M1 — Video I/O & Multi-Object Tracking (`utils/`, `trackers/`)**:
  - Streaming video read/write utilities (`utils/video_utils.py`, `utils/bbox_utils.py`).
  - Fine-tuned YOLO detection (`models/best.pt` for `ball`, `goalkeeper`, `player`, `referee`) and persistent multi-object tracking (`trackers/tracker.py`).
  - 4-stage ball tracking refinement and interpolation (`trackers/ball_tracker.py`).
- **M2 — Dynamic Homography (`homography/`)**:
  - Pitch masking, line/keypoint extraction, and per-frame 3x3 homography estimation (`homography/homography_estimator.py`, `homography/pitch_model.py`).
  - 2D tactical pitch radar overlay (`homography/tactical_view_drawer.py`).
- **M3 — Team Affiliation, Possession & Formations (`team_assigner/`, `formations/`)**:
  - K-Means clustering on player torso HSV colour features for team assignment (`team_assigner/team_assigner.py`).
  - Ball possession assignment (`team_assigner/possession_assigner.py`).
  - Spatial clustering on smoothed 2D coordinates to classify attacking and defensive formations and render the tactical HUD (`formations/formation_analyzer.py`, `formations/formation_drawer.py`).

### Phase 2: Parallax-Corrected Offside Engine *(Planned)*
- **M5 — Skeletal Pose**: YOLO-Pose (17 keypoints) for furthest-forward playable body parts.
- **M6 — Parallax Correction**: Vertical vanishing point estimation and ground-plane projection prior to homography.
- **M7 — Pass Spotting**: 1D-CNN on ball trajectory vectors to pinpoint pass initiation frames.
- **M8 — Offside Inference & AR Overlay**: Perspective-correct AR offside line rendering via OpenCV.

## Project Structure

```text
├── formations/        # Formation analysis and HUD drawing (M3)
├── homography/        # Dynamic field registration, pitch model, 2D radar (M2)
├── input_videos/      # Raw broadcast input clips (gitignored)
├── models/            # Trained YOLO weights, e.g. best.pt (gitignored)
├── output_videos/     # Rendered videos and debug frames (gitignored)
├── stubs/             # Cached .pkl inference/tracking stubs (gitignored)
├── team_assigner/     # K-Means team assignment & ball possession (M3)
├── trackers/          # YOLO + ByteTrack/BoT-SORT tracking & ball refinement (M1)
├── training/          # YOLO training notebook and dataset config
├── utils/             # Video I/O and bounding-box helpers
├── main.py            # End-to-end pipeline orchestrator
└── requirements.txt   # Python dependencies
```

## Setup & Usage

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add model weights & input video**:
   - Place your trained YOLO weights at `models/best.pt`.
   - Place your broadcast clip at `input_videos/A1606b0e6_0.mp4` (or update `input_path` in `main.py`).

4. **Run the pipeline**:
   ```bash
   python main.py
   ```
   Cached `.pkl` stubs in `stubs/` allow re-running homography, team assignment, formation clustering, and rendering without repeating full YOLO inference.
