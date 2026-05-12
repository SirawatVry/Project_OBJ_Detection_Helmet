# Helmet Detection Project

A machine learning project for real-time helmet violation detection using YOLOv8 with object tracking, false positive reduction pipeline, and a Streamlit monitoring dashboard.

## Project Overview

This project detects helmet violations on motorcycles in video footage and logs them automatically. It identifies:

- **Helmet** — Riders wearing helmets
- **Motorcycle** — Motorcycles in the frame
- **No Helmet** — Riders not wearing helmets (violation)

Key capabilities:

- YOLOv8/v9/v10 fine-tuned model inference
- Multi-object tracking via **Norfair** (Euclidean distance tracker)
- Temporal voting & stability scoring for false positive reduction
- Automatic violation image capture and CSV logging
- Real-time **Streamlit** dashboard with gallery and analytics

---

## Directory Structure

```
Project_ML/
├── basemodel/                          # Base model weights and plots
├── dataset/                            # Original training dataset
│   ├── train/                          # Training images and labels
│   └── valid/                          # Validation images and labels
├── dataset_Finetune/                   # Fine-tuning dataset
├── finetundata/                        # Additional fine-tuning data
├── testaugroboflow/                    # Training and augmentation scripts
│   ├── train_helmet_detection.py       # Main training script
│   ├── fix_annotations.py              # Label annotation fixes
│   └── dataset/                        # Roboflow-augmented dataset
├── validation_results/                 # Validation metrics and results
├── error_visualization/                # Error analysis visualizations
│   ├── errors/                         # General errors
│   ├── false_positives/                # False positive cases
│   ├── false_negatives/                # False negative cases
│   └── true_positives/                 # True positive examples
├── violations/
│   ├── no_helmet_captures/             # Cropped violation images (.jpg)
│   └── no_helmet_log.csv               # Violation log (timestamp, track ID, confidence, etc.)
├── media_pipeline.py                   # FalsePositiveReducer + VideoAnalyzer
├── app.py                              # Streamlit dashboard
├── runs/                               # Training run outputs
└── norfair_env/                        # Python virtual environment
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (recommended for GPU acceleration)

### Setup

```bash
# 1. Navigate to project directory
cd Project_ML

# 2. Create and activate virtual environment
python -m venv norfair_env
.\norfair_env\Scripts\activate   # Windows
# source norfair_env/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install ultralytics torch torchvision opencv-python \
            pyyaml pandas numpy matplotlib streamlit norfair
```

### Dataset Format (YOLO)

```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

`data.yaml`:
```yaml
train: train/images
val: valid/images
nc: 3
names: [helmet, motorcycle, no_helmet]
```

---

## Usage

### Running the Dashboard

```bash
streamlit run app.py
```

Upload a video in the **Video Analysis** tab and click **▶️ Run Detection**.

### Performance Settings (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Frame Skip Interval | 1 | Process every N frames (2–3 recommended for real-time) |
| Video Scale Factor | 1.0 | Downscale input (0.5–0.75 for faster processing) |
| UI Update Interval | 1 | Refresh display every N processed frames |
| Enable Preprocessing | OFF | Denoise + contrast enhancement (~25ms overhead) |
| Enable Detection Filtering | OFF | Area/confidence quality checks (~10ms overhead) |
| Pipeline Preset | balanced | `strict` / `balanced` / `lenient` |

**Recommended for real-time:** Frame Skip 2–3, Scale 0.5–0.75, UI Update 2–3, Preprocessing OFF.

### Detection Thresholds (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Min Brightness | 30 | Skip frames darker than this |
| Max Brightness | 220 | Skip frames brighter than this |
| Min Blur Threshold | 100 | Skip frames below this Laplacian variance |
| No-Helmet Confidence | 0.65 | Minimum confidence to log a violation |

### Training a Model

```bash
python testaugroboflow/train_helmet_detection.py
```

Modify in the script:
- `model` — variant (e.g. `yolov8s.pt`, `yolov9s.pt`)
- `epochs`, `batch`, `imgsz`, `device`

### Running Inference Directly

```python
from ultralytics import YOLO

model = YOLO('runs/detect/helmet_detection_v1/weights/best.pt')
results = model.predict(source='video.mp4', conf=0.5)
```

---

## Detection Pipeline

### Tracking

Uses **Norfair** Euclidean tracker:

```
distance_threshold = 250
initialization_delay = 2
hit_counter_max = 5
past_detections_length = 7
```

Tracks must reach `min_track_age = 3` frames before being considered.

### False Positive Reduction (`media_pipeline.py`)

**Temporal voting** — each track accumulates a rolling vote over the last 5 frames. A detection is only marked confident when the vote score exceeds 0.7.

**Stability scoring** — position variance across recent detections produces a `stability_score` (0–1). Low-stability tracks are filtered from violation logs.

**Frame filtering** (when enabled) — checks brightness, blur (Laplacian variance), bounding box area, and per-class confidence thresholds before passing detections to the tracker.

### Violation Logging

When a `no_helmet` detection is confident and stable:

1. Crop is saved to `violations/no_helmet_captures/`
2. A row is appended to `violations/no_helmet_log.csv`

CSV columns: `timestamp`, `frame_number`, `track_id`, `confidence`, `x1`, `y1`, `x2`, `y2`, `image_filename`, `vote_score`, `stability_score`

Captures are throttled: at most one capture per track per 200 processed frames.

---

## Dashboard

### Video Analysis Tab

- Live annotated video feed during processing
- Per-frame bounding boxes labeled with `ID`, class, vote score, and stability
- Post-processing metrics: motorcycles detected, violations, violation rate, frame count
- Pipeline statistics: filtered detections, blur/brightness issue rates

### Dashboard Tab

- Violation timeline chart (1-minute bins)
- Recent violation records table (latest 20)
- Violation gallery (latest 12 crops with vote and stability scores)

---

## Model Performance

| Metric | YOLOv8s | YOLOv8n | YOLOv9s | YOLOv10n |
|--------|---------|---------|---------|----------|
| **Precision** | **0.9698** ⭐ | 0.6647 | 0.7452 | 0.7749 |
| **Recall** | 0.7601 | 0.6842 | **0.7818** ⭐ | 0.5793 |
| **mAP@0.5** | **0.8804** ⭐ | 0.6328 | 0.7083 | 0.7177 |
| **mAP@0.5-0.95** | **0.6764** ⭐ | 0.4181 | 0.4945 | 0.4399 |

**YOLOv8s** (`finetune.pt`) is used by default — best overall precision and mAP. Use **YOLOv9s** if minimising missed detections is the priority.

---

## Data Augmentation

Roboflow augmentation applied to training data:

- Rotation ±7°, Shearing ±2°, Horizontal flip 50%, Mosaic 50%

Datasets: `dataset/` (original), `dataset_Finetune/`, `testaugroboflow/dataset/` (augmented), `finetundata/`.

**Class imbalance handling:** The `no_helmet` class was oversampled to match majority-class size, improving recall for violations.

---

## Dependencies

```
ultralytics       # YOLO models
torch, torchvision
opencv-python     # Video I/O and frame processing
norfair           # Multi-object tracking
streamlit         # Dashboard UI
pandas, numpy
matplotlib
pyyaml
```

---

## Troubleshooting

**GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory during training:** Reduce `batch`, use a smaller model variant (n/s), or lower `imgsz` to 416/512.

**Too many false positives:** Enable Detection Filtering in the sidebar, switch Pipeline Preset to `strict`, or raise the No-Helmet Confidence threshold.

**Too many missed detections:** Lower No-Helmet Confidence threshold, switch preset to `lenient`, or increase Frame Skip to reduce blur from fast motion.

**Dashboard shows no data:** Process a video first — the log and capture folder are created on the first confirmed violation.

---

## Project Status

- [x] Dataset preparation and augmentation
- [x] Multi-model training (YOLOv8, YOLOv9, YOLOv10)
- [x] Model validation and error analysis
- [x] Norfair multi-object tracking integration
- [x] Temporal voting + stability-based false positive reduction
- [x] Streamlit monitoring dashboard
- [x] Automated violation logging (image + CSV)
- [x] Class imbalance handling (oversampling)
- [ ] Skeleton/pose-based detection (planned)
---

## License

Provided as-is for research and development purposes.