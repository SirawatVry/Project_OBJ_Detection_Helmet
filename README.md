# Helmet Detection Project

A machine learning project for detecting helmets, motorcycles, and people without helmets in images and video streams using YOLOv8, YOLOv9, and YOLOv10 models.

## Project Overview

This project implements real-time helmet detection to identify:
- **Helmet** - People wearing helmets
- **Motorcycle** - Motorcycles in the frame
- **No Helmet** - People not wearing helmets

The project includes:
- Multiple YOLO model versions (YOLOv8, YOLOv9, YOLOv10)
- Data augmentation and preprocessing pipelines
- Model training and fine-tuning capabilities
- Comprehensive validation and error analysis
- Real-time detection inference

## Directory Structure

```
Project_ML/
├── basemodel/                          # Base model weights and plots
├── dataset/                            # Original training dataset
│   ├── train/                          # Training images and labels
│   └── valid/                          # Validation images and labels
├── dataset_Finetune/                   # Fine-tuning dataset
├── finetundata/                        # Additional fine-tuning data
├── testaugroboflow/                    # Test and augmentation scripts
│   ├── train_helmet_detection.py       # Main training script
│   ├── fix_annotations.py              # Label annotation fixes
│   └── dataset/                        # Roboflow-augmented dataset
├── validation_results/                 # Validation metrics and results
├── error_visualization/                # Error analysis visualizations
│   ├── errors/                         # General errors
│   ├── false_positives/                # False positive cases
│   ├── false_negatives/                # False negative cases
│   └── true_positives/                 # True positive examples
├── false_negatives_with_boxes/         # Detailed false negative analysis
├── no_helmet_captures/                 # Samples of no-helmet detections
├── violations/                         # Violation logs and captures
├── runs/                               # Training runs and model outputs
└── norfair_env/                        # Python virtual environment

```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, recommended)
- pip or conda

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd c:\Users\title\Downloads\Project_ML
```

2. **Create and activate virtual environment:**
```bash
# Using existing environment or create new one
python -m venv norfair_env
.\norfair_env\Scripts\activate
```

3. **Install required dependencies:**
```bash
pip install ultralytics torch torchvision opencv-python pyyaml pandas numpy matplotlib
```

### Dataset Format

The project uses YOLO format for annotations:
- Images stored in `images/` subdirectory
- Labels stored in `labels/` subdirectory (`.txt` files)
- `data.yaml` configuration file specifying class names

Example `data.yaml`:
```yaml
train: train/images
val: valid/images

nc: 3
names: [helmet, motorcycle, no_helmet]
```

## Usage

### Training a Model

Use the training script to train YOLO models:

```bash
python testaugroboflow/train_helmet_detection.py
```

The script:
- Auto-detects GPU availability
- Supports multiple YOLO model sizes (n, s, m, l, x)
- Includes data augmentation options
- Implements early stopping with patience=20
- Saves results to `runs/detect/`

### Model Configuration

Modify these parameters in `train_helmet_detection.py`:
- `model`: Model variant (e.g., 'yolov8s.pt', 'yolov9s.pt', 'yolov10n.pt')
- `epochs`: Number of training epochs
- `batch`: Batch size (adjust based on GPU memory)
- `imgsz`: Image size (default: 640)
- `device`: GPU device ID or 'cpu'

### Running Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/helmet_detection_v1/weights/best.pt')

# Detect on image
results = model.predict(source='image.jpg', conf=0.5)

# Detect on video
results = model.predict(source='video.mp4', conf=0.5)
```

## Model Variants Tested

The project includes experiments with multiple YOLO versions:
- **YOLOv8** (nano, small, medium, large, xlarge)
- **YOLOv9** (small variant)
- **YOLOv10** (nano variant)

Results stored in `runs/detect/` with experiment names:
- `helmet_detection_v8_aug` - YOLOv8 with augmentation
- `helmet_detection_v8_noaug` - YOLOv8 without augmentation
- `helmet_detection_yolov9s` - YOLOv9 small
- `helmet_detection_yolov10n` - YOLOv10 nano

## Validation & Evaluation

### Validation Results

Validation outputs are stored in `validation_results/`:
- `v16_validation/` through `v19_validation/` - Multiple validation runs
- Contains precision, recall, and mAP metrics
- Includes confusion matrices and performance plots

### Error Analysis

The `error_visualization/` directory contains:
- **True Positives**: Correctly detected objects
- **False Positives**: Incorrect detections
- **False Negatives**: Missed detections
- **Errors**: General classification errors

### Fixing Annotations

Run annotation fixes if needed:
```bash
python testaugroboflow/fix_annotations.py
```

## Data Augmentation

The project uses Roboflow for data augmentation with options including:
- Rotation (±7°)
- Shearing (±2°)
- Horizontal flipping (50%)
- Mosaic augmentation (50%)

Original and augmented datasets:
- `dataset/` - Original dataset
- `dataset_Finetune/` - Fine-tuning dataset
- `testaugroboflow/dataset/` - Roboflow-augmented dataset
- `finetundata/` - Additional fine-tuning data

## Performance Metrics

### Model Comparison Results

| Metric | YOLOv8s | YOLOv8n | YOLOv9s | YOLOv10n |
|--------|---------|---------|---------|----------|
| **Precision** | **0.9698** ⭐ | 0.6647 | 0.7452 | 0.7749 |
| **Recall** | 0.7601 | 0.6842 | **0.7818** ⭐ | 0.5793 |
| **mAP@0.5** | **0.8804** ⭐ | 0.6328 | 0.7083 | 0.7177 |
| **mAP@0.5-0.95** | **0.6764** ⭐ | 0.4181 | 0.4945 | 0.4399 |

**Key Findings:**
- **YOLOv8s** achieves the best overall performance with highest Precision (96.98%), mAP@0.5 (88.04%), and mAP@0.5-0.95 (67.64%)
- **YOLOv9s** provides best Recall (78.18%), useful for minimizing missed detections
- **YOLOv10n** offers lightweight inference at the cost of recall performance
- **YOLOv8n** is the fastest but shows lower detection accuracy

### Metric Definitions

- **Precision**: Accuracy of positive predictions (true positives / all positive predictions)
- **Recall**: Coverage of actual positives (true positives / all actual positives)
- **mAP@0.5**: Mean Average Precision at IoU threshold of 0.5
- **mAP@0.5-0.95**: Mean Average Precision averaged across IoU thresholds from 0.5 to 0.95

## Dependencies

Key Python packages:
- `ultralytics` - YOLO implementation
- `torch` & `torchvision` - Deep learning framework
- `opencv-python` - Computer vision operations
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `pyyaml` - Configuration files

See `norfair_env/` for full environment details.

## Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If False, install CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
- Reduce `batch` size in training script
- Use smaller model variant (n, s instead of l, x)
- Reduce `imgsz` to 416 or 512

### Poor Detection Results
- Verify dataset labels are correct (check with fix_annotations.py)
- Increase training epochs
- Use augmentation if not already enabled
- Check class distribution and balance if needed

## Challenges & Solutions

### Class Imbalance Problem
**Challenge**: The "No Helmet" class had significantly fewer samples compared to "Helmet" and "Motorcycle" classes, leading to:
- Biased model predictions
- Lower recall for the minority class
- Suboptimal detection performance

**Solution Implemented**:
- **Oversampling**: Applied oversampling technique to balance class distribution
- Generated synthetic samples of the "No Helmet" class to match majority class size
- Result: More balanced training data improving model generalization

### Future Optimization: Skeleton-Based Detection
**Approach**: Reduce dataset size and computational requirements by:
- Extracting human skeleton/pose information from images
- Using relative skeleton positions instead of raw pixel data
- Benefits:
  - Significantly smaller dataset size
  - Faster training and inference
  - Better generalization across different camera angles
- **Status**: Planned for future implementation

## Project Status

- [x] Dataset preparation and augmentation
- [x] Multiple model training (YOLOv8, YOLOv9, YOLOv10)
- [x] Model validation and evaluation
- [x] Error analysis and visualization
- [x] Fine-tuning pipeline
- [x] Class imbalance handling with oversampling
- [ ] Skeleton-based detection optimization
- [ ] Real-time inference optimization
- [ ] Model deployment and API

## License

This project is provided as-is for research and development purposes.

## Contact

For questions or issues, refer to the project documentation or contact the project maintainer.
