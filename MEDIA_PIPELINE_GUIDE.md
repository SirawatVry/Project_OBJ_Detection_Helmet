# Media Pipeline for False Positive Reduction

## Overview

The media pipeline is a comprehensive system designed to reduce false positives in the helmet detection system. It includes:

1. **Image Preprocessing** - Denoising, contrast enhancement, quality validation
2. **Detection Filtering** - Confidence-based filtering, size/visibility validation
3. **Temporal Consistency** - Voting-based class smoothing, stability tracking
4. **Advanced Metrics** - Position stability, track quality scoring

## Key Components

### 1. ImagePreprocessor
Handles frame-level preprocessing and quality validation.

**Features:**
- Bilateral filtering for noise reduction
- CLAHE for contrast enhancement
- Brightness and blur scoring
- Frame quality validation

**Usage:**
```python
from media_pipeline import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Denoise a frame
denoised = preprocessor.denoise(frame)

# Enhance contrast
enhanced = preprocessor.enhance_contrast(frame)

# Check frame quality
is_valid, metrics = preprocessor.is_frame_valid(frame)
print(f"Brightness: {metrics['brightness']}")
print(f"Blur Score: {metrics['blur_score']}")
```

### 2. DetectionFilter
Validates individual detections against multiple criteria.

**Filters Applied:**
- **Confidence Threshold**: Class-specific minimum confidence requirements
- **Box Size**: Minimum area and aspect ratio validation
- **Visibility**: Ensures object is sufficiently visible in frame

**Usage:**
```python
from media_pipeline import DetectionFilter

filter_obj = DetectionFilter()

# Validate confidence
is_conf_valid = filter_obj.validate_confidence(
    confidence=0.85,
    class_id=2,  # no_helmet
    thresholds={0: 0.5, 1: 0.4, 2: 0.6}
)

# Validate box size
is_size_valid, metrics = filter_obj.validate_box_size(
    box=np.array([10, 20, 100, 150]),
    min_area=100.0,
    max_area_ratio=0.8,
    frame_width=640,
    frame_height=480
)

# Validate visibility
is_visible, visibility_ratio = filter_obj.validate_visibility(
    box=np.array([10, 20, 100, 150]),
    frame_width=640,
    frame_height=480,
    min_visible_ratio=0.2
)
```

### 3. TemporalConsistency
Tracks temporal patterns to smooth class predictions and identify stable objects.

**Features:**
- Voting-based class smoothing
- Average confidence tracking
- Position stability measurement
- Track history management

**Usage:**
```python
from media_pipeline import TemporalConsistency

temporal = TemporalConsistency(window_size=5)

# Update history for a track
temporal.update_history(
    track_id=1,
    class_id=2,
    confidence=0.85,
    box_center=(320, 240)
)

# Get smoothed class with voting
smoothed_class, vote_score = temporal.get_smoothed_class(
    track_id=1,
    current_class=2,
    voting_threshold=0.7
)

# Check if track is stable
is_stable = temporal.is_stable_track(track_id=1, max_position_cv=0.3)

# Get average confidence
avg_conf = temporal.get_average_confidence(track_id=1)

# Clean up track
temporal.clean_track(track_id=1)
```

### 4. FalsePositiveReducer
Main orchestrator combining all components into a unified pipeline.

**Usage:**
```python
from media_pipeline import FalsePositiveReducer

reducer = FalsePositiveReducer(
    min_brightness=30.0,
    max_brightness=220.0,
    min_blur_threshold=100.0,
    confidence_thresholds={0: 0.5, 1: 0.4, 2: 0.65},
    min_detection_area=100.0,
    max_area_ratio=0.8,
    voting_threshold=0.7,
    min_track_age=3
)

# Preprocess frame
processed, metrics = reducer.preprocess_frame(
    frame,
    denoise=True,
    enhance=True
)

# Filter detection
is_valid, results = reducer.filter_detection(
    box=np.array([10, 20, 100, 150]),
    confidence=0.85,
    class_id=2,
    frame_width=640,
    frame_height=480
)

# Apply temporal smoothing
smoothed_class, vote_score, is_confident = reducer.apply_temporal_smoothing(
    track_id=1,
    class_id=2,
    confidence=0.85,
    box=np.array([10, 20, 100, 150]),
    track_age=5
)

# Clean up when track ends
reducer.cleanup_track(track_id=1)
```

## Configuration Presets

The pipeline includes several pre-configured presets for different scenarios:

### Preset: `balanced` (Default)
- **Use Case**: General monitoring
- **Description**: Good balance between detection rate and false positives
- **Helmet Conf**: 0.5
- **No-Helmet Conf**: 0.65
- **Voting Threshold**: 0.7

### Preset: `strict`
- **Use Case**: Security-critical scenarios
- **Description**: Maximum false positive reduction
- **Helmet Conf**: 0.65
- **No-Helmet Conf**: 0.75
- **Voting Threshold**: 0.8

### Preset: `lenient`
- **Use Case**: Comprehensive coverage needed
- **Description**: Maximum detection coverage, accepts some false positives
- **Helmet Conf**: 0.4
- **No-Helmet Conf**: 0.5
- **Voting Threshold**: 0.6

### Preset: `night_mode`
- **Use Case**: Low-light conditions
- **Description**: Optimized for night-time monitoring
- **Adjustments**: Lower blur threshold, relaxed brightness bounds

### Preset: `highway`
- **Use Case**: High-speed traffic scenarios
- **Description**: Faster response time for moving objects
- **Adjustments**: Shorter voting window, adapted blur tolerance

**Using Presets:**
```python
from media_pipeline import create_reducer

# Create with preset
reducer = create_reducer(preset='strict')

# Create with preset and overrides
reducer = create_reducer(
    preset='balanced',
    confidence_thresholds={0: 0.6, 1: 0.5, 2: 0.7}
)
```

## Integration with Streamlit Application

The Streamlit application automatically integrates the media pipeline with the following features:

### Sidebar Configuration
Users can adjust pipeline settings in the sidebar:
- **Preset Selection**: Quick switching between presets
- **Brightness Range**: Min/max brightness thresholds
- **Blur Threshold**: Image clarity requirement
- **No-Helmet Confidence**: Specific threshold for no-helmet class

### Pipeline Statistics
After processing, displays:
- Number of filtered detections (rejected)
- Number of violations logged
- Percentage of frames with blur issues
- Percentage of frames with brightness issues

### Enhanced Labels
Detections show:
- Tracking ID and class
- Voting score (confidence in class prediction)
- Stability score (for no-helmet detections)

## Performance Tips

### Reducing False Positives
1. Use `strict` preset as starting point
2. Increase `no_helmet_conf` threshold
3. Increase `voting_threshold` to require more consensus
4. Increase `min_blur_threshold` to skip unclear frames

### Improving Detection Rate
1. Use `lenient` preset
2. Lower confidence thresholds
3. Lower voting thresholds
4. Reduce min_track_age

### For Specific Conditions
- **Poor lighting**: Use `night_mode` preset
- **Fast-moving objects**: Use `highway` preset
- **Crowded scenes**: Use `strict` preset with larger `min_detection_area`
- **Static scenes**: Use `balanced` preset with higher voting threshold

## Customization Examples

### Example 1: High-Security Installation
```python
config = {
    'min_brightness': 40.0,
    'max_brightness': 210.0,
    'min_blur_threshold': 150.0,
    'confidence_thresholds': {0: 0.7, 1: 0.6, 2: 0.8},
    'min_detection_area': 250.0,
    'max_area_ratio': 0.65,
    'voting_threshold': 0.85,
    'min_track_age': 6
}
reducer = FalsePositiveReducer(**config)
```

### Example 2: Urban Traffic Monitoring
```python
config = {
    'min_brightness': 25.0,
    'max_brightness': 230.0,
    'min_blur_threshold': 90.0,
    'confidence_thresholds': {0: 0.48, 1: 0.38, 2: 0.62},
    'min_detection_area': 80.0,
    'max_area_ratio': 0.85,
    'voting_threshold': 0.68,
    'min_track_age': 2
}
reducer = FalsePositiveReducer(**config)
```

### Example 3: Night-Time Monitoring
```python
config = {
    'min_brightness': 15.0,
    'max_brightness': 250.0,
    'min_blur_threshold': 60.0,
    'confidence_thresholds': {0: 0.42, 1: 0.32, 2: 0.52},
    'min_detection_area': 70.0,
    'max_area_ratio': 0.88,
    'voting_threshold': 0.65,
    'min_track_age': 3
}
reducer = FalsePositiveReducer(**config)
```

## Monitoring and Diagnostics

### Check Pipeline Metrics
```python
metrics = reducer.get_pipeline_metrics()
print(f"Tracked Objects: {metrics['tracked_objects']}")
print(f"Brightness Range: {metrics['brightness_range']}")
print(f"Confidence Thresholds: {metrics['confidence_thresholds']}")
```

### Analyze Video Quality
```python
from media_pipeline import VideoAnalyzer

analyzer = VideoAnalyzer(reducer)

# Process frames...
processed, quality = analyzer.analyze_frame(frame)

# Get statistics
stats = analyzer.get_stats()
print(f"Total Frames: {stats['total_frames']}")
print(f"Blur Issues: {stats['blur_issue_rate']:.1f}%")
print(f"Brightness Issues: {stats['brightness_issue_rate']:.1f}%")
```

## Best Practices

1. **Start Conservative**: Begin with `strict` preset and gradually relax if needed
2. **Monitor Metrics**: Regularly check pipeline statistics for quality issues
3. **Test Different Scenarios**: Validate performance in various lighting/traffic conditions
4. **Adjust Iteratively**: Make small changes to thresholds and observe impact
5. **Keep Logs**: Review violation logs to identify pattern changes

## Troubleshooting

### Too Many False Positives
- Increase `confidence_thresholds[2]` (no-helmet threshold)
- Increase `voting_threshold`
- Use `strict` preset
- Check for lighting issues with blur/brightness metrics

### Missing Detections
- Decrease confidence thresholds
- Decrease voting threshold
- Use `lenient` preset
- Check frame quality (blur/brightness)

### Unstable Tracking
- Increase `voting_window`
- Check tracking parameters in YOLO tracker
- Ensure consistent lighting
- Verify camera angle and mounting

## Performance Benchmarks

Typical performance on standard hardware:
- **Frame processing**: ~30-50ms per frame (with preprocessing)
- **Memory overhead**: ~50-100MB for tracking history
- **False positive reduction**: 30-60% depending on conditions

## References

- OpenCV Bilateral Filter: Noise reduction with edge preservation
- CLAHE: Contrast Limited Adaptive Histogram Equalization
- Norfair Tracking: Multi-object tracking library
- YOLOv8: Real-time object detection
