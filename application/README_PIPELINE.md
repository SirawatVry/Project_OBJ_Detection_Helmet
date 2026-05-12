# Application - Media Pipeline Integration

## Quick Start

### Running the Streamlit Application

```bash
# Activate virtual environment (if not already active)
.\norfair_env\Scripts\Activate.ps1

# Run the application
streamlit run streamlit_deploy.py
```

The application will be available at `http://localhost:8501`

## New Features

### Media Pipeline Controls

In the left sidebar, you now have access to:

1. **Pipeline Preset** - Choose from:
   - `strict`: Maximum false positive reduction (high security)
   - `balanced`: Good balance (default, recommended)
   - `lenient`: Maximum detection coverage

2. **Advanced Settings**:
   - **Min Brightness**: Minimum acceptable frame brightness
   - **Max Brightness**: Maximum acceptable frame brightness
   - **Blur Threshold**: Minimum image clarity required
   - **No Helmet Confidence**: Threshold for no-helmet detections

### Analysis Tabs

#### Video Analysis Tab
- Upload and process videos
- Real-time detection visualization
- Pipeline statistics showing:
  - Motorcycles detected
  - Violations found
  - Violation rate
  - Filtered detections
  - Violations logged
  - Frame quality metrics (blur/brightness issues)

#### Dashboard Tab
- View violation timeline
- Browse recent violations
- Gallery of violation captures
- Stability scores for each violation

## Understanding Pipeline Metrics

### Detection Filtering
- **Filtered Detections**: Number of detections rejected by quality checks
- **Violations Logged**: Number of confirmed no-helmet violations

### Frame Quality
- **Blur Issues**: Percentage of frames that don't meet blur threshold
- **Brightness Issues**: Percentage of frames outside brightness range

## Configuration Recommendations

### For Different Scenarios

**High Security (Banks, Government)**
- Use `strict` preset
- Increase No Helmet Confidence to 0.75+
- Set high Blur Threshold

**Urban Traffic**
- Use `balanced` preset (default)
- Keep settings at mid-range
- Monitor blur/brightness metrics

**Night Time Monitoring**
- Use `lenient` preset
- Lower Min Brightness to 20-30
- Reduce Blur Threshold to 80-90

**Highway/Fast Moving**
- Use `balanced` preset
- Lower Blur Threshold to 80
- Increase No Helmet Confidence slightly

## Files Included

### Core Pipeline Files
- `media_pipeline.py` - Main pipeline implementation
- `pipeline_config.py` - Configuration utilities and presets
- `pipeline_examples.py` - Usage examples

### Application Files
- `streamlit_deploy.py` - Updated Streamlit application
- `MEDIA_PIPELINE_GUIDE.md` - Comprehensive documentation

### Documentation
- `README.md` - Project overview
- This file - Quick start guide

## Using the Pipeline in Custom Code

```python
from media_pipeline import FalsePositiveReducer, create_reducer

# Option 1: Use preset
reducer = create_reducer(preset='balanced')

# Option 2: Custom configuration
reducer = FalsePositiveReducer(
    confidence_thresholds={0: 0.5, 1: 0.4, 2: 0.7},
    voting_threshold=0.75,
    min_blur_threshold=100.0
)

# Preprocess frame
processed, metrics = reducer.preprocess_frame(frame)

# Filter detection
is_valid, results = reducer.filter_detection(
    box=detection_box,
    confidence=detection_conf,
    class_id=detection_class,
    frame_width=640,
    frame_height=480
)

# Apply temporal smoothing
smoothed_class, vote_score, is_confident = reducer.apply_temporal_smoothing(
    track_id=track_id,
    class_id=class_id,
    confidence=confidence,
    box=box,
    track_age=age
)
```

## Performance Tips

### Reducing False Positives
1. Switch to `strict` preset
2. Increase No Helmet Confidence slider
3. Check frame quality metrics
4. Increase Blur Threshold

### Improving Detection Rate
1. Switch to `lenient` preset
2. Lower No Helmet Confidence
3. Use in well-lit environments

### Optimizing for Speed
1. Use `balanced` preset
2. Lower Blur Threshold
3. Reduce Min Brightness requirements

## Troubleshooting

### Too Many False Positives?
- Switch from `balanced` to `strict` preset
- Increase "No Helmet Confidence" slider
- Check if "Blur Issues" percentage is high

### Missing Detections?
- Switch to `lenient` preset
- Lower "No Helmet Confidence" slider
- Check lighting conditions (low brightness errors)

### Slow Processing?
- The pipeline should add minimal overhead
- Most time is spent on YOLO inference
- Try reducing video resolution if needed

## Advanced Usage

### Batch Processing
See `pipeline_examples.py` for batch processing example

### Custom Presets
```python
from pipeline_config import PipelineConfig

# Create custom based on preset
config = PipelineConfig.create_custom(
    base_preset='balanced',
    confidence_thresholds={0: 0.55, 1: 0.45, 2: 0.72}
)

# Validate
is_valid, errors = PipelineConfig.validate_config(config)

# Use
reducer = FalsePositiveReducer(**config)
```

### Monitoring Statistics
```python
from media_pipeline import VideoAnalyzer

analyzer = VideoAnalyzer(reducer)

# Process frames...
processed, metrics = analyzer.analyze_frame(frame)

# Get stats
stats = analyzer.get_stats()
print(f"Blur Issues: {stats['blur_issue_rate']:.1f}%")
```

## Output Files

### Violation Captures
- Location: `./violations/no_helmet_captures/`
- Naming: `no_helmet_{track_id}_{timestamp}.jpg`

### Violation Log
- Location: `./violations/no_helmet_log.csv`
- Columns: timestamp, frame_number, track_id, confidence, x1, y1, x2, y2, image_filename, vote_score, stability_score

## Support & Documentation

For detailed information:
- See `MEDIA_PIPELINE_GUIDE.md` for comprehensive documentation
- See `pipeline_examples.py` for code examples
- See `pipeline_config.py` for available configurations

## Version History

### v1.0 - Initial Release
- Image preprocessing with denoising and contrast enhancement
- Multi-criteria detection filtering
- Temporal consistency tracking
- 5 preset configurations
- Streamlit integration with sidebar controls
- Real-time statistics and monitoring
- Batch processing capabilities

## Next Steps

1. **Test Different Presets**: Try `strict`, `balanced`, and `lenient` on your videos
2. **Monitor Metrics**: Check blur/brightness issue percentages
3. **Fine-tune**: Adjust sliders based on your specific scenario
4. **Collect Data**: Review violation logs to validate accuracy
5. **Optimize**: Fine-tune thresholds for your environment

---

For questions or improvements, refer to the detailed documentation in `MEDIA_PIPELINE_GUIDE.md`
