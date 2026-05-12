"""
Example: Using Media Pipeline for Batch Video Processing
Demonstrates how to use the media pipeline in different scenarios
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from media_pipeline import FalsePositiveReducer, VideoAnalyzer, create_reducer
from pipeline_config import PipelineConfig


def example_1_basic_preprocessing():
    """Example 1: Basic frame preprocessing and quality check"""
    print("\n=== Example 1: Basic Frame Preprocessing ===")
    
    # Create reducer
    reducer = create_reducer(preset='balanced')
    
    # Load a frame (example - replace with actual frame)
    frame = cv2.imread('path/to/frame.jpg')
    
    # Preprocess
    processed, metrics = reducer.preprocess_frame(frame)
    
    print(f"Brightness: {metrics['brightness']:.1f}")
    print(f"Blur Score: {metrics['blur_score']:.1f}")
    print(f"Frame Size: {metrics['width']}x{metrics['height']}")
    
    # Save processed frame
    cv2.imwrite('processed_frame.jpg', processed)


def example_2_detection_filtering():
    """Example 2: Filter detections based on multiple criteria"""
    print("\n=== Example 2: Detection Filtering ===")
    
    reducer = create_reducer(preset='strict')
    
    # Example detections from YOLO
    detections = [
        {'box': np.array([10, 20, 100, 150]), 'conf': 0.95, 'class': 0},  # Helmet - high conf
        {'box': np.array([120, 30, 200, 200]), 'conf': 0.35, 'class': 2},  # No-helmet - low conf
        {'box': np.array([5, 5, 10, 10]), 'conf': 0.8, 'class': 1},  # Motorcycle - too small
    ]
    
    frame_width, frame_height = 640, 480
    valid_detections = []
    
    for det in detections:
        is_valid, results = reducer.filter_detection(
            det['box'], det['conf'], det['class'],
            frame_width, frame_height
        )
        
        print(f"\nClass: {['helmet', 'motorcycle', 'no_helmet'][det['class']]}")
        print(f"  Confidence: {det['conf']:.2f}")
        print(f"  Valid: {is_valid}")
        print(f"  Filters: {results['filters']}")
        
        if is_valid:
            valid_detections.append(det)
    
    print(f"\nTotal: {len(detections)} -> {len(valid_detections)} (Valid)")


def example_3_temporal_smoothing():
    """Example 3: Temporal consistency and class smoothing"""
    print("\n=== Example 3: Temporal Smoothing ===")
    
    reducer = create_reducer(preset='balanced')
    
    # Simulate tracking over 10 frames
    track_id = 1
    detections_sequence = [
        {'class': 2, 'conf': 0.55},  # Frame 1
        {'class': 2, 'conf': 0.58},  # Frame 2
        {'class': 1, 'conf': 0.45},  # Frame 3 (different class - noise)
        {'class': 2, 'conf': 0.62},  # Frame 4
        {'class': 2, 'conf': 0.61},  # Frame 5
    ]
    
    for frame_idx, det in enumerate(detections_sequence):
        # Simulate box (just for example)
        box = np.array([100 + frame_idx*10, 100, 200 + frame_idx*10, 200])
        
        smoothed_class, vote_score, is_confident = reducer.apply_temporal_smoothing(
            track_id=track_id,
            class_id=det['class'],
            confidence=det['conf'],
            box=box,
            track_age=frame_idx + 1
        )
        
        class_names = ['helmet', 'motorcycle', 'no_helmet']
        print(f"Frame {frame_idx + 1}: {class_names[det['class']]} "
              f"({det['conf']:.2f}) -> {class_names[smoothed_class]} "
              f"({vote_score:.0%}) {'✓ Confident' if is_confident else '✗ Not sure'}")
    
    # Check stability
    stability_cv = reducer.temporal_consistency.get_position_stability(track_id)
    print(f"\nPosition Stability: {stability_cv:.3f} (lower = more stable)")
    
    # Cleanup
    reducer.cleanup_track(track_id)


def example_4_batch_processing():
    """Example 4: Batch process multiple videos with statistics"""
    print("\n=== Example 4: Batch Processing ===")
    
    # Configuration for batch processing
    video_dir = Path('path/to/videos')
    results_file = 'batch_results.json'
    
    reducer = create_reducer(preset='balanced')
    analyzer = VideoAnalyzer(reducer)
    
    batch_results = {}
    
    # Process each video
    for video_file in video_dir.glob('*.mp4'):
        print(f"\nProcessing: {video_file.name}")
        
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        detected_violations = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess and analyze
            processed, metrics = analyzer.analyze_frame(frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        
        # Store results
        stats = analyzer.get_stats()
        batch_results[video_file.name] = {
            'total_frames': stats['total_frames'],
            'blur_issues': stats['blur_issue_rate'],
            'brightness_issues': stats['brightness_issue_rate']
        }
        
        analyzer.reset_stats()
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


def example_5_custom_configuration():
    """Example 5: Create custom configuration for specific scenario"""
    print("\n=== Example 5: Custom Configuration ===")
    
    # List available presets
    presets = PipelineConfig.list_presets()
    print("Available Presets:")
    for name, description in presets.items():
        print(f"  - {name}: {description}")
    
    # Get detailed preset info
    preset_info = PipelineConfig.get_preset_info('strict')
    print(f"\nStrict Preset Details:")
    print(f"  Description: {preset_info['description']}")
    print(f"  Use Case: {preset_info['use_case']}")
    print(f"  No-Helmet Confidence: {preset_info['config']['confidence_thresholds'][2]}")
    
    # Create custom based on preset
    custom_config = PipelineConfig.create_custom(
        base_preset='balanced',
        confidence_thresholds={0: 0.55, 1: 0.45, 2: 0.70},
        voting_threshold=0.75
    )
    
    # Validate configuration
    is_valid, errors = PipelineConfig.validate_config(custom_config)
    print(f"\nCustom Config Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Create reducer with custom config
    reducer = FalsePositiveReducer(**custom_config)
    print(f"\nCreated reducer with custom configuration")


def example_6_adaptive_pipeline():
    """Example 6: Adaptive pipeline based on frame quality"""
    print("\n=== Example 6: Adaptive Pipeline ===")
    
    # Create multiple reducers for different conditions
    reducers = {
        'normal': create_reducer(preset='balanced'),
        'low_light': create_reducer(preset='night_mode'),
        'high_speed': create_reducer(preset='highway')
    }
    
    # Analyze frame quality and choose appropriate reducer
    frame = cv2.imread('path/to/frame.jpg')  # Replace with actual frame
    if frame is None:
        print("Example frame not found - skipping quality analysis")
        return
    
    reducer = reducers['normal']
    processed, metrics = reducer.preprocess_frame(frame)
    
    brightness = metrics['brightness']
    blur_score = metrics['blur_score']
    
    print(f"Frame Analysis:")
    print(f"  Brightness: {brightness:.1f}")
    print(f"  Blur Score: {blur_score:.1f}")
    
    # Choose reducer based on quality
    if brightness < 50:
        reducer = reducers['low_light']
        print("  -> Using low_light reducer")
    elif blur_score < 80:
        reducer = reducers['high_speed']
        print("  -> Using high_speed reducer")
    else:
        print("  -> Using normal reducer")


def example_7_comparison():
    """Example 7: Compare different presets on same video"""
    print("\n=== Example 7: Preset Comparison ===")
    
    presets_to_test = ['strict', 'balanced', 'lenient']
    
    # Example detection
    detections = [
        {'box': np.array([100, 100, 200, 200]), 'conf': 0.52, 'class': 2},
    ]
    
    results = {}
    
    for preset_name in presets_to_test:
        reducer = create_reducer(preset=preset_name)
        
        valid_count = 0
        for det in detections:
            is_valid, _ = reducer.filter_detection(
                det['box'], det['conf'], det['class'], 640, 480
            )
            if is_valid:
                valid_count += 1
        
        preset_info = PipelineConfig.get_preset_info(preset_name)
        results[preset_name] = {
            'valid_detections': valid_count,
            'no_helmet_threshold': preset_info['config']['confidence_thresholds'][2],
            'description': preset_info['description']
        }
    
    print(f"\nComparison Results (1 detection with no-helmet conf=0.52):")
    for preset, result in results.items():
        print(f"\n{preset.upper()}:")
        print(f"  Description: {result['description']}")
        print(f"  No-Helmet Threshold: {result['no_helmet_threshold']}")
        print(f"  Valid Detections: {result['valid_detections']}")


if __name__ == "__main__":
    print("Media Pipeline Examples")
    print("=" * 50)
    
    # Run examples (comment out as needed)
    try:
        example_1_basic_preprocessing()
    except Exception as e:
        print(f"Example 1 skipped: {e}")
    
    example_2_detection_filtering()
    example_3_temporal_smoothing()
    
    try:
        example_4_batch_processing()
    except Exception as e:
        print(f"Example 4 skipped: {e}")
    
    example_5_custom_configuration()
    
    try:
        example_6_adaptive_pipeline()
    except Exception as e:
        print(f"Example 6 skipped: {e}")
    
    example_7_comparison()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
