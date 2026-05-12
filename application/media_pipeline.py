"""
Media Pipeline for Reducing False Positives in Helmet Detection
Includes: preprocessing, filtering, temporal consistency, and quality checks
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing and quality validation"""
    
    @staticmethod
    def denoise(frame: np.ndarray, strength: int = 10) -> np.ndarray:
        """Apply bilateral filter to reduce noise while preserving edges"""
        return cv2.bilateralFilter(frame, strength, 75, 75)
    
    @staticmethod
    def enhance_contrast(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def get_brightness(frame: np.ndarray) -> float:
        """Calculate average brightness of image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    @staticmethod
    def get_blur_score(frame: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance (higher = less blur)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    @staticmethod
    def is_frame_valid(frame: np.ndarray, 
                      min_brightness: float = 30.0,
                      max_brightness: float = 220.0,
                      min_blur_threshold: float = 100.0) -> Tuple[bool, Dict]:
        """
        Validate frame quality
        Returns: (is_valid, quality_metrics)
        """
        metrics = {
            'brightness': ImagePreprocessor.get_brightness(frame),
            'blur_score': ImagePreprocessor.get_blur_score(frame),
            'width': frame.shape[1],
            'height': frame.shape[0]
        }
        
        is_bright = min_brightness <= metrics['brightness'] <= max_brightness
        is_clear = metrics['blur_score'] >= min_blur_threshold
        
        return is_bright and is_clear, metrics


class DetectionFilter:
    """Filters detections based on multiple criteria"""
    
    @staticmethod
    def validate_box_size(box: np.ndarray, 
                         min_area: float = 100.0,
                         max_area_ratio: float = 0.8,
                         frame_width: int = 640,
                         frame_height: int = 480) -> Tuple[bool, Dict]:
        """
        Validate bounding box size
        Returns: (is_valid, metrics)
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame_width * frame_height
        
        metrics = {
            'area': area,
            'width': width,
            'height': height,
            'aspect_ratio': width / (height + 1e-6)
        }
        
        is_valid = (
            area >= min_area and 
            (area / frame_area) <= max_area_ratio and
            0.2 <= metrics['aspect_ratio'] <= 5.0  # Reasonable aspect ratio
        )
        
        return is_valid, metrics
    
    @staticmethod
    def validate_confidence(confidence: float, 
                           class_id: int,
                           thresholds: Dict[int, float]) -> bool:
        """
        Apply class-specific confidence thresholds
        thresholds: {0: helmet_threshold, 1: motorcycle_threshold, 2: no_helmet_threshold}
        """
        threshold = thresholds.get(class_id, 0.5)
        return confidence >= threshold
    
    @staticmethod
    def validate_visibility(box: np.ndarray,
                           frame_width: int,
                           frame_height: int,
                           min_visible_ratio: float = 0.2) -> Tuple[bool, float]:
        """
        Check if object is sufficiently visible in frame
        Returns: (is_valid, visibility_ratio)
        """
        x1, y1, x2, y2 = box
        
        # Clip to frame bounds
        x1_c = max(0, min(x1, frame_width - 1))
        y1_c = max(0, min(y1, frame_height - 1))
        x2_c = max(0, min(x2, frame_width - 1))
        y2_c = max(0, min(y2, frame_height - 1))
        
        visible_area = (x2_c - x1_c) * (y2_c - y1_c)
        total_area = (x2 - x1) * (y2 - y1)
        
        visibility_ratio = visible_area / max(total_area, 1)
        
        return visibility_ratio >= min_visible_ratio, visibility_ratio


class TemporalConsistency:
    """Handles temporal smoothing and consistency checks"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.class_history = defaultdict(list)
        self.confidence_history = defaultdict(list)
        self.position_history = defaultdict(list)
    
    def update_history(self, track_id: int, 
                      class_id: int,
                      confidence: float,
                      box_center: Tuple[float, float]):
        """Update temporal history for a track"""
        self.class_history[track_id].append(class_id)
        self.confidence_history[track_id].append(confidence)
        self.position_history[track_id].append(box_center)
        
        # Keep history within window size
        if len(self.class_history[track_id]) > self.window_size:
            self.class_history[track_id].pop(0)
            self.confidence_history[track_id].pop(0)
            self.position_history[track_id].pop(0)
    
    def get_smoothed_class(self, track_id: int,
                          current_class: int,
                          voting_threshold: float = 0.7) -> Tuple[int, float]:
        """
        Get majority class with voting
        Returns: (final_class, vote_score)
        """
        if track_id not in self.class_history or len(self.class_history[track_id]) == 0:
            return current_class, 1.0
        
        history = self.class_history[track_id]
        class_counts = defaultdict(int)
        
        for cls in history:
            class_counts[cls] += 1
        
        majority_class = max(class_counts, key=class_counts.get)
        vote_score = class_counts[majority_class] / len(history)
        
        return majority_class, vote_score
    
    def get_average_confidence(self, track_id: int) -> float:
        """Get average confidence for a track"""
        if track_id not in self.confidence_history or len(self.confidence_history[track_id]) == 0:
            return 0.0
        
        return float(np.mean(self.confidence_history[track_id]))
    
    def get_position_stability(self, track_id: int) -> float:
        """
        Calculate position stability (lower = more stable)
        Uses coefficient of variation of positions
        """
        if track_id not in self.position_history or len(self.position_history[track_id]) < 2:
            return 1.0
        
        positions = np.array(self.position_history[track_id])
        std_dev = np.std(positions, axis=0)
        mean_pos = np.mean(positions, axis=0)
        
        # Coefficient of variation
        cv = np.mean(std_dev / (np.abs(mean_pos) + 1e-6))
        
        return cv
    
    def is_stable_track(self, track_id: int, 
                       max_position_cv: float = 0.3) -> bool:
        """Check if track has stable position"""
        cv = self.get_position_stability(track_id)
        return cv <= max_position_cv
    
    def clean_track(self, track_id: int):
        """Remove track from history"""
        self.class_history.pop(track_id, None)
        self.confidence_history.pop(track_id, None)
        self.position_history.pop(track_id, None)


class FalsePositiveReducer:
    """Combined strategy to reduce false positives"""
    
    def __init__(self,
                 # Image quality thresholds
                 min_brightness: float = 30.0,
                 max_brightness: float = 220.0,
                 min_blur_threshold: float = 100.0,
                 # Detection thresholds
                 confidence_thresholds: Dict[int, float] = None,
                 min_detection_area: float = 100.0,
                 max_area_ratio: float = 0.8,
                 # Temporal thresholds
                 voting_window: int = 5,
                 voting_threshold: float = 0.7,
                 min_track_age: int = 3):
        
        self.image_preprocessor = ImagePreprocessor()
        self.detection_filter = DetectionFilter()
        self.temporal_consistency = TemporalConsistency(window_size=voting_window)
        
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_blur_threshold = min_blur_threshold
        
        # Default confidence thresholds: helmet, motorcycle, no_helmet
        self.confidence_thresholds = confidence_thresholds or {
            0: 0.5,  # helmet
            1: 0.4,  # motorcycle
            2: 0.6   # no_helmet (stricter)
        }
        
        self.min_detection_area = min_detection_area
        self.max_area_ratio = max_area_ratio
        self.voting_threshold = voting_threshold
        self.min_track_age = min_track_age
    
    def preprocess_frame(self, frame: np.ndarray, 
                        denoise: bool = True,
                        enhance: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess frame and validate quality
        Returns: (processed_frame, quality_metrics)
        """
        processed = frame.copy()
        
        if denoise:
            processed = self.image_preprocessor.denoise(processed)
        
        if enhance:
            processed = self.image_preprocessor.enhance_contrast(processed)
        
        is_valid, metrics = self.image_preprocessor.is_frame_valid(
            processed,
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
            min_blur_threshold=self.min_blur_threshold
        )
        
        return processed, metrics
    
    def filter_detection(self, box: np.ndarray,
                        confidence: float,
                        class_id: int,
                        frame_width: int,
                        frame_height: int) -> Tuple[bool, Dict]:
        """
        Apply all filters to a detection
        Returns: (is_valid, filter_results)
        """
        results = {'passed_all': True, 'filters': {}}
        
        # Filter 1: Confidence threshold
        conf_valid = self.detection_filter.validate_confidence(
            confidence, class_id, self.confidence_thresholds
        )
        results['filters']['confidence'] = conf_valid
        
        # Filter 2: Box size
        size_valid, size_metrics = self.detection_filter.validate_box_size(
            box, self.min_detection_area, self.max_area_ratio, 
            frame_width, frame_height
        )
        results['filters']['size'] = size_valid
        results['size_metrics'] = size_metrics
        
        # Filter 3: Visibility
        vis_valid, vis_ratio = self.detection_filter.validate_visibility(
            box, frame_width, frame_height
        )
        results['filters']['visibility'] = vis_valid
        results['visibility_ratio'] = vis_ratio
        
        results['passed_all'] = all(results['filters'].values())
        
        return results['passed_all'], results
    
    def apply_temporal_smoothing(self, track_id: int,
                                 class_id: int,
                                 confidence: float,
                                 box: np.ndarray,
                                 track_age: int) -> Tuple[int, float, bool]:
        """
        Apply temporal consistency to detection
        Returns: (smoothed_class, vote_score, is_confident_detection)
        """
        # Calculate box center
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        
        self.temporal_consistency.update_history(
            track_id, class_id, confidence, (cx, cy)
        )
        
        smoothed_class, vote_score = self.temporal_consistency.get_smoothed_class(
            track_id, class_id, self.voting_threshold
        )
        
        # A detection is confident if:
        # 1. Voting threshold is met
        # 2. Track is stable
        # 3. Track age is sufficient
        avg_conf = self.temporal_consistency.get_average_confidence(track_id)
        is_stable = self.temporal_consistency.is_stable_track(track_id)
        has_age = track_age >= self.min_track_age
        
        is_confident = (vote_score >= self.voting_threshold and 
                       is_stable and 
                       has_age and
                       avg_conf >= self.confidence_thresholds.get(smoothed_class, 0.5))
        
        return smoothed_class, vote_score, is_confident
    
    def cleanup_track(self, track_id: int):
        """Remove track from temporal history"""
        self.temporal_consistency.clean_track(track_id)
    
    def get_pipeline_metrics(self) -> Dict:
        """Get current pipeline statistics"""
        return {
            'tracked_objects': len(self.temporal_consistency.class_history),
            'brightness_range': (self.min_brightness, self.max_brightness),
            'blur_threshold': self.min_blur_threshold,
            'confidence_thresholds': self.confidence_thresholds,
            'voting_threshold': self.voting_threshold
        }


class VideoAnalyzer:
    """High-level video analysis with media pipeline"""
    
    def __init__(self, reducer: FalsePositiveReducer):
        self.reducer = reducer
        self.frame_stats = defaultdict(int)
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze frame quality
        Returns quality metrics
        """
        processed, metrics = self.reducer.preprocess_frame(frame)
        self.frame_stats['total_frames'] += 1
        
        if not (self.reducer.min_brightness <= metrics['brightness'] <= self.reducer.max_brightness):
            self.frame_stats['brightness_issues'] += 1
        
        if metrics['blur_score'] < self.reducer.min_blur_threshold:
            self.frame_stats['blur_issues'] += 1
        
        return processed, metrics
    
    def get_stats(self) -> Dict:
        """Get analysis statistics"""
        total = self.frame_stats['total_frames']
        return {
            'total_frames': total,
            'brightness_issues': self.frame_stats['brightness_issues'],
            'blur_issues': self.frame_stats['blur_issues'],
            'brightness_issue_rate': (self.frame_stats['brightness_issues'] / max(total, 1)) * 100,
            'blur_issue_rate': (self.frame_stats['blur_issues'] / max(total, 1)) * 100
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.frame_stats = defaultdict(int)


# Example usage and configuration presets
REDUCER_PRESETS = {
    'strict': {
        'confidence_thresholds': {0: 0.6, 1: 0.5, 2: 0.75},
        'min_detection_area': 200.0,
        'voting_threshold': 0.8,
        'min_blur_threshold': 150.0,
    },
    'balanced': {
        'confidence_thresholds': {0: 0.5, 1: 0.4, 2: 0.6},
        'min_detection_area': 100.0,
        'voting_threshold': 0.7,
        'min_blur_threshold': 100.0,
    },
    'lenient': {
        'confidence_thresholds': {0: 0.4, 1: 0.3, 2: 0.5},
        'min_detection_area': 50.0,
        'voting_threshold': 0.6,
        'min_blur_threshold': 80.0,
    }
}


def create_reducer(preset: str = 'balanced', **kwargs) -> FalsePositiveReducer:
    """Create a FalsePositiveReducer with preset configuration"""
    config = REDUCER_PRESETS.get(preset, REDUCER_PRESETS['balanced']).copy()
    config.update(kwargs)
    return FalsePositiveReducer(**config)
