"""
Media Pipeline Configuration and Utilities
Presets and helper functions for different detection scenarios
"""

from media_pipeline import FalsePositiveReducer, REDUCER_PRESETS
from typing import Dict, Optional


class PipelineConfig:
    """Centralized configuration for media pipeline"""
    
    # Preset configurations for different scenarios
    PRESETS = {
        'strict': {
            'description': 'Maximum false positive reduction - high confidence required',
            'use_case': 'Security-critical scenarios',
            'config': {
                'min_brightness': 40.0,
                'max_brightness': 210.0,
                'min_blur_threshold': 150.0,
                'confidence_thresholds': {
                    0: 0.65,  # helmet
                    1: 0.55,  # motorcycle
                    2: 0.75   # no_helmet
                },
                'min_detection_area': 200.0,
                'max_area_ratio': 0.7,
                'voting_threshold': 0.8,
                'voting_window': 7,
                'min_track_age': 5
            }
        },
        'balanced': {
            'description': 'Good balance between detection rate and false positives',
            'use_case': 'General monitoring',
            'config': {
                'min_brightness': 30.0,
                'max_brightness': 220.0,
                'min_blur_threshold': 100.0,
                'confidence_thresholds': {
                    0: 0.5,   # helmet
                    1: 0.4,   # motorcycle
                    2: 0.65   # no_helmet
                },
                'min_detection_area': 100.0,
                'max_area_ratio': 0.8,
                'voting_threshold': 0.7,
                'voting_window': 5,
                'min_track_age': 3
            }
        },
        'lenient': {
            'description': 'Maximum detection coverage - accepts some false positives',
            'use_case': 'Comprehensive coverage needed',
            'config': {
                'min_brightness': 20.0,
                'max_brightness': 240.0,
                'min_blur_threshold': 80.0,
                'confidence_thresholds': {
                    0: 0.4,   # helmet
                    1: 0.3,   # motorcycle
                    2: 0.5    # no_helmet
                },
                'min_detection_area': 50.0,
                'max_area_ratio': 0.9,
                'voting_threshold': 0.6,
                'voting_window': 3,
                'min_track_age': 2
            }
        },
        'night_mode': {
            'description': 'Optimized for low-light conditions',
            'use_case': 'Night-time monitoring',
            'config': {
                'min_brightness': 10.0,
                'max_brightness': 255.0,
                'min_blur_threshold': 50.0,
                'confidence_thresholds': {
                    0: 0.45,
                    1: 0.35,
                    2: 0.55
                },
                'min_detection_area': 80.0,
                'max_area_ratio': 0.85,
                'voting_threshold': 0.65,
                'voting_window': 6,
                'min_track_age': 3
            }
        },
        'highway': {
            'description': 'Optimized for high-speed scenarios',
            'use_case': 'Highway/fast-moving traffic',
            'config': {
                'min_brightness': 30.0,
                'max_brightness': 220.0,
                'min_blur_threshold': 80.0,  # Lower blur threshold for moving objects
                'confidence_thresholds': {
                    0: 0.55,
                    1: 0.45,
                    2: 0.68
                },
                'min_detection_area': 150.0,
                'max_area_ratio': 0.75,
                'voting_threshold': 0.72,
                'voting_window': 4,  # Shorter window for faster response
                'min_track_age': 2
            }
        }
    }
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict:
        """Get preset configuration"""
        if preset_name not in PipelineConfig.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PipelineConfig.PRESETS.keys())}")
        return PipelineConfig.PRESETS[preset_name]['config'].copy()
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """List all available presets with descriptions"""
        return {
            name: info['description'] 
            for name, info in PipelineConfig.PRESETS.items()
        }
    
    @staticmethod
    def get_preset_info(preset_name: str) -> Dict:
        """Get full preset information including description and use case"""
        if preset_name not in PipelineConfig.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        preset = PipelineConfig.PRESETS[preset_name]
        return {
            'name': preset_name,
            'description': preset['description'],
            'use_case': preset['use_case'],
            'config': preset['config']
        }
    
    @staticmethod
    def create_custom(
        base_preset: str = 'balanced',
        **overrides
    ) -> Dict:
        """Create custom configuration based on a preset with overrides"""
        config = PipelineConfig.get_preset(base_preset)
        config.update(overrides)
        return config
    
    @staticmethod
    def validate_config(config: Dict) -> tuple[bool, list]:
        """
        Validate configuration dictionary
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        required_keys = [
            'min_brightness', 'max_brightness', 'min_blur_threshold',
            'confidence_thresholds', 'min_detection_area', 'max_area_ratio',
            'voting_threshold', 'min_track_age'
        ]
        
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")
        
        # Validate ranges
        if config.get('min_brightness', 0) >= config.get('max_brightness', 255):
            errors.append("min_brightness must be less than max_brightness")
        
        if not 0 <= config.get('voting_threshold', 0.5) <= 1.0:
            errors.append("voting_threshold must be between 0 and 1")
        
        if config.get('min_detection_area', 0) < 0:
            errors.append("min_detection_area must be non-negative")
        
        if not 0 <= config.get('max_area_ratio', 0.5) <= 1.0:
            errors.append("max_area_ratio must be between 0 and 1")
        
        return len(errors) == 0, errors


# Export utilities
__all__ = ['PipelineConfig', 'FalsePositiveReducer', 'REDUCER_PRESETS']
