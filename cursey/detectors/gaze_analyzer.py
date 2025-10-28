"""
Gaze Analysis and Mind Wandering Detection Module for Cursey
Analyzes eye movement patterns to detect mind wandering states

Author: Cursey Team
Date: 2025
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque


class GazeAnalyzer:
    """Analyzes gaze patterns to detect mind wandering states."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.gaze_history = deque(maxlen=window_size)
        self.attention_history = deque(maxlen=window_size)
        self.FIXATION_THRESHOLD = 0.05
        self.SACCADE_THRESHOLD = 10
        self.DISPERSION_THRESHOLD = 30
        
    def calculate_gaze_center(self, left_eye: Dict, right_eye: Dict) -> Optional[tuple]:
        if left_eye is None or right_eye is None:
            return None
        left_center = left_eye['features']['center']
        right_center = right_eye['features']['center']
        center_x = (left_center[0] + right_center[0]) / 2
        center_y = (left_center[1] + right_center[1]) / 2
        return (center_x, center_y)
    
    def analyze_fixation(self, gaze_points: List[tuple]) -> Dict:
        if len(gaze_points) < 2:
            return None
        x_coords = [p[0] for p in gaze_points]
        y_coords = [p[1] for p in gaze_points]
        dispersion_x = max(x_coords) - min(x_coords)
        dispersion_y = max(y_coords) - min(y_coords)
        dispersion = np.sqrt(dispersion_x**2 + dispersion_y**2)
        variance = np.var(x_coords) + np.var(y_coords)
        return {
            'duration': len(gaze_points),
            'dispersion': dispersion,
            'stability': variance,
            'center': (np.mean(x_coords), np.mean(y_coords))
        }
    
    def calculate_attention_score(self, eye_data: Dict) -> float:
        if not eye_data or not eye_data.get('eyes'):
            return 0.0
        brightness_scores = []
        aspect_ratios = []
        for eye_pair in eye_data['eyes']:
            if eye_pair.get('left_eye'):
                features = eye_pair['left_eye']['features']
                brightness_scores.append(features['brightness'])
                aspect_ratios.append(features['aspect_ratio'])
            if eye_pair.get('right_eye'):
                features = eye_pair['right_eye']['features']
                brightness_scores.append(features['brightness'])
                aspect_ratios.append(features['aspect_ratio'])
        if not brightness_scores:
            return 0.0
        avg_brightness = np.mean(brightness_scores) / 255.0
        avg_aspect = np.mean(aspect_ratios)
        abs_score = 1.0 - abs(avg_aspect - 1.0)
        attention_score = (avg_brightness * 0.6 + abs_score * 0.4)
        return min(max(attention_score, 0.0), 1.0)
    
    def detect_mind_wandering(self, eye_data: Dict) -> Dict:
        attention_score = self.calculate_attention_score(eye_data)
        self.attention_history.append(attention_score)
        gaze_center = None
        if eye_data and eye_data.get('eyes'):
            for eye_pair in eye_data['eyes']:
                if eye_pair.get('left_eye') and eye_pair.get('right_eye'):
                    gaze_center = self.calculate_gaze_center(eye_pair['left_eye'], eye_pair['right_eye'])
                    break
        if gaze_center:
            self.gaze_history.append(gaze_center)
        result = {'is_wandering': False, 'confidence': 0.0, 'features': {}}
        if len(self.attention_history) < 10:
            return result
        avg_attention = np.mean(list(self.attention_history))
        low_attention = avg_attention < 0.5
        if len(self.gaze_history) >= 5:
            gaze_points = list(self.gaze_history)[-10:]
            fixation_features = self.analyze_fixation(gaze_points)
            if fixation_features:
                high_dispersion = fixation_features['dispersion'] > self.DISPERSION_THRESHOLD
                unstable_gaze = fixation_features['stability'] > self.FIXATION_THRESHOLD
                result['features'] = {
                    'attention_score': avg_attention,
                    'dispersion': fixation_features['dispersion'],
                    'stability': fixation_features['stability'],
                    'low_attention': low_attention,
                    'high_dispersion': high_dispersion,
                    'unstable_gaze': unstable_gaze
                }
                indicators = [low_attention, high_dispersion, unstable_gaze]
                num_indicators = sum(indicators)
                result['is_wandering'] = num_indicators >= 2
                result['confidence'] = num_indicators / 3.0
        return result
    
    def reset(self):
        self.gaze_history.clear()
        self.attention_history.clear()
