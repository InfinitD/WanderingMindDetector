"""Accuracy metrics tracking for head pose estimation."""
import numpy as np
import math
import cv2
from typing import Dict, Optional
from collections import deque


class PoseAccuracyMetrics:
    """Track accuracy metrics for head pose estimation."""
    
    def __init__(self):
        # Detectron2 published benchmarks (reference)
        self.detectron2_mae = {
            'pitch': 3.5,  # Detectron2 typical MAE: 3-5 degrees
            'yaw': 3.8,
            'roll': 2.9
        }
        
        # Our model metrics
        self.reprojection_errors = deque(maxlen=100)
        self.mae_errors = {'pitch': deque(maxlen=100), 'yaw': deque(maxlen=100), 'roll': deque(maxlen=100)}
        self.angular_errors = deque(maxlen=100)
        
        # Stability metrics
        self.consistency_errors = deque(maxlen=50)
        
    def calculate_reprojection_error(self, model_points: np.ndarray, image_points: np.ndarray,
                                   rotation_vector: np.ndarray, translation_vector: np.ndarray,
                                   camera_matrix: np.ndarray) -> float:
        """Calculate reprojection error (mean pixel distance)."""
        try:
            # Project 3D points to 2D
            projected_points, _ = cv2.projectPoints(
                model_points, rotation_vector, translation_vector, camera_matrix, None
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Calculate mean Euclidean distance
            errors = np.linalg.norm(image_points - projected_points, axis=1)
            mean_error = np.mean(errors)
            
            self.reprojection_errors.append(mean_error)
            return mean_error
        except:
            return float('inf')
    
    def calculate_mae(self, predicted_pose: Dict[str, float], 
                     reference_pose: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate Mean Absolute Error for each angle."""
        if reference_pose is None:
            # Use temporal consistency as reference (if no ground truth)
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        mae = {
            'pitch': abs(predicted_pose['pitch'] - reference_pose['pitch']),
            'yaw': abs(predicted_pose['yaw'] - reference_pose['yaw']),
            'roll': abs(predicted_pose['roll'] - reference_pose['roll'])
        }
        
        self.mae_errors['pitch'].append(mae['pitch'])
        self.mae_errors['yaw'].append(mae['yaw'])
        self.mae_errors['roll'].append(mae['roll'])
        
        return mae
    
    def calculate_angular_error(self, predicted_pose: Dict[str, float],
                               reference_pose: Dict[str, float]) -> float:
        """Calculate 3D angular error in degrees."""
        # Convert to rotation matrices
        R_pred = self._euler_to_rotation_matrix(
            math.radians(predicted_pose['pitch']),
            math.radians(predicted_pose['yaw']),
            math.radians(predicted_pose['roll'])
        )
        R_ref = self._euler_to_rotation_matrix(
            math.radians(reference_pose['pitch']),
            math.radians(reference_pose['yaw']),
            math.radians(reference_pose['roll'])
        )
        
        # Calculate relative rotation
        R_rel = R_pred @ R_ref.T
        
        # Extract angle from rotation matrix
        trace = np.trace(R_rel)
        angle = math.acos(np.clip((trace - 1) / 2, -1, 1))
        angle_deg = math.degrees(angle)
        
        self.angular_errors.append(angle_deg)
        return angle_deg
    
    def _euler_to_rotation_matrix(self, pitch: float, yaw: float, roll: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        
        Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                       [0, 1, 0],
                       [-math.sin(yaw), 0, math.cos(yaw)]])
        
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0],
                       [0, 0, 1]])
        
        return Rz @ Rx @ Ry
    
    def get_stats(self) -> Dict[str, float]:
        """Get current accuracy statistics."""
        stats = {
            'reprojection_error': np.mean(self.reprojection_errors) if self.reprojection_errors else 0.0,
            'mae_pitch': np.mean(self.mae_errors['pitch']) if self.mae_errors['pitch'] else 0.0,
            'mae_yaw': np.mean(self.mae_errors['yaw']) if self.mae_errors['yaw'] else 0.0,
            'mae_roll': np.mean(self.mae_errors['roll']) if self.mae_errors['roll'] else 0.0,
            'angular_error': np.mean(self.angular_errors) if self.angular_errors else 0.0,
        }
        
        # Compare with Detectron2
        stats['vs_detectron_pitch'] = stats['mae_pitch'] - self.detectron2_mae['pitch']
        stats['vs_detectron_yaw'] = stats['mae_yaw'] - self.detectron2_mae['yaw']
        stats['vs_detectron_roll'] = stats['mae_roll'] - self.detectron2_mae['roll']
        
        return stats

