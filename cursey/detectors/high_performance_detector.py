"""
High-Performance Face Detection System
Optimized for 60 FPS real-time detection

Author: Cursey Team
Date: 2025
"""

import cv2
import numpy as np
import math
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FastFaceDetection:
    """Optimized face detection result for high performance."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    pose_angles: Optional[Dict[str, float]] = None
    attention_vector: Optional[Tuple[float, float, float]] = None
    quality_score: float = 0.0


@dataclass
class FastEyeDetection:
    """Optimized eye detection for high performance."""
    center: Tuple[int, int]
    confidence: float
    gaze_vector: Tuple[float, float]


class HighPerformanceDetector:
    """
    High-performance face detection system optimized for 60 FPS.
    
    Optimizations:
    - Reduced processing overhead
    - Cached computations
    - Optimized algorithms
    - Minimal memory allocation
    - Frame skipping strategies
    """
    
    def __init__(self, max_persons: int = 3, confidence_threshold: float = 0.5):
        self.max_persons = max_persons
        self.confidence_threshold = confidence_threshold
        
        # Detection history for temporal stability (reduced size)
        self.detection_history = deque(maxlen=5)  # Reduced from 15
        self.tracking_persons = {}
        self.next_person_id = 0
        
        # Performance optimizations
        self.frame_skip = 1  # Process every frame for 60fps
        self.frame_counter = 0
        self.last_detection_time = 0
        self.detection_cache = None
        self.cache_valid_frames = 2  # Cache valid for 2 frames
        
        # Pre-computed values for performance
        self.dof_limits = {
            'pitch': {'min': -45, 'max': 45},
            'yaw': {'min': -60, 'max': 60},
            'roll': {'min': -30, 'max': 30}
        }
        
        # Load cascade classifiers once
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Pre-allocate arrays for performance
        self.temp_gray = None
        self.temp_roi = None
        
        # AOI system
        self.aoi_reference = None
        self.aoi_tolerance = 0.05
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'avg_confidence': 0.0,
            'detection_time': 0.0,
            'fps': 0.0
        }
        
        # Timing for FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
    
    def _preprocess_frame_fast(self, frame: np.ndarray) -> np.ndarray:
        """Fast preprocessing optimized for performance."""
        # Pre-allocate if needed
        if self.temp_gray is None or self.temp_gray.shape != frame.shape[:2]:
            self.temp_gray = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Convert to grayscale (in-place if possible)
        if len(frame.shape) == 3:
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.temp_gray)
            return self.temp_gray
        else:
            return frame
    
    def _detect_faces_fast(self, frame: np.ndarray) -> List[FastFaceDetection]:
        """Fast face detection with minimal overhead."""
        # Use cached detection if available
        if (self.detection_cache is not None and 
            self.frame_counter - self.last_detection_time < self.cache_valid_frames):
            return self.detection_cache
        
        gray = self._preprocess_frame_fast(frame)
        
        # Optimized cascade detection parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Slightly coarser for speed
            minNeighbors=4,       # Reduced for speed
            minSize=(30, 30),     # Smaller minimum size
            maxSize=(200, 200),   # Smaller maximum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        
        for (x, y, w, h) in faces:
            # Fast confidence calculation
            confidence = self._calculate_fast_confidence(gray, (x, y, w, h))
            
            if confidence >= self.confidence_threshold:
                # Fast pose calculation
                pose_angles = self._calculate_fast_pose(gray, (x, y, w, h))
                attention_vector = self._calculate_fast_attention_vector(pose_angles)
                
                detection = FastFaceDetection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    pose_angles=pose_angles,
                    attention_vector=attention_vector,
                    quality_score=confidence  # Use confidence as quality for speed
                )
                detections.append(detection)
        
        # Cache the result
        self.detection_cache = detections
        self.last_detection_time = self.frame_counter
        
        return detections
    
    def _calculate_fast_confidence(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Fast confidence calculation with minimal operations."""
        x, y, w, h = bbox
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            return 0.0
        
        # Fast confidence factors
        factors = []
        
        # 1. Edge density (simplified)
        edges = cv2.Canny(face_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        edge_factor = min(1.0, edge_density * 10)
        factors.append(edge_factor)
        
        # 2. Contrast (simplified)
        contrast = np.std(face_roi)
        contrast_factor = min(1.0, contrast / 40.0)
        factors.append(contrast_factor)
        
        # 3. Aspect ratio (simplified)
        aspect_ratio = w / h
        aspect_factor = max(0.0, 1.0 - abs(aspect_ratio - 1.0))
        factors.append(aspect_factor)
        
        # 4. Size factor (simplified)
        size_factor = min(1.0, (w * h) / (60 * 60))
        factors.append(size_factor)
        
        # Fast weighted average
        confidence = sum(factors) / len(factors)
        return min(1.0, max(0.0, confidence))
    
    def _calculate_fast_pose(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Fast pose calculation using simplified methods."""
        x, y, w, h = bbox
        face_roi = gray[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        # Fast eye detection
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.2,  # Coarser for speed
            minNeighbors=3,  # Reduced for speed
            minSize=(10, 10),
            maxSize=(w//2, h//2)
        )
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # Calculate eye centers
            left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
            right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
            
            # Fast pose calculation
            eye_distance = abs(right_center[0] - left_center[0])
            if eye_distance > 0:
                # Roll from eye line
                roll = math.degrees(math.atan2(right_center[1] - left_center[1], 
                                             right_center[0] - left_center[0]))
                
                # Yaw from eye center offset
                eye_center_x = (left_center[0] + right_center[0]) / 2
                nose_offset_x = (w//2) - eye_center_x
                yaw = (nose_offset_x / eye_distance) * 30
                
                # Pitch from vertical offset
                eye_center_y = (left_center[1] + right_center[1]) / 2
                nose_offset_y = (h//2) - eye_center_y
                pitch = (nose_offset_y / eye_distance) * 30
            else:
                pitch = yaw = roll = 0.0
        else:
            pitch = yaw = roll = 0.0
        
        return {
            'pitch': np.clip(pitch, self.dof_limits['pitch']['min'], self.dof_limits['pitch']['max']),
            'yaw': np.clip(yaw, self.dof_limits['yaw']['min'], self.dof_limits['yaw']['max']),
            'roll': np.clip(roll, self.dof_limits['roll']['min'], self.dof_limits['roll']['max'])
        }
    
    def _calculate_fast_attention_vector(self, pose_angles: Dict[str, float]) -> Tuple[float, float, float]:
        """Fast attention vector calculation."""
        if not pose_angles:
            return (0.0, 0.0, 0.0)
        
        pitch = pose_angles.get('pitch', 0.0)
        yaw = pose_angles.get('yaw', 0.0)
        roll = pose_angles.get('roll', 0.0)
        
        return (pitch, yaw, roll)
    
    def _detect_eyes_fast(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Optional[FastEyeDetection]]:
        """Fast eye detection with minimal processing."""
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {'left': None, 'right': None}
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = face_roi
        
        # Fast eye detection
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi, 
            scaleFactor=1.2,  # Coarser for speed
            minNeighbors=3,   # Reduced for speed
            minSize=(8, 8),   # Smaller minimum
            maxSize=(w//3, h//3)
        )
        
        result = {'left': None, 'right': None}
        
        if len(eyes) >= 2:
            # Sort by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_box = eyes[0]
            right_eye_box = eyes[1]
            
            # Calculate centers
            left_center = (x + left_eye_box[0] + left_eye_box[2]//2, 
                          y + left_eye_box[1] + left_eye_box[3]//2)
            right_center = (x + right_eye_box[0] + right_eye_box[2]//2, 
                           y + right_eye_box[1] + right_eye_box[3]//2)
            
            # Fast gaze estimation
            left_gaze = self._estimate_fast_gaze(gray_roi, left_eye_box)
            right_gaze = self._estimate_fast_gaze(gray_roi, right_eye_box)
            
            result['left'] = FastEyeDetection(
                center=left_center,
                confidence=0.8,  # Simplified confidence
                gaze_vector=left_gaze
            )
            result['right'] = FastEyeDetection(
                center=right_center,
                confidence=0.8,
                gaze_vector=right_gaze
            )
        
        return result
    
    def _estimate_fast_gaze(self, eye_roi: np.ndarray, eye_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Fast gaze estimation using brightness distribution."""
        ex, ey, ew, eh = eye_box
        eye_region = eye_roi[ey:ey+eh, ex:ex+ew]
        
        if eye_region.size == 0:
            return (0.0, 0.0)
        
        h, w = eye_region.shape
        
        # Fast brightness-based gaze estimation
        left_brightness = np.mean(eye_region[:, :w//2])
        right_brightness = np.mean(eye_region[:, w//2:])
        top_brightness = np.mean(eye_region[:h//2, :])
        bottom_brightness = np.mean(eye_region[h//2:, :])
        
        gaze_x = (right_brightness - left_brightness) / max(np.mean(eye_region), 1) * 15
        gaze_y = (bottom_brightness - top_brightness) / max(np.mean(eye_region), 1) * 15
        
        return (gaze_x, gaze_y)
    
    def _assign_person_id_fast(self, bbox: Tuple[int, int, int, int]) -> int:
        """Fast person ID assignment."""
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        
        # Find closest existing person
        min_distance = float('inf')
        closest_id = None
        
        for person_id, person_data in self.tracking_persons.items():
            if time.time() - person_data['timestamp'] > 2.0:  # Stale detection
                continue
            
            prev_bbox = person_data['face_detection'].bbox
            prev_center = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
            
            distance = math.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_id = person_id
        
        # Assign ID
        if min_distance < 60 and closest_id is not None:
            return closest_id
        else:
            # New person
            person_id = self.next_person_id
            self.next_person_id += 1
            return person_id
    
    def set_aoi_reference(self, pose_angles: Dict[str, float]) -> None:
        """Set AOI reference values."""
        self.aoi_reference = pose_angles.copy()
        logger.info(f"AOI reference set: {self.aoi_reference}")
    
    def check_aoi_compliance(self, current_pose: Dict[str, float]) -> bool:
        """Check if current pose is within AOI tolerance."""
        if not self.aoi_reference:
            return False
        
        for angle_name in ['pitch', 'yaw', 'roll']:
            if angle_name in current_pose and angle_name in self.aoi_reference:
                current_val = current_pose[angle_name]
                reference_val = self.aoi_reference[angle_name]
                
                # Calculate tolerance range
                tolerance_range = abs(reference_val) * self.aoi_tolerance
                min_val = reference_val - tolerance_range
                max_val = reference_val + tolerance_range
                
                if not (min_val <= current_val <= max_val):
                    return False
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame with high-performance detection."""
        start_time = time.time()
        
        # Update frame counter
        self.frame_counter += 1
        
        try:
            # Detect faces with fast algorithm
            face_detections = self._detect_faces_fast(frame)
            
            # Process each detection
            persons = {}
            for detection in face_detections[:self.max_persons]:
                # Detect eyes
                eyes = self._detect_eyes_fast(frame, detection.bbox)
                
                # Create person tracker
                person_id = self._assign_person_id_fast(detection.bbox)
                
                # Store person data
                persons[person_id] = {
                    'face_detection': detection,
                    'eyes': eyes,
                    'timestamp': time.time(),
                    'aoi_compliant': self.check_aoi_compliance(detection.pose_angles or {})
                }
            
            # Update performance stats
            detection_time = time.time() - start_time
            self.performance_stats['detection_time'] = detection_time
            self.performance_stats['total_detections'] += len(face_detections)
            
            # Calculate FPS
            self.fps_frame_count += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                fps = self.fps_frame_count / (current_time - self.fps_start_time)
                self.performance_stats['fps'] = fps
                self.fps_frame_count = 0
                self.fps_start_time = current_time
            
            if face_detections:
                avg_confidence = sum(d.confidence for d in face_detections) / len(face_detections)
                self.performance_stats['avg_confidence'] = avg_confidence
            
            return {
                'persons': persons,
                'frame': frame,
                'detection_count': len(face_detections),
                'performance_stats': self.performance_stats
            }
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return {
                'persons': {},
                'frame': frame,
                'detection_count': 0,
                'performance_stats': self.performance_stats
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def get_dof_limits(self) -> Dict[str, Dict[str, int]]:
        """Get DOF limits for face detection."""
        return self.dof_limits.copy()


if __name__ == "__main__":
    print("High-Performance Face Detection System loaded successfully!")
    print("Features:")
    print("- Optimized for 60 FPS performance")
    print("- Fast preprocessing and detection")
    print("- Cached computations")
    print("- Minimal memory allocation")
    print("- Real-time pose estimation")
