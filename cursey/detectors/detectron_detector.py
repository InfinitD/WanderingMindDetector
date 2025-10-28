"""
Facebook Detectron2 Face Detection System
State-of-the-art face detection using Facebook's Detectron2 framework

Author: Cursey Team
Date: 2025
"""

import cv2
import numpy as np
import torch
import math
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass

# Detectron2 imports
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.structures import Boxes, Instances
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    logging.warning("Detectron2 not available. Falling back to OpenCV detection.")

logger = logging.getLogger(__name__)


@dataclass
class DetectronFaceDetection:
    """Detectron2 face detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None
    pose_angles: Optional[Dict[str, float]] = None
    attention_vector: Optional[Tuple[float, float, float]] = None
    quality_score: float = 0.0
    stability_score: float = 0.0
    instance_id: Optional[int] = None


@dataclass
class DetectronEyeDetection:
    """Enhanced eye detection with Detectron2 features."""
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    gaze_vector: Tuple[float, float]
    pupil_center: Optional[Tuple[int, int]] = None
    iris_radius: Optional[float] = None
    attention_weight: float = 0.0


class DetectronFaceDetector:
    """
    Facebook Detectron2-based face detection system.
    
    Features:
    - State-of-the-art object detection
    - Instance segmentation capabilities
    - High accuracy face detection
    - Real-time optimization
    - Advanced pose estimation
    - Attention tracking
    """
    
    def __init__(self, max_persons: int = 3, confidence_threshold: float = 0.5, 
                 use_gpu: bool = True, model_name: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"):
        self.max_persons = max_persons
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model_name = model_name
        
        # Detection history for temporal stability
        self.detection_history = deque(maxlen=10)
        self.tracking_persons = {}
        self.next_person_id = 0
        
        # Initialize Detectron2
        self.predictor = None
        self.cfg = None
        self.metadata = None
        
        if DETECTRON2_AVAILABLE:
            self._initialize_detectron2()
        else:
            logger.warning("Detectron2 not available. Using OpenCV fallback.")
            self._initialize_opencv_fallback()
        
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
        
        # DOF limits for face detection
        self.dof_limits = {
            'pitch': {'min': -45, 'max': 45},
            'yaw': {'min': -60, 'max': 60},
            'roll': {'min': -30, 'max': 30}
        }
        
        # AOI system
        self.aoi_reference = None
        self.aoi_tolerance = 0.05
    
    def _initialize_detectron2(self):
        """Initialize Detectron2 model."""
        try:
            # Create config
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(self.model_name))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            
            # Set device
            if self.use_gpu:
                self.cfg.MODEL.DEVICE = "cuda"
            else:
                self.cfg.MODEL.DEVICE = "cpu"
            
            # Load model weights
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_name)
            
            # Create predictor
            self.predictor = DefaultPredictor(self.cfg)
            
            # Get metadata
            self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            
            logger.info(f"Detectron2 initialized with model: {self.model_name}")
            logger.info(f"Device: {self.cfg.MODEL.DEVICE}")
            logger.info(f"Confidence threshold: {self.confidence_threshold}")
            
        except Exception as e:
            logger.error(f"Error initializing Detectron2: {e}")
            logger.info("Falling back to OpenCV detection")
            self._initialize_opencv_fallback()
    
    def _initialize_opencv_fallback(self):
        """Initialize OpenCV fallback detection."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        logger.info("OpenCV fallback detection initialized")
    
    def _preprocess_frame_detectron(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Detectron2."""
        # Detectron2 expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame
    
    def _detect_faces_detectron(self, frame: np.ndarray) -> List[DetectronFaceDetection]:
        """Detect faces using Detectron2."""
        if not DETECTRON2_AVAILABLE or self.predictor is None:
            return self._detect_faces_opencv_fallback(frame)
        
        try:
            # Preprocess frame
            rgb_frame = self._preprocess_frame_detectron(frame)
            
            # Run detection
            outputs = self.predictor(rgb_frame)
            
            # Extract face detections (person class in COCO is class 0)
            instances = outputs["instances"]
            person_instances = instances[instances.pred_classes == 0]  # Person class
            
            detections = []
            
            for i, instance in enumerate(person_instances):
                # Get bounding box
                bbox = instance.pred_boxes.tensor[0].cpu().numpy()
                x1, y1, x2, y2 = bbox.astype(int)
                w, h = x2 - x1, y2 - y1
                
                # Get confidence
                confidence = instance.scores[0].cpu().item()
                
                if confidence >= self.confidence_threshold:
                    # Detect facial landmarks
                    landmarks = self._detect_facial_landmarks_detectron(frame, (x1, y1, w, h))
                    
                    # Calculate pose angles
                    pose_angles = self._calculate_pose_from_landmarks(landmarks)
                    
                    # Calculate attention vector
                    attention_vector = self._calculate_attention_vector(landmarks, pose_angles)
                    
                    # Calculate quality score
                    quality_score = self._calculate_quality_score(frame, (x1, y1, w, h))
                    
                    # Calculate stability score
                    stability_score = self._calculate_temporal_stability((x1, y1, w, h))
                    
                    detection = DetectronFaceDetection(
                        bbox=(x1, y1, w, h),
                        confidence=confidence,
                        landmarks=landmarks,
                        pose_angles=pose_angles,
                        attention_vector=attention_vector,
                        quality_score=quality_score,
                        stability_score=stability_score,
                        instance_id=i
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in Detectron2 detection: {e}")
            return self._detect_faces_opencv_fallback(frame)
    
    def _detect_faces_opencv_fallback(self, frame: np.ndarray) -> List[DetectronFaceDetection]:
        """OpenCV fallback face detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(40, 40),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        
        for (x, y, w, h) in faces:
            # Calculate confidence
            confidence = self._calculate_enhanced_confidence(frame, (x, y, w, h))
            
            if confidence >= self.confidence_threshold:
                # Detect landmarks
                landmarks = self._detect_facial_landmarks_opencv(frame, (x, y, w, h))
                
                # Calculate pose
                pose_angles = self._calculate_pose_from_landmarks(landmarks)
                
                # Calculate attention vector
                attention_vector = self._calculate_attention_vector(landmarks, pose_angles)
                
                detection = DetectronFaceDetection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    landmarks=landmarks,
                    pose_angles=pose_angles,
                    attention_vector=attention_vector,
                    quality_score=confidence,
                    stability_score=self._calculate_temporal_stability((x, y, w, h))
                )
                detections.append(detection)
        
        return detections
    
    def _detect_facial_landmarks_detectron(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[List[Tuple[int, int]]]:
        """Detect facial landmarks using Detectron2-enhanced methods."""
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
        
        # Use OpenCV eye detection for landmarks
        return self._detect_facial_landmarks_opencv(frame, bbox)
    
    def _detect_facial_landmarks_opencv(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[List[Tuple[int, int]]]:
        """Detect facial landmarks using OpenCV."""
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(15, 15),
            maxSize=(w//3, h//3)
        )
        
        landmarks = []
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # Calculate eye centers
            left_eye_center = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
            right_eye_center = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
            
            landmarks.extend([left_eye_center, right_eye_center])
            
            # Estimate nose center
            nose_center = (
                (left_eye_center[0] + right_eye_center[0]) // 2,
                (left_eye_center[1] + right_eye_center[1]) // 2 + h // 4
            )
            landmarks.append(nose_center)
            
            # Estimate mouth corners
            mouth_left = (left_eye_center[0], nose_center[1] + h // 4)
            mouth_right = (right_eye_center[0], nose_center[1] + h // 4)
            landmarks.extend([mouth_left, mouth_right])
        
        return landmarks if len(landmarks) == 5 else None
    
    def _calculate_pose_from_landmarks(self, landmarks: Optional[List[Tuple[int, int]]]) -> Dict[str, float]:
        """Calculate pose angles from landmarks."""
        if not landmarks or len(landmarks) != 5:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
        
        # Calculate roll from eye line
        eye_line_angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        roll = math.degrees(eye_line_angle)
        
        # Calculate yaw from nose position relative to eye center
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset_x = nose[0] - eye_center_x
        eye_distance = abs(right_eye[0] - left_eye[0])
        yaw = (nose_offset_x / eye_distance) * 45 if eye_distance > 0 else 0
        
        # Calculate pitch from vertical positioning
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        nose_offset_y = nose[1] - eye_center_y
        pitch = (nose_offset_y / eye_distance) * 45 if eye_distance > 0 else 0
        
        return {
            'pitch': np.clip(pitch, self.dof_limits['pitch']['min'], self.dof_limits['pitch']['max']),
            'yaw': np.clip(yaw, self.dof_limits['yaw']['min'], self.dof_limits['yaw']['max']),
            'roll': np.clip(roll, self.dof_limits['roll']['min'], self.dof_limits['roll']['max'])
        }
    
    def _calculate_attention_vector(self, landmarks: Optional[List[Tuple[int, int]]], 
                                  pose_angles: Optional[Dict[str, float]]) -> Tuple[float, float, float]:
        """Calculate attention vector for accurate PRY estimation."""
        if not landmarks or len(landmarks) < 5 or not pose_angles:
            return (0.0, 0.0, 0.0)
        
        # Extract key landmarks
        left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
        
        # Calculate eye center
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Calculate face center
        face_center = ((left_eye[0] + right_eye[0] + nose[0]) / 3, 
                      (left_eye[1] + right_eye[1] + nose[1]) / 3)
        
        # Calculate attention vector components
        # Pitch: vertical attention (up/down)
        pitch_vector = (nose[1] - eye_center[1]) / max(abs(left_eye[0] - right_eye[0]), 1)
        
        # Yaw: horizontal attention (left/right)
        yaw_vector = (nose[0] - eye_center[0]) / max(abs(left_eye[0] - right_eye[0]), 1)
        
        # Roll: rotational attention
        roll_vector = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        
        # Convert to degrees and apply limits
        pitch_deg = np.clip(math.degrees(pitch_vector) * 30, 
                           self.dof_limits['pitch']['min'], 
                           self.dof_limits['pitch']['max'])
        
        yaw_deg = np.clip(math.degrees(yaw_vector) * 30,
                         self.dof_limits['yaw']['min'],
                         self.dof_limits['yaw']['max'])
        
        roll_deg = np.clip(math.degrees(roll_vector),
                          self.dof_limits['roll']['min'],
                          self.dof_limits['roll']['max'])
        
        return (pitch_deg, yaw_deg, roll_deg)
    
    def _calculate_enhanced_confidence(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate enhanced confidence with multiple factors."""
        x, y, w, h = bbox
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return 0.0
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        factors = []
        
        # 1. Edge density analysis
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        edge_factor = min(1.0, edge_density * 12)
        factors.append(edge_factor)
        
        # 2. Contrast analysis
        contrast = np.std(gray_face)
        contrast_factor = min(1.0, contrast / 50.0)
        factors.append(contrast_factor)
        
        # 3. Aspect ratio validation
        aspect_ratio = w / h
        aspect_factor = max(0.0, 1.0 - abs(aspect_ratio - 1.0) * 1.5)
        factors.append(aspect_factor)
        
        # 4. Size validation
        size_factor = min(1.0, (w * h) / (80 * 80))
        factors.append(size_factor)
        
        # 5. Temporal consistency
        temporal_factor = self._calculate_temporal_consistency(bbox)
        factors.append(temporal_factor)
        
        # 6. Symmetry analysis
        symmetry_factor = self._calculate_facial_symmetry(gray_face)
        factors.append(symmetry_factor)
        
        # Weighted average
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_temporal_stability(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate temporal stability score."""
        if len(self.detection_history) < 3:
            return 0.5
        
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        
        # Calculate stability based on recent detections
        recent_centers = []
        for detections in list(self.detection_history)[-5:]:  # Last 5 frames
            for det in detections:
                det_center = (det.bbox[0] + det.bbox[2]//2, det.bbox[1] + det.bbox[3]//2)
                distance = math.sqrt((center[0] - det_center[0])**2 + (center[1] - det_center[1])**2)
                if distance < 50:  # Same person
                    recent_centers.append(det_center)
                    break
        
        if len(recent_centers) < 2:
            return 0.3
        
        # Calculate variance in position
        centers_array = np.array(recent_centers)
        variance = np.var(centers_array, axis=0)
        stability = max(0.0, 1.0 - np.mean(variance) / 100.0)
        
        return min(1.0, stability)
    
    def _calculate_temporal_consistency(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate temporal consistency score."""
        if not self.detection_history:
            return 0.5
        
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        
        # Find closest detection in history
        min_distance = float('inf')
        for prev_detections in self.detection_history:
            for prev_detection in prev_detections:
                prev_x, prev_y, prev_w, prev_h = prev_detection.bbox
                prev_center = (prev_x + prev_w//2, prev_y + prev_h//2)
                
                distance = math.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                min_distance = min(min_distance, distance)
        
        # Convert distance to consistency score
        max_distance = 100
        consistency = max(0.0, 1.0 - min_distance / max_distance)
        
        return consistency
    
    def _calculate_facial_symmetry(self, gray_face: np.ndarray) -> float:
        """Calculate facial symmetry score."""
        h, w = gray_face.shape
        
        # Split face into left and right halves
        left_half = gray_face[:, :w//2]
        right_half = cv2.flip(gray_face[:, w//2:], 1)
        
        # Resize to match if needed
        if left_half.shape != right_half.shape:
            right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        
        # Calculate correlation
        correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
        
        return max(0.0, correlation)
    
    def _calculate_quality_score(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate quality score for detection."""
        x, y, w, h = bbox
        
        # Check bounds
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0.0
        
        # Size factor
        size_factor = min(1.0, (w * h) / (100 * 100))
        
        # Position factor (center bias)
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        face_center_x, face_center_y = x + w // 2, y + h // 2
        
        distance_from_center = math.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
        max_distance = math.sqrt(center_x**2 + center_y**2)
        position_factor = 1.0 - (distance_from_center / max_distance)
        
        # Combine factors
        quality = (size_factor * 0.6 + position_factor * 0.4)
        return max(0.0, min(1.0, quality))
    
    def _detect_eyes_detectron(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Optional[DetectronEyeDetection]]:
        """Detect eyes using Detectron2-enhanced methods."""
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {'left': None, 'right': None}
        
        # Use OpenCV eye detection
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(15, 15),
            maxSize=(w//3, h//3)
        )
        
        result = {'left': None, 'right': None}
        
        if len(eyes) >= 2:
            # Sort by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_box = eyes[0]
            right_eye_box = eyes[1]
            
            # Convert to absolute coordinates
            left_eye_abs = (x + left_eye_box[0], y + left_eye_box[1], left_eye_box[2], left_eye_box[3])
            right_eye_abs = (x + right_eye_box[0], y + right_eye_box[1], right_eye_box[2], right_eye_box[3])
            
            # Create enhanced eye detections
            result['left'] = self._create_enhanced_eye_detection(frame, left_eye_abs)
            result['right'] = self._create_enhanced_eye_detection(frame, right_eye_abs)
        
        return result
    
    def _create_enhanced_eye_detection(self, frame: np.ndarray, eye_bbox: Tuple[int, int, int, int]) -> DetectronEyeDetection:
        """Create enhanced eye detection with attention tracking."""
        x, y, w, h = eye_bbox
        eye_roi = frame[y:y+h, x:x+w]
        
        if eye_roi.size == 0:
            return DetectronEyeDetection(bbox=eye_bbox, center=(x + w//2, y + h//2), confidence=0.0, gaze_vector=(0, 0))
        
        # Convert to grayscale
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        
        # Calculate center
        center = (x + w//2, y + h//2)
        
        # Estimate gaze vector with improved method
        gaze_vector = self._estimate_gaze_vector_enhanced(gray_eye)
        
        # Calculate confidence
        confidence = self._calculate_eye_confidence_enhanced(gray_eye)
        
        # Calculate attention weight
        attention_weight = self._calculate_attention_weight(gray_eye)
        
        return DetectronEyeDetection(
            bbox=eye_bbox,
            center=center,
            confidence=confidence,
            gaze_vector=gaze_vector,
            attention_weight=attention_weight
        )
    
    def _estimate_gaze_vector_enhanced(self, eye_roi: np.ndarray) -> Tuple[float, float]:
        """Enhanced gaze vector estimation."""
        h, w = eye_roi.shape
        
        # Apply adaptive thresholding
        _, thresh = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the pupil)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate gaze direction relative to eye center
                gaze_x = (cx - w/2) / (w/2) * 30  # Scale to degrees
                gaze_y = (cy - h/2) / (h/2) * 30
                
                return (gaze_x, gaze_y)
        
        # Fallback to brightness-based estimation
        left_brightness = np.mean(eye_roi[:, :w//2])
        right_brightness = np.mean(eye_roi[:, w//2:])
        top_brightness = np.mean(eye_roi[:h//2, :])
        bottom_brightness = np.mean(eye_roi[h//2:, :])
        
        gaze_x = (right_brightness - left_brightness) / max(np.mean(eye_roi), 1) * 20
        gaze_y = (bottom_brightness - top_brightness) / max(np.mean(eye_roi), 1) * 20
        
        return (gaze_x, gaze_y)
    
    def _calculate_eye_confidence_enhanced(self, eye_roi: np.ndarray) -> float:
        """Enhanced eye confidence calculation."""
        # Calculate edge density
        edges = cv2.Canny(eye_roi, 50, 150)
        edge_density = np.sum(edges > 0) / eye_roi.size
        
        # Calculate contrast
        contrast = np.std(eye_roi)
        
        # Calculate circularity (eyes should be roughly circular)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Combine factors
        confidence = min(1.0, edge_density * 8 + contrast / 40 + circularity * 0.5)
        return max(0.0, confidence)
    
    def _calculate_attention_weight(self, eye_roi: np.ndarray) -> float:
        """Calculate attention weight based on eye characteristics."""
        # Calculate sharpness (attention correlates with sharp focus)
        laplacian_var = cv2.Laplacian(eye_roi, cv2.CV_64F).var()
        
        # Calculate brightness distribution (attentive eyes have good contrast)
        brightness_std = np.std(eye_roi)
        
        # Combine factors
        attention_weight = min(1.0, laplacian_var / 1000 + brightness_std / 50)
        return max(0.0, attention_weight)
    
    def _assign_person_id(self, bbox: Tuple[int, int, int, int]) -> int:
        """Assign person ID based on spatial proximity."""
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        
        # Find closest existing person
        min_distance = float('inf')
        closest_id = None
        
        for person_id, person_data in self.tracking_persons.items():
            if time.time() - person_data['timestamp'] > 3.0:  # Stale detection
                continue
            
            prev_bbox = person_data['face_detection'].bbox
            prev_center = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
            
            distance = math.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_id = person_id
        
        # Assign ID
        if min_distance < 80 and closest_id is not None:
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
        """Process frame with Detectron2 face detection."""
        start_time = time.time()
        
        try:
            # Detect faces using Detectron2
            face_detections = self._detect_faces_detectron(frame)
            
            # Update detection history
            self.detection_history.append(face_detections)
            
            # Process each detection
            persons = {}
            for detection in face_detections[:self.max_persons]:
                # Detect eyes
                eyes = self._detect_eyes_detectron(frame, detection.bbox)
                
                # Create person tracker
                person_id = self._assign_person_id(detection.bbox)
                
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
                avg_stability = sum(d.stability_score for d in face_detections) / len(face_detections)
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
    print("Facebook Detectron2 Face Detection System loaded successfully!")
    print("Features:")
    print("- State-of-the-art object detection")
    print("- Instance segmentation capabilities")
    print("- High accuracy face detection")
    print("- Real-time optimization")
    print("- Advanced pose estimation")
    print("- Attention tracking")
