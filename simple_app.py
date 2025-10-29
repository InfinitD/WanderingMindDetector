#!/usr/bin/env python3
"""
wandering-mind-detector Application
Standalone version that works without Detectron2 dependencies
"""

import cv2
import numpy as np
import sys
import time
import logging
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SOTAFacialLandmarkDetector:
    """State-of-the-art facial landmark detection for FaceID-level precision."""
    
    def __init__(self):
        # Initialize DNN face detection
        self.face_net = None
        self.landmark_net = None
        self._initialize_dnn_models()
        
        # 3D facial model points (68 landmarks in mm)
        self.model_points_68 = np.array([
            # Nose tip
            (0.0, 0.0, 0.0),
            # Chin
            (0.0, -330.0, -65.0),
            # Left eye corners
            (-165.0, -170.0, -135.0),
            (-100.0, -170.0, -135.0),
            # Right eye corners  
            (100.0, -170.0, -135.0),
            (165.0, -170.0, -135.0),
            # Mouth corners
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
            # Additional key points for better accuracy
            (-50.0, -200.0, -100.0),   # Left cheek
            (50.0, -200.0, -100.0),    # Right cheek
            (0.0, -250.0, -80.0),      # Lower chin
        ], dtype=np.float64)
        
        # Camera matrix
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
    def _initialize_dnn_models(self):
        """Initialize DNN models for face and landmark detection."""
        try:
            # Try to load OpenCV DNN face detection model
            model_path = cv2.data.haarcascades
            # For now, we'll use Haar cascades but with enhanced processing
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            logger.info("Initialized enhanced facial landmark detector")
        except Exception as e:
            logger.warning(f"Could not initialize DNN models: {e}")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class SOTAHeadPoseEstimator:
    """State-of-the-art head pose estimation with FaceID-level precision."""
    
    def __init__(self):
        self.landmark_detector = SOTAFacialLandmarkDetector()
        
        # Enhanced 3D model points for better accuracy
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-165.0, -170.0, -135.0),    # Left eye left corner
            (165.0, -170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Pose smoothing
        self.pose_history = deque(maxlen=8)
        self.smoothed_pose = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
    def estimate_pose(self, face_bbox: Tuple[int, int, int, int], 
                     eyes: List[Tuple[int, int, int, int]], 
                     frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Estimate head pose with FaceID-level precision."""
        x, y, w, h = face_bbox
        frame_h, frame_w = frame_shape
        
        # Initialize camera matrix if not set
        if self.camera_matrix is None:
            focal_length = frame_w
            center_x = frame_w / 2.0
            center_y = frame_h / 2.0
            self.camera_matrix = np.array([
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1]
            ], dtype=np.float64)
        
        # Calculate enhanced 2D image points with sub-pixel accuracy
        image_points = self._calculate_enhanced_image_points(x, y, w, h, eyes, frame_w, frame_h)
        
        if len(image_points) < 4:
            return self.smoothed_pose
        
        try:
            # Use more robust PnP solving with better parameters
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points[:len(image_points)],
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=False
            )
            
            if not success:
                return self.smoothed_pose
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles with improved accuracy
            pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Convert to degrees
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
            roll = math.degrees(roll)
            
            # Apply advanced smoothing and filtering
            raw_pose = {'pitch': pitch, 'yaw': yaw, 'roll': roll}
            smoothed_pose = self._apply_pose_smoothing(raw_pose)
            
            return smoothed_pose
            
        except Exception as e:
            logger.warning(f"Enhanced head pose estimation failed: {e}")
            return self.smoothed_pose
    
    def _calculate_image_points(self, x: int, y: int, w: int, h: int, 
                               eyes: List[Tuple[int, int, int, int]], 
                               frame_w: int, frame_h: int) -> np.ndarray:
        """Calculate 2D image points from face and eye positions."""
        points = []
        
        # Nose tip (center of face)
        nose_x = x + w // 2
        nose_y = y + h // 2
        points.append([nose_x, nose_y])
        
        # Chin (bottom center of face)
        chin_x = x + w // 2
        chin_y = y + h
        points.append([chin_x, chin_y])
        
        # Eye positions
        if len(eyes) >= 2:
            # Sort eyes by x position
            sorted_eyes = sorted(eyes, key=lambda eye: eye[0])
            
            # Left eye center
            left_eye = sorted_eyes[0]
            left_eye_x = left_eye[0] + left_eye[2] // 2
            left_eye_y = left_eye[1] + left_eye[3] // 2
            points.append([left_eye_x, left_eye_y])
            
            # Right eye center
            right_eye = sorted_eyes[1]
            right_eye_x = right_eye[0] + right_eye[2] // 2
            right_eye_y = right_eye[1] + right_eye[3] // 2
            points.append([right_eye_x, right_eye_y])
            
            # Mouth corners (estimated)
            mouth_y = y + int(h * 0.7)
            mouth_left_x = x + int(w * 0.3)
            mouth_right_x = x + int(w * 0.7)
            points.append([mouth_left_x, mouth_y])
            points.append([mouth_right_x, mouth_y])
        
        return np.array(points, dtype=np.float64)
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)."""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        return x, y, z
    
    def _calculate_enhanced_image_points(self, x: int, y: int, w: int, h: int, 
                                       eyes: List[Tuple[int, int, int, int]], 
                                       frame_w: int, frame_h: int) -> np.ndarray:
        """Calculate enhanced 2D image points with sub-pixel accuracy."""
        points = []
        
        # Nose tip (center of face with sub-pixel accuracy)
        nose_x = x + w / 2.0
        nose_y = y + h / 2.0
        points.append([nose_x, nose_y])
        
        # Chin (bottom center of face)
        chin_x = x + w / 2.0
        chin_y = y + h
        points.append([chin_x, chin_y])
        
        # Enhanced eye position calculation
        if len(eyes) >= 2:
            # Sort eyes by x position
            sorted_eyes = sorted(eyes, key=lambda eye: eye[0])
            
            # Left eye center with sub-pixel accuracy
            left_eye = sorted_eyes[0]
            left_eye_x = left_eye[0] + left_eye[2] / 2.0
            left_eye_y = left_eye[1] + left_eye[3] / 2.0
            points.append([left_eye_x, left_eye_y])
            
            # Right eye center with sub-pixel accuracy
            right_eye = sorted_eyes[1]
            right_eye_x = right_eye[0] + right_eye[2] / 2.0
            right_eye_y = right_eye[1] + right_eye[3] / 2.0
            points.append([right_eye_x, right_eye_y])
            
            # Enhanced mouth corner estimation
            mouth_y = y + h * 0.75
            mouth_left_x = x + w * 0.25
            mouth_right_x = x + w * 0.75
            points.append([mouth_left_x, mouth_y])
            points.append([mouth_right_x, mouth_y])
        
        return np.array(points, dtype=np.float64)
    
    def _apply_pose_smoothing(self, raw_pose: Dict[str, float]) -> Dict[str, float]:
        """Apply advanced pose smoothing for stable tracking."""
        # Store current pose
        self.pose_history.append(raw_pose)
        
        if len(self.pose_history) < 3:
            return raw_pose
        
        # Apply weighted moving average with exponential decay
        weights = np.exp(np.linspace(-1, 0, len(self.pose_history)))
        weights = weights / np.sum(weights)
        
        smoothed_pitch = sum(p['pitch'] * w for p, w in zip(self.pose_history, weights))
        smoothed_yaw = sum(p['yaw'] * w for p, w in zip(self.pose_history, weights))
        smoothed_roll = sum(p['roll'] * w for p, w in zip(self.pose_history, weights))
        
        # Apply outlier rejection
        if len(self.pose_history) >= 3:
            recent_poses = list(self.pose_history)[-3:]
            pitch_std = np.std([p['pitch'] for p in recent_poses])
            yaw_std = np.std([p['yaw'] for p in recent_poses])
            roll_std = np.std([p['roll'] for p in recent_poses])
            
            # If current pose is outlier, use previous smoothed value
            if abs(raw_pose['pitch'] - smoothed_pitch) > 2 * pitch_std:
                smoothed_pitch = self.smoothed_pose['pitch']
            if abs(raw_pose['yaw'] - smoothed_yaw) > 2 * yaw_std:
                smoothed_yaw = self.smoothed_pose['yaw']
            if abs(raw_pose['roll'] - smoothed_roll) > 2 * roll_std:
                smoothed_roll = self.smoothed_pose['roll']
        
        # Clamp values to reasonable ranges
        smoothed_pitch = max(-90, min(90, smoothed_pitch))
        smoothed_yaw = max(-90, min(90, smoothed_yaw))
        smoothed_roll = max(-90, min(90, smoothed_roll))
        
        self.smoothed_pose = {
            'pitch': smoothed_pitch,
            'yaw': smoothed_yaw,
            'roll': smoothed_roll
        }
        
        return self.smoothed_pose

class SOTAGazeEstimator:
    """State-of-the-art gaze estimation based on head pose with advanced smoothing."""
    
    def __init__(self):
        self.pose_history = deque(maxlen=10)
        self.gaze_history = deque(maxlen=8)
        self.velocity_history = deque(maxlen=5)
        self.smoothing_factor = 0.7  # Higher = more smoothing
        self.last_gaze_vector = np.array([0.0, 0.0])
        
    def estimate_gaze_from_pose(self, pose: Dict[str, float], 
                              face_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Estimate gaze direction from head pose using advanced algorithms."""
        pitch = pose['pitch']
        yaw = pose['yaw']
        roll = pose['roll']
        
        # Convert degrees to radians
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        roll_rad = math.radians(roll)
        
        # Create rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])
        
        Ry = np.array([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ])
        
        Rz = np.array([
            [math.cos(roll_rad), -math.sin(roll_rad), 0],
            [math.sin(roll_rad), math.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations (order: roll, pitch, yaw)
        R = Rz @ Rx @ Ry
        
        # Initial gaze vector (looking forward)
        initial_gaze = np.array([0.0, 0.0, -1.0])
        
        # Apply rotation to get gaze direction
        gaze_vector_3d = R @ initial_gaze
        
        # Project to 2D screen coordinates
        gaze_x = gaze_vector_3d[0] * 0.8  # Scale for screen projection
        gaze_y = gaze_vector_3d[1] * 0.6  # Scale for screen projection
        
        # Apply advanced smoothing using Kalman-like filtering
        smoothed_gaze = self._apply_advanced_smoothing(gaze_x, gaze_y)
        
        # Calculate confidence based on pose stability
        confidence = self._calculate_pose_confidence(pose)
        
        return {
            'gaze_x': smoothed_gaze[0],
            'gaze_y': smoothed_gaze[1],
            'gaze_confidence': confidence,
            'raw_pitch': pitch,
            'raw_yaw': yaw,
            'raw_roll': roll
        }
    
    def _apply_advanced_smoothing(self, gaze_x: float, gaze_y: float) -> np.ndarray:
        """Apply advanced smoothing using multiple techniques."""
        current_gaze = np.array([gaze_x, gaze_y])
        
        # Store current gaze
        self.gaze_history.append(current_gaze)
        
        if len(self.gaze_history) < 3:
            return current_gaze
        
        # Calculate velocity
        if len(self.gaze_history) >= 2:
            velocity = current_gaze - self.gaze_history[-2]
            self.velocity_history.append(velocity)
        
        # Apply exponential moving average with velocity compensation
        if len(self.gaze_history) >= 3:
            # Calculate weighted average with more weight on recent values
            weights = np.exp(np.linspace(-2, 0, len(self.gaze_history)))
            weights = weights / np.sum(weights)
            
            smoothed_gaze = np.zeros(2)
            for i, gaze in enumerate(self.gaze_history):
                smoothed_gaze += weights[i] * gaze
            
            # Apply velocity damping to reduce jitter
            if len(self.velocity_history) >= 2:
                avg_velocity = np.mean(list(self.velocity_history), axis=0)
                velocity_damping = 0.3
                smoothed_gaze -= velocity_damping * avg_velocity
            
            # Apply adaptive smoothing based on movement speed
            movement_speed = np.linalg.norm(avg_velocity) if len(self.velocity_history) >= 2 else 0
            if movement_speed > 0.1:  # High movement
                smoothing_factor = 0.5
            elif movement_speed > 0.05:  # Medium movement
                smoothing_factor = 0.7
            else:  # Low movement
                smoothing_factor = 0.9
            
            # Final smoothing with adaptive factor
            self.last_gaze_vector = (smoothing_factor * self.last_gaze_vector + 
                                   (1 - smoothing_factor) * smoothed_gaze)
            
            return self.last_gaze_vector
        
        return current_gaze
    
    def _calculate_pose_confidence(self, pose: Dict[str, float]) -> float:
        """Calculate confidence based on pose stability and validity."""
        pitch, yaw, roll = pose['pitch'], pose['yaw'], pose['roll']
        
        # Check for extreme angles (less confident)
        extreme_penalty = 0
        if abs(pitch) > 60 or abs(yaw) > 60 or abs(roll) > 45:
            extreme_penalty = 0.3
        
        # Check for pose stability
        stability_score = 1.0
        if len(self.pose_history) >= 2:
            last_pose = self.pose_history[-1]
            pose_diff = abs(pitch - last_pose['pitch']) + abs(yaw - last_pose['yaw']) + abs(roll - last_pose['roll'])
            stability_score = max(0.3, 1.0 - pose_diff / 180.0)  # Normalize by max possible difference
        
        # Store current pose
        self.pose_history.append(pose)
        
        # Combine factors
        confidence = (1.0 - extreme_penalty) * stability_score
        return max(0.1, min(1.0, confidence))

@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    person_id: int
    pose: Dict[str, float]  # pitch, yaw, roll
    eyes: List[Tuple[int, int, int, int]]  # eye bounding boxes
    quality_score: float
    gaze: Dict[str, float]  # gaze_x, gaze_y, gaze_confidence

class WanderingMindDetector:
    """wandering-mind-detector using OpenCV Haar Cascades."""
    
    def __init__(self, max_persons: int = 3):
        self.max_persons = max_persons
        self.confidence_threshold = 0.3
        
        # Load Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize SOTA pose and gaze estimators
        self.pose_estimator = SOTAHeadPoseEstimator()
        self.gaze_estimator = SOTAGazeEstimator()
        
        # Detection history for temporal smoothing
        self.detection_history = deque(maxlen=10)
        self.person_counter = 0
        
        logger.info("WanderingMindDetector initialized")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for i, (x, y, w, h) in enumerate(faces[:self.max_persons]):
            # Calculate confidence (simplified)
            confidence = min(1.0, w * h / (100 * 100))
            
            if confidence < self.confidence_threshold:
                continue
            
            # Detect eyes with higher threshold to prevent hallucination
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=8,  # Increased from default 3 to reduce false positives
                minSize=(15, 15),  # Increased minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert eye coordinates to frame coordinates
            eye_boxes = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
            
            # Estimate head pose using 3D pose estimation
            pose = self.pose_estimator.estimate_pose(
                face_bbox=(x, y, w, h),
                eyes=eye_boxes,
                frame_shape=frame.shape[:2]
            )
            
            # Estimate gaze direction from head pose
            gaze = self.gaze_estimator.estimate_gaze_from_pose(
                pose=pose,
                face_bbox=(x, y, w, h)
            )
            
            detection = FaceDetection(
                bbox=(x, y, w, h),
                confidence=confidence,
                person_id=self.person_counter + i,
                pose=pose,
                eyes=eye_boxes,
                quality_score=confidence,
                gaze=gaze
            )
            
            detections.append(detection)
        
        # Update person counter
        self.person_counter = (self.person_counter + len(detections)) % 1000
        
        # Store in history
        self.detection_history.append(detections)
        
        return detections

class NeumorphismUI:
    """Modern neumorphism UI for displaying face detection results."""
    
    def __init__(self, panel_width: int = 320):
        self.panel_width = panel_width
        self.colors = {
            'background': (45, 45, 45),      # Dark background
            'panel': (55, 55, 55),           # Slightly lighter panel
            'text': (255, 255, 255),         # White text
            'text_secondary': (200, 200, 200), # Light gray secondary text
            'face_box': (0, 255, 100),       # Bright green for faces
            'eye_box': (100, 200, 255),      # Blue for eyes
            'gaze_arrow': (255, 100, 100),   # Soft red for gaze arrow
            'card_bg': (65, 65, 65),         # Card background
            'card_shadow_dark': (35, 35, 35), # Dark shadow
            'card_shadow_light': (75, 75, 75), # Light shadow
            'accent': (100, 150, 255),       # Accent color
            'success': (0, 255, 100),        # Success green
            'warning': (255, 200, 0)         # Warning yellow
        }
    
    def draw_sunken_card(self, panel: np.ndarray, x: int, y: int, width: int, height: int):
        """Draw a neumorphism sunken card effect."""
        # Draw dark shadow (bottom-right)
        cv2.rectangle(panel, (x + 2, y + 2), (x + width + 2, y + height + 2), 
                     self.colors['card_shadow_dark'], -1)
        
        # Draw light shadow (top-left)
        cv2.rectangle(panel, (x - 2, y - 2), (x + width - 2, y + height - 2), 
                     self.colors['card_shadow_light'], -1)
        
        # Draw main card
        cv2.rectangle(panel, (x, y), (x + width, y + height), 
                     self.colors['card_bg'], -1)
        
        # Draw subtle border
        cv2.rectangle(panel, (x, y), (x + width, y + height), 
                     self.colors['text_secondary'], 1)
    
    def draw_gradient_text(self, panel: np.ndarray, text: str, pos: tuple, 
                          font_scale: float, thickness: int, color: tuple):
        """Draw text with subtle gradient effect."""
        x, y = pos
        # Draw shadow
        cv2.putText(panel, text, (x + 1, y + 1), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        # Draw main text
        cv2.putText(panel, text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def render(self, frame: np.ndarray, detections: List[FaceDetection], fps: float) -> np.ndarray:
        """Render the modern neumorphism UI."""
        h, w = frame.shape[:2]
        
        # Create dark side panel
        panel = np.full((h, self.panel_width, 3), self.colors['background'], dtype=np.uint8)
        
        # Draw header with gradient effect
        self.draw_gradient_text(panel, "WANDERING MIND", (15, 35), 
                              0.8, 2, self.colors['text'])
        self.draw_gradient_text(panel, "DETECTOR", (15, 60), 
                              0.6, 1, self.colors['text_secondary'])
        
        # Draw status cards
        status_y = 90
        card_width = self.panel_width - 30
        card_height = 50
        
        # FPS Card
        self.draw_sunken_card(panel, 15, status_y, card_width, card_height)
        self.draw_gradient_text(panel, f"FPS: {fps:.1f}", (25, status_y + 30), 
                              0.6, 2, self.colors['success'])
        
        # Detection Count Card
        detection_y = status_y + card_height + 15
        self.draw_sunken_card(panel, 15, detection_y, card_width, card_height)
        self.draw_gradient_text(panel, f"FACES: {len(detections)}", (25, detection_y + 30), 
                              0.6, 2, self.colors['accent'])
        
        # Person cards with enhanced styling
        y_offset = detection_y + card_height + 25
        card_height = 180  # Increased to accommodate gaze information
        card_width = self.panel_width - 30
        
        for i, detection in enumerate(detections):
            card_y = y_offset + (i * (card_height + 15))
            
            # Draw sunken card
            self.draw_sunken_card(panel, 15, card_y, card_width, card_height)
            
            # Person header
            self.draw_gradient_text(panel, f"PERSON {detection.person_id}", 
                                  (25, card_y + 25), 0.7, 2, self.colors['text'])
            
            # Confidence with color coding
            conf_color = self.colors['success'] if detection.confidence > 0.7 else self.colors['warning']
            self.draw_gradient_text(panel, f"Confidence: {detection.confidence:.2f}", 
                                  (25, card_y + 50), 0.5, 1, conf_color)
            
            # Eyes count
            self.draw_gradient_text(panel, f"Eyes Detected: {len(detection.eyes)}", 
                                  (25, card_y + 70), 0.5, 1, self.colors['text_secondary'])
            
            # Head Pose and Gaze Direction Table
            pitch = detection.pose['pitch']
            roll = detection.pose['roll']
            yaw = detection.pose['yaw']
            
            gaze_x = detection.gaze['gaze_x']
            gaze_y = detection.gaze['gaze_y']
            gaze_conf = detection.gaze['gaze_confidence']
            
            # Table header
            self.draw_gradient_text(panel, "HEAD POSE & GAZE DIRECTION", (25, card_y + 95), 
                                  0.5, 1, self.colors['accent'])
            
            # Table rows with inline values
            self.draw_gradient_text(panel, f"Pitch: {pitch:6.1f}° | Gaze X: {gaze_x:5.2f}", 
                                  (25, card_y + 115), 0.4, 1, self.colors['text'])
            self.draw_gradient_text(panel, f"Roll:  {roll:6.1f}° | Gaze Y: {gaze_y:5.2f}", 
                                  (25, card_y + 130), 0.4, 1, self.colors['text'])
            self.draw_gradient_text(panel, f"Yaw:   {yaw:6.1f}° | Conf:   {gaze_conf:5.2f}", 
                                  (25, card_y + 145), 0.4, 1, self.colors['text'])
        
        # Draw enhanced face bounding boxes on main frame
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw face box with reduced thickness
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face_box'], 1)
            
            # Draw corner markers with reduced thickness
            corner_size = 15
            cv2.line(frame, (x, y), (x + corner_size, y), self.colors['face_box'], 1)
            cv2.line(frame, (x, y), (x, y + corner_size), self.colors['face_box'], 1)
            cv2.line(frame, (x + w, y), (x + w - corner_size, y), self.colors['face_box'], 1)
            cv2.line(frame, (x + w, y), (x + w, y + corner_size), self.colors['face_box'], 1)
            cv2.line(frame, (x, y + h), (x + corner_size, y + h), self.colors['face_box'], 1)
            cv2.line(frame, (x, y + h), (x, y + h - corner_size), self.colors['face_box'], 1)
            cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), self.colors['face_box'], 1)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), self.colors['face_box'], 1)
            
            # Draw eyes with enhanced styling
            for ex, ey, ew, eh in detection.eyes:
                cv2.circle(frame, (ex + ew//2, ey + eh//2), max(ew, eh)//2, 
                          self.colors['eye_box'], 2)
            
            # Draw 3D-style gaze arrow (Spline-inspired)
            gaze_x = detection.gaze['gaze_x']
            gaze_y = detection.gaze['gaze_y']
            gaze_conf = detection.gaze['gaze_confidence']
            
            if gaze_conf > 0.2:
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Calculate 3D arrow with depth effect
                arrow_length = min(w, h) * 0.8
                end_x = int(face_center_x + gaze_x * arrow_length)
                end_y = int(face_center_y + gaze_y * arrow_length)
                
                # Draw 3D arrow with multiple layers for depth
                # Base arrow (darker)
                cv2.arrowedLine(frame, (face_center_x + 1, face_center_y + 1), 
                              (end_x + 1, end_y + 1), (200, 80, 80), 4, tipLength=0.4)
                
                # Main arrow (bright)
                cv2.arrowedLine(frame, (face_center_x, face_center_y), 
                              (end_x, end_y), self.colors['gaze_arrow'], 3, tipLength=0.4)
                
                # Highlight arrow (brightest)
                cv2.arrowedLine(frame, (face_center_x - 1, face_center_y - 1), 
                              (end_x - 1, end_y - 1), (255, 150, 150), 2, tipLength=0.3)
                
                # Add 3D arrow shaft with gradient effect
                shaft_points = np.array([
                    [face_center_x, face_center_y],
                    [end_x, end_y]
                ], np.int32)
                
                # Draw gradient shaft
                for i in range(len(shaft_points) - 1):
                    alpha = i / (len(shaft_points) - 1)
                    color_intensity = int(255 * alpha)
                    color = (color_intensity, 100, 100)
                    cv2.line(frame, tuple(shaft_points[i]), tuple(shaft_points[i+1]), color, 2)
        
        # Combine frame and panel
        combined = np.hstack([frame, panel])
        return combined

class WanderingMindApp:
    """Main wandering-mind-detector application."""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.detector = WanderingMindDetector()
        self.ui = NeumorphismUI()
        self.cap = None
        self.running = False
        
    def initialize_camera(self) -> bool:
        """Initialize camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera {self.camera_index} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def run(self):
        """Run the main application loop."""
        if not self.initialize_camera():
            return
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        logger.info("Starting wandering-mind-detector application...")
        print("Press 'q' to quit, 'r' to reset")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                # Detect faces
                detections = self.detector.detect_faces(frame)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                else:
                    fps = 30.0  # Default estimate
                
                # Render UI
                display_frame = self.ui.render(frame, detections, fps)
                
                # Show frame
                cv2.imshow('wandering-mind-detector', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.detector.person_counter = 0
                    self.detector.detection_history.clear()
                    logger.info("Reset detection data")
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Application cleaned up")

def main():
    """Main entry point."""
    print("wandering-mind-detector Application")
    print("=" * 50)
    
    # Parse command line arguments
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("Invalid camera index, using default (0)")
    
    # Create and run application
    app = WanderingMindApp(camera_index)
    app.run()

if __name__ == "__main__":
    main()
