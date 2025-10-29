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

class HeadPoseEstimator:
    """Head pose estimation using facial landmarks."""
    
    def __init__(self):
        # 3D model points for a generic face (in mm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-165.0, -170.0, -135.0),    # Left eye left corner
            (165.0, -170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (approximate)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
    def estimate_pose(self, face_bbox: Tuple[int, int, int, int], 
                     eyes: List[Tuple[int, int, int, int]], 
                     frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Estimate head pose from face bounding box and eye positions."""
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
        
        # Calculate 2D image points from face and eye positions
        image_points = self._calculate_image_points(x, y, w, h, eyes, frame_w, frame_h)
        
        if len(image_points) < 4:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        try:
            # Solve PnP to get rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points[:len(image_points)],
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not success:
                return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles (pitch, yaw, roll)
            pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Convert to degrees
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
            roll = math.degrees(roll)
            
            # Clamp values to reasonable ranges
            pitch = max(-90, min(90, pitch))
            yaw = max(-90, min(90, yaw))
            roll = max(-90, min(90, roll))
            
            return {'pitch': pitch, 'yaw': yaw, 'roll': roll}
            
        except Exception as e:
            logger.warning(f"Head pose estimation failed: {e}")
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
    
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

class EyeGazeEstimator:
    """Eye gaze estimation from eye positions."""
    
    def __init__(self):
        self.gaze_history = deque(maxlen=5)
        
    def estimate_gaze(self, eyes: List[Tuple[int, int, int, int]], 
                     face_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Estimate gaze direction from eye positions."""
        if len(eyes) < 2:
            return {'gaze_x': 0.0, 'gaze_y': 0.0, 'gaze_confidence': 0.0}
        
        # Sort eyes by x position
        sorted_eyes = sorted(eyes, key=lambda eye: eye[0])
        left_eye = sorted_eyes[0]
        right_eye = sorted_eyes[1]
        
        # Calculate eye centers
        left_center_x = left_eye[0] + left_eye[2] // 2
        left_center_y = left_eye[1] + left_eye[3] // 2
        right_center_x = right_eye[0] + right_eye[2] // 2
        right_center_y = right_eye[1] + right_eye[3] // 2
        
        # Calculate inter-eye distance
        inter_eye_distance = math.sqrt((right_center_x - left_center_x)**2 + 
                                      (right_center_y - left_center_y)**2)
        
        # Calculate gaze direction relative to face center
        face_center_x = face_bbox[0] + face_bbox[2] // 2
        face_center_y = face_bbox[1] + face_bbox[3] // 2
        
        # Average eye center
        avg_eye_x = (left_center_x + right_center_x) // 2
        avg_eye_y = (left_center_y + right_center_y) // 2
        
        # Calculate gaze offset
        gaze_x = (avg_eye_x - face_center_x) / (face_bbox[2] / 2)
        gaze_y = (avg_eye_y - face_center_y) / (face_bbox[3] / 2)
        
        # Clamp gaze values
        gaze_x = max(-1.0, min(1.0, gaze_x))
        gaze_y = max(-1.0, min(1.0, gaze_y))
        
        # Calculate confidence based on eye detection quality
        confidence = min(1.0, inter_eye_distance / 50.0)  # Normalize by expected eye distance
        
        # Apply temporal smoothing
        gaze_data = {'gaze_x': gaze_x, 'gaze_y': gaze_y, 'gaze_confidence': confidence}
        self.gaze_history.append(gaze_data)
        
        # Return smoothed values
        if len(self.gaze_history) > 1:
            avg_gaze_x = sum(d['gaze_x'] for d in self.gaze_history) / len(self.gaze_history)
            avg_gaze_y = sum(d['gaze_y'] for d in self.gaze_history) / len(self.gaze_history)
            avg_confidence = sum(d['gaze_confidence'] for d in self.gaze_history) / len(self.gaze_history)
            
            return {'gaze_x': avg_gaze_x, 'gaze_y': avg_gaze_y, 'gaze_confidence': avg_confidence}
        
        return gaze_data

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
        
        # Initialize pose and gaze estimators
        self.pose_estimator = HeadPoseEstimator()
        self.gaze_estimator = EyeGazeEstimator()
        
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
            
            # Estimate gaze direction
            gaze = self.gaze_estimator.estimate_gaze(
                eyes=eye_boxes,
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
            
            # Pose values with enhanced styling
            pitch = detection.pose['pitch']
            roll = detection.pose['roll']
            yaw = detection.pose['yaw']
            
            # Gaze values
            gaze_x = detection.gaze['gaze_x']
            gaze_y = detection.gaze['gaze_y']
            gaze_conf = detection.gaze['gaze_confidence']
            
            # Pose header
            self.draw_gradient_text(panel, "HEAD POSE", (25, card_y + 95), 
                                  0.5, 1, self.colors['accent'])
            
            # Pose values in a grid
            self.draw_gradient_text(panel, f"Pitch: {pitch:.1f}°", (25, card_y + 115), 
                                  0.4, 1, self.colors['text'])
            self.draw_gradient_text(panel, f"Roll: {roll:.1f}°", (25, card_y + 130), 
                                  0.4, 1, self.colors['text'])
            self.draw_gradient_text(panel, f"Yaw: {yaw:.1f}°", (25, card_y + 145), 
                                  0.4, 1, self.colors['text'])
            
            # Gaze header
            self.draw_gradient_text(panel, "GAZE DIRECTION", (25, card_y + 165), 
                                  0.5, 1, self.colors['accent'])
            
            # Gaze values
            self.draw_gradient_text(panel, f"X: {gaze_x:.2f}", (25, card_y + 185), 
                                  0.4, 1, self.colors['text'])
            self.draw_gradient_text(panel, f"Y: {gaze_y:.2f}", (25, card_y + 200), 
                                  0.4, 1, self.colors['text'])
            self.draw_gradient_text(panel, f"Conf: {gaze_conf:.2f}", (25, card_y + 215), 
                                  0.4, 1, self.colors['text'])
        
        # Draw enhanced face bounding boxes on main frame
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw face box with gradient effect
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face_box'], 3)
            
            # Draw corner markers
            corner_size = 15
            cv2.line(frame, (x, y), (x + corner_size, y), self.colors['face_box'], 3)
            cv2.line(frame, (x, y), (x, y + corner_size), self.colors['face_box'], 3)
            cv2.line(frame, (x + w, y), (x + w - corner_size, y), self.colors['face_box'], 3)
            cv2.line(frame, (x + w, y), (x + w, y + corner_size), self.colors['face_box'], 3)
            cv2.line(frame, (x, y + h), (x + corner_size, y + h), self.colors['face_box'], 3)
            cv2.line(frame, (x, y + h), (x, y + h - corner_size), self.colors['face_box'], 3)
            cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), self.colors['face_box'], 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), self.colors['face_box'], 3)
            
            # Draw eyes with enhanced styling
            for ex, ey, ew, eh in detection.eyes:
                cv2.circle(frame, (ex + ew//2, ey + eh//2), max(ew, eh)//2, 
                          self.colors['eye_box'], 2)
            
            # Draw gaze direction arrow
            gaze_x = detection.gaze['gaze_x']
            gaze_y = detection.gaze['gaze_y']
            gaze_conf = detection.gaze['gaze_confidence']
            
            if gaze_conf > 0.3:  # Only draw if confidence is reasonable
                # Calculate arrow start and end points
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Scale gaze vector
                arrow_length = min(w, h) // 3
                end_x = int(face_center_x + gaze_x * arrow_length)
                end_y = int(face_center_y + gaze_y * arrow_length)
                
                # Draw gaze arrow
                cv2.arrowedLine(frame, (face_center_x, face_center_y), 
                              (end_x, end_y), self.colors['eye_box'], 3, tipLength=0.3)
                
                # Draw gaze confidence circle
                cv2.circle(frame, (face_center_x, face_center_y), 
                          int(gaze_conf * 20), self.colors['eye_box'], 1)
        
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
