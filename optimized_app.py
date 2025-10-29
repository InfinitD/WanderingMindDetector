#!/usr/bin/env python3
"""
wandering-mind-detector - Optimized Lightweight Version
High-performance face detection and head pose estimation
"""

import cv2
import numpy as np
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque

# Optimized logging configuration
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Optimized detection result."""
    bbox: Tuple[int, int, int, int]
    pose: Tuple[float, float, float]  # pitch, yaw, roll
    confidence: float
    person_id: int

class OptimizedDetector:
    """Lightweight, high-performance face and pose detector."""
    
    def __init__(self, max_persons: int = 3):
        # Pre-load cascades once
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Optimized 3D model points (minimal set for accuracy)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-165.0, -170.0, -135.0),    # Left eye
            (165.0, -170.0, -135.0),     # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ], dtype=np.float32)  # Use float32 for better performance
        
        # Pre-allocate arrays
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Efficient smoothing
        self.pose_history = deque(maxlen=5)  # Reduced buffer size
        self.person_counter = 0
        self.max_persons = max_persons
        
        # Pre-computed constants
        self.eye_detection_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 8,
            'minSize': (15, 15),
            'flags': cv2.CASCADE_SCALE_IMAGE
        }
        
        self.face_detection_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 5,
            'minSize': (30, 30),
            'flags': cv2.CASCADE_SCALE_IMAGE
        }
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Optimized detection pipeline."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, **self.face_detection_params)
        
        detections = []
        for i, (x, y, w, h) in enumerate(faces[:self.max_persons]):
            # Quick confidence check
            confidence = min(1.0, w * h / 10000.0)
            if confidence < 0.3:
                continue
            
            # Detect eyes in ROI
            roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi, **self.eye_detection_params)
            
            # Convert to frame coordinates
            eye_points = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
            
            # Estimate pose
            pose = self._estimate_pose_fast((x, y, w, h), eye_points, frame.shape)
            
            detections.append(Detection(
                bbox=(x, y, w, h),
                pose=pose,
                confidence=confidence,
                person_id=self.person_counter + i
            ))
        
        self.person_counter = (self.person_counter + len(detections)) % 1000
        return detections
    
    def _estimate_pose_fast(self, bbox: Tuple[int, int, int, int], 
                          eyes: List[Tuple[int, int, int, int]], 
                          frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """Fast pose estimation with minimal computation."""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape
        
        # Initialize camera matrix once
        if self.camera_matrix is None:
            focal_length = max(frame_w, frame_h) * 0.8
            self.camera_matrix = np.array([
                [focal_length, 0, frame_w/2],
                [0, focal_length, frame_h/2],
                [0, 0, 1]
            ], dtype=np.float32)
        
        # Calculate image points efficiently
        image_points = self._get_image_points_fast(x, y, w, h, eyes)
        
        if len(image_points) < 4:
            return (0.0, 0.0, 0.0)
        
        try:
            # Use iterative PnP for speed
            success, rvec, tvec = cv2.solvePnP(
                self.model_points[:len(image_points)],
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return (0.0, 0.0, 0.0)
            
            # Convert to Euler angles efficiently
            R, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = self._rotation_to_euler_fast(R)
            
            # Apply smoothing
            pose = (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))
            return self._smooth_pose(pose)
            
        except:
            return (0.0, 0.0, 0.0)
    
    def _get_image_points_fast(self, x: int, y: int, w: int, h: int, 
                             eyes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Fast image point calculation."""
        points = [
            [x + w/2, y + h/2],      # Nose tip
            [x + w/2, y + h],        # Chin
        ]
        
        if len(eyes) >= 2:
            # Sort eyes by x position
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            
            # Eye centers
            left_eye = sorted_eyes[0]
            right_eye = sorted_eyes[1]
            
            points.extend([
                [left_eye[0] + left_eye[2]/2, left_eye[1] + left_eye[3]/2],
                [right_eye[0] + right_eye[2]/2, right_eye[1] + right_eye[3]/2],
                [x + w*0.25, y + h*0.75],  # Left mouth
                [x + w*0.75, y + h*0.75]   # Right mouth
            ])
        
        return np.array(points, dtype=np.float32)
    
    def _rotation_to_euler_fast(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Fast Euler angle extraction."""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        
        return pitch, yaw, roll
    
    def _smooth_pose(self, pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Efficient pose smoothing."""
        self.pose_history.append(pose)
        
        if len(self.pose_history) < 3:
            return pose
        
        # Simple moving average
        recent = list(self.pose_history)[-3:]
        smoothed = (
            sum(p[0] for p in recent) / len(recent),
            sum(p[1] for p in recent) / len(recent),
            sum(p[2] for p in recent) / len(recent)
        )
        
        # Clamp values
        return (
            max(-90, min(90, smoothed[0])),
            max(-90, min(90, smoothed[1])),
            max(-90, min(90, smoothed[2]))
        )

class OptimizedUI:
    """Lightweight, efficient UI renderer."""
    
    def __init__(self, panel_width: int = 300):
        self.panel_width = panel_width
        
        # Pre-define colors (BGR format)
        self.colors = {
            'bg': (45, 45, 45),
            'text': (255, 255, 255),
            'accent': (100, 150, 255),
            'face': (0, 255, 100),
            'eye': (100, 200, 255),
            'arrow': (100, 100, 255)
        }
    
    def render(self, frame: np.ndarray, detections: List[Detection], fps: float) -> np.ndarray:
        """Optimized rendering pipeline."""
        h, w = frame.shape[:2]
        
        # Create panel efficiently
        panel = np.full((h, self.panel_width, 3), self.colors['bg'], dtype=np.uint8)
        
        # Draw header
        cv2.putText(panel, "WANDERING MIND DETECTOR", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Draw stats
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        cv2.putText(panel, f"Faces: {len(detections)}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        
        # Draw detection cards
        y_offset = 110
        for i, det in enumerate(detections):
            card_y = y_offset + i * 80
            
            # Simple card background
            cv2.rectangle(panel, (10, card_y), (self.panel_width - 10, card_y + 70), 
                         (65, 65, 65), -1)
            cv2.rectangle(panel, (10, card_y), (self.panel_width - 10, card_y + 70), 
                         self.colors['text'], 1)
            
            # Person info
            cv2.putText(panel, f"Person {det.person_id}", (20, card_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            # Pose values in compact format
            pitch, yaw, roll = det.pose
            cv2.putText(panel, f"Pitch: {pitch:6.1f}° | Yaw: {yaw:6.1f}°", 
                       (20, card_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            cv2.putText(panel, f"Roll:  {roll:6.1f}° | Conf: {det.confidence:.2f}", 
                       (20, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Draw detections on frame
        for det in detections:
            x, y, w, h = det.bbox
            
            # Face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face'], 1)
            
            # Gaze arrow
            pitch, yaw, roll = det.pose
            if det.confidence > 0.3:
                center_x, center_y = x + w//2, y + h//2
                
                # Calculate arrow direction from pose
                arrow_length = min(w, h) * 0.6
                end_x = int(center_x + math.sin(math.radians(yaw)) * arrow_length)
                end_y = int(center_y + math.sin(math.radians(pitch)) * arrow_length)
                
                # Draw arrow
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                              self.colors['arrow'], 2, tipLength=0.3)
        
        return np.hstack([frame, panel])

class OptimizedApp:
    """Lightweight main application."""
    
    def __init__(self, camera_index: int = 0):
        self.detector = OptimizedDetector()
        self.ui = OptimizedUI()
        self.cap = None
        self.camera_index = camera_index
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 30.0
    
    def initialize_camera(self) -> bool:
        """Initialize camera with optimal settings."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            return True
        except:
            return False
    
    def run(self):
        """Optimized main loop."""
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return
        
        print("wandering-mind-detector - Optimized Version")
        print("Press 'q' to quit, 'r' to reset")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Detect faces and poses
                detections = self.detector.detect(frame)
                
                # Update FPS
                self.fps_counter += 1
                if self.fps_counter % 30 == 0:
                    elapsed = time.time() - self.fps_start
                    self.current_fps = 30 / elapsed
                    self.fps_start = time.time()
                
                # Render UI
                display_frame = self.ui.render(frame, detections, self.current_fps)
                
                # Show frame
                cv2.imshow('wandering-mind-detector', display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.detector.person_counter = 0
                    self.detector.pose_history.clear()
        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point."""
    import sys
    import time
    
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            pass
    
    app = OptimizedApp(camera_index)
    app.run()

if __name__ == "__main__":
    main()
