#!/usr/bin/env python3
"""
wandering-mind-detector - Clean Optimized Version
High-performance head pose and gaze tracking with 3D visualization
"""

import cv2
import numpy as np
import math
import time
from typing import Tuple, List, Dict
from dataclasses import dataclass
from collections import deque

@dataclass
class PoseData:
    """Clean pose data structure."""
    pitch: float
    yaw: float
    roll: float
    confidence: float

class SmoothingFilter:
    """Advanced smoothing filter to reduce glitch effects."""
    
    def __init__(self, window_size: int = 5, alpha: float = 0.7):
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        self.smoothed_values = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
    def update(self, pitch: float, yaw: float, roll: float) -> Tuple[float, float, float]:
        """Apply exponential moving average with outlier rejection."""
        # Store current values
        current = {'pitch': pitch, 'yaw': yaw, 'roll': roll}
        self.history.append(current)
        
        if len(self.history) < 3:
            return pitch, yaw, roll
        
        # Calculate moving average
        recent = list(self.history)[-3:]
        avg_pitch = sum(p['pitch'] for p in recent) / len(recent)
        avg_yaw = sum(p['yaw'] for p in recent) / len(recent)
        avg_roll = sum(p['roll'] for p in recent) / len(recent)
        
        # Apply exponential smoothing
        self.smoothed_values['pitch'] = (self.alpha * self.smoothed_values['pitch'] + 
                                       (1 - self.alpha) * avg_pitch)
        self.smoothed_values['yaw'] = (self.alpha * self.smoothed_values['yaw'] + 
                                     (1 - self.alpha) * avg_yaw)
        self.smoothed_values['roll'] = (self.alpha * self.smoothed_values['roll'] + 
                                      (1 - self.alpha) * avg_roll)
        
        # Outlier rejection
        if abs(pitch - self.smoothed_values['pitch']) > 30:  # Large jump
            self.smoothed_values['pitch'] = pitch
        if abs(yaw - self.smoothed_values['yaw']) > 30:
            self.smoothed_values['yaw'] = yaw
        if abs(roll - self.smoothed_values['roll']) > 30:
            self.smoothed_values['roll'] = roll
        
        return (self.smoothed_values['pitch'], 
                self.smoothed_values['yaw'], 
                self.smoothed_values['roll'])

class HeadPoseDetector:
    """Clean, efficient head pose detector."""
    
    def __init__(self):
        # Load cascades once
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Optimized 3D model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-165.0, -170.0, -135.0),    # Left eye
            (165.0, -170.0, -135.0),     # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ], dtype=np.float32)
        
        # Camera matrix (initialized once)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Smoothing filter
        self.smoother = SmoothingFilter()
        
    def detect_pose(self, frame: np.ndarray) -> List[PoseData]:
        """Detect head poses in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        poses = []
        for i, (x, y, w, h) in enumerate(faces[:3]):  # Max 3 faces
            # Detect eyes in face ROI
            roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15)
            )
            
            # Convert eye coordinates to frame coordinates
            eye_points = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
            
            # Estimate pose
            pose = self._estimate_pose((x, y, w, h), eye_points, frame.shape)
            poses.append(pose)
        
        return poses
    
    def _estimate_pose(self, bbox: Tuple[int, int, int, int], 
                      eyes: List[Tuple[int, int, int, int]], 
                      frame_shape: Tuple[int, int, int]) -> PoseData:
        """Estimate head pose using PnP."""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Initialize camera matrix once
        if self.camera_matrix is None:
            focal_length = max(frame_w, frame_h) * 0.8
            self.camera_matrix = np.array([
                [focal_length, 0, frame_w/2],
                [0, focal_length, frame_h/2],
                [0, 0, 1]
            ], dtype=np.float32)
        
        # Calculate 2D image points
        image_points = self._get_image_points(x, y, w, h, eyes)
        
        if len(image_points) < 4:
            return PoseData(0.0, 0.0, 0.0, 0.0)
        
        try:
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                self.model_points[:len(image_points)],
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return PoseData(0.0, 0.0, 0.0, 0.0)
            
            # Convert to Euler angles
            R, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = self._rotation_to_euler(R)
            
            # Apply smoothing
            smooth_pitch, smooth_yaw, smooth_roll = self.smoother.update(
                math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
            )
            
            # Calculate confidence based on pose stability
            confidence = self._calculate_confidence(smooth_pitch, smooth_yaw, smooth_roll)
            
            return PoseData(smooth_pitch, smooth_yaw, smooth_roll, confidence)
            
        except:
            return PoseData(0.0, 0.0, 0.0, 0.0)
    
    def _get_image_points(self, x: int, y: int, w: int, h: int, 
                         eyes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Calculate 2D image points from face and eyes."""
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
    
    def _rotation_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles."""
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
    
    def _calculate_confidence(self, pitch: float, yaw: float, roll: float) -> float:
        """Calculate pose confidence."""
        # Check for extreme angles
        if abs(pitch) > 60 or abs(yaw) > 60 or abs(roll) > 45:
            return 0.3
        
        # Higher confidence for more stable poses
        stability = 1.0 - (abs(pitch) + abs(yaw) + abs(roll)) / 180.0
        return max(0.1, min(1.0, stability))

class ThreeJSArrowRenderer:
    """Three.js-style 3D arrow renderer."""
    
    def __init__(self):
        self.arrow_length = 200  # 200px as requested
        self.colors = {
            'arrow': (100, 100, 255),    # Blue arrow
            'shadow': (50, 50, 150),     # Darker shadow
            'highlight': (150, 150, 255) # Bright highlight
        }
    
    def render_3d_arrow(self, frame: np.ndarray, pose: PoseData, 
                       center: Tuple[int, int]) -> np.ndarray:
        """Render Three.js-style 3D arrow."""
        if pose.confidence < 0.3:
            return frame
        
        cx, cy = center
        
        # Convert pose to 3D direction vector
        pitch_rad = math.radians(pose.pitch)
        yaw_rad = math.radians(pose.yaw)
        roll_rad = math.radians(pose.roll)
        
        # Calculate 3D direction
        x = math.sin(yaw_rad) * math.cos(pitch_rad)
        y = math.sin(pitch_rad)
        z = math.cos(yaw_rad) * math.cos(pitch_rad)
        
        # Project to 2D screen coordinates
        screen_x = int(cx + x * self.arrow_length)
        screen_y = int(cy + y * self.arrow_length)
        
        # Apply roll rotation
        cos_roll = math.cos(roll_rad)
        sin_roll = math.sin(roll_rad)
        
        # Rotate arrow endpoints
        dx = screen_x - cx
        dy = screen_y - cy
        
        rotated_x = int(cx + dx * cos_roll - dy * sin_roll)
        rotated_y = int(cy + dx * sin_roll + dy * cos_roll)
        
        # Render 3D arrow with depth effect
        self._draw_3d_arrow(frame, (cx, cy), (rotated_x, rotated_y))
        
        return frame
    
    def _draw_3d_arrow(self, frame: np.ndarray, start: Tuple[int, int], 
                      end: Tuple[int, int]):
        """Draw 3D arrow with depth effects."""
        sx, sy = start
        ex, ey = end
        
        # Calculate arrow properties
        dx = ex - sx
        dy = ey - sy
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 10:  # Too short
            return
        
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Arrow head size
        head_length = min(30, length * 0.2)
        head_width = 15
        
        # Calculate arrow head points
        head_x1 = int(ex - head_length * dx_norm + head_width * dy_norm)
        head_y1 = int(ey - head_length * dy_norm - head_width * dx_norm)
        head_x2 = int(ex - head_length * dx_norm - head_width * dy_norm)
        head_y2 = int(ey - head_length * dy_norm + head_width * dx_norm)
        
        # Draw shadow (offset)
        shadow_offset = 2
        cv2.line(frame, (sx + shadow_offset, sy + shadow_offset), 
                (ex + shadow_offset, ey + shadow_offset), self.colors['shadow'], 3)
        
        # Draw main arrow shaft
        cv2.line(frame, (sx, sy), (ex, ey), self.colors['arrow'], 3)
        
        # Draw arrow head
        cv2.line(frame, (ex, ey), (head_x1, head_y1), self.colors['arrow'], 3)
        cv2.line(frame, (ex, ey), (head_x2, head_y2), self.colors['arrow'], 3)
        cv2.line(frame, (head_x1, head_y1), (head_x2, head_y2), self.colors['arrow'], 2)
        
        # Draw highlight
        cv2.line(frame, (sx, sy), (ex, ey), self.colors['highlight'], 1)

class CleanUI:
    """Clean, efficient UI renderer."""
    
    def __init__(self, panel_width: int = 300):
        self.panel_width = panel_width
        self.colors = {
            'bg': (45, 45, 45),
            'text': (255, 255, 255),
            'accent': (100, 150, 255),
            'face': (0, 255, 100),
            'card_bg': (65, 65, 65)
        }
    
    def render(self, frame: np.ndarray, poses: List[PoseData], fps: float) -> np.ndarray:
        """Render clean UI with pose information."""
        h, w = frame.shape[:2]
        
        # Create panel
        panel = np.full((h, self.panel_width, 3), self.colors['bg'], dtype=np.uint8)
        
        # Header
        cv2.putText(panel, "WANDERING MIND DETECTOR", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Stats
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        cv2.putText(panel, f"Poses: {len(poses)}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        
        # Pose cards
        y_offset = 110
        for i, pose in enumerate(poses):
            card_y = y_offset + i * 80
            
            # Card background
            cv2.rectangle(panel, (10, card_y), (self.panel_width - 10, card_y + 70), 
                         self.colors['card_bg'], -1)
            cv2.rectangle(panel, (10, card_y), (self.panel_width - 10, card_y + 70), 
                         self.colors['text'], 1)
            
            # Pose info
            cv2.putText(panel, f"Person {i+1}", (20, card_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            # Pose values
            cv2.putText(panel, f"Pitch: {pose.pitch:6.1f}° | Yaw: {pose.yaw:6.1f}°", 
                       (20, card_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            cv2.putText(panel, f"Roll:  {pose.roll:6.1f}° | Conf: {pose.confidence:.2f}", 
                       (20, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return np.hstack([frame, panel])

class CleanApp:
    """Clean, optimized main application."""
    
    def __init__(self, camera_index: int = 0):
        self.detector = HeadPoseDetector()
        self.arrow_renderer = ThreeJSArrowRenderer()
        self.ui = CleanUI()
        self.cap = None
        self.camera_index = camera_index
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 30.0
        
        # Face tracking
        self.face_centers = []
    
    def initialize_camera(self) -> bool:
        """Initialize camera with optimal settings."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Optimize camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            return True
        except:
            return False
    
    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return
        
        print("wandering-mind-detector - Clean Version")
        print("Press 'q' to quit, 'r' to reset")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Detect poses
                poses = self.detector.detect_pose(frame)
                
                # Update face centers for arrow rendering
                self.face_centers = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for x, y, w, h in faces[:len(poses)]:
                    center = (x + w//2, y + h//2)
                    self.face_centers.append(center)
                    
                    # Draw face box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 1)
                
                # Render 3D arrows
                for i, (pose, center) in enumerate(zip(poses, self.face_centers)):
                    frame = self.arrow_renderer.render_3d_arrow(frame, pose, center)
                
                # Update FPS
                self.fps_counter += 1
                if self.fps_counter % 30 == 0:
                    elapsed = time.time() - self.fps_start
                    self.current_fps = 30 / elapsed
                    self.fps_start = time.time()
                
                # Render UI
                display_frame = self.ui.render(frame, poses, self.current_fps)
                
                # Show frame
                cv2.imshow('wandering-mind-detector', display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.detector.smoother = SmoothingFilter()  # Reset smoother
        
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
    
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            pass
    
    app = CleanApp(camera_index)
    app.run()

if __name__ == "__main__":
    main()
