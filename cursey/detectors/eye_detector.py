"""
Eye Detection Module for Cursey
Detects eyes and extracts eye features from webcam feed using OpenCV

Author: Cursey Team
Date: 2025
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os


class EyeDetector:
    """
    Advanced eye detection using Haar Cascade classifiers.
    Detects faces and eyes, extracts eye regions, and tracks gaze direction.
    """
    
    def __init__(self, cascade_path: str = None):
        """
        Initialize the eye detector with cascade classifiers.
        
        Args:
            cascade_path: Path to directory containing cascade XML files
        """
        # Default paths for cascade files
        if cascade_path is None:
            cascade_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Load face cascade classifier
        face_cascade_path = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Load eye cascade classifier
        eye_cascade_path = os.path.join(cascade_path, 'haarcascade_eye.xml')
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Verify cascade classifiers loaded successfully
        if self.face_cascade.empty():
            print(f"Warning: Could not load face cascade from {face_cascade_path}")
        if self.eye_cascade.empty():
            print(f"Warning: Could not load eye cascade from {eye_cascade_path}")
    
    def detect_face(self, frame: np.ndarray) -> Optional[List]:
        """
        Detect faces in the frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            List of face rectangles (x, y, width, height) or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces if len(faces) > 0 else None
    
    def detect_eyes(self, face_roi: np.ndarray) -> Tuple[Optional[List], Optional[List]]:
        """
        Detect eyes within a face region.
        
        Args:
            face_roi: Region of interest containing a face
            
        Returns:
            Tuple of (left_eye, right_eye) coordinates
        """
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate to determine left and right
            eyes_sorted = sorted(eyes, key=lambda x: x[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1]
            return left_eye, right_eye
        else:
            return None, None
    
    def extract_eye_features(self, eye_roi: np.ndarray) -> dict:
        """
        Extract features from an eye region.
        
        Args:
            eye_roi: Region containing an eye
            
        Returns:
            Dictionary containing eye features:
            - center: (x, y) center coordinates
            - area: Eye area in pixels
            - aspect_ratio: Width/height ratio
            - brightness: Average pixel intensity
            - gaze_direction: Estimated gaze direction angle in degrees
        """
        if eye_roi is None or eye_roi.size == 0:
            return None
        
        h, w = eye_roi.shape[:2]
        center = (w // 2, h // 2)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate average brightness
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        brightness = np.mean(gray)
        
        # Estimate gaze direction based on iris position
        # Simplified gaze estimation: analyze brightness distribution
        left_brightness = np.mean(gray[:, :w//2])
        right_brightness = np.mean(gray[:, w//2:])
        top_brightness = np.mean(gray[:h//2, :])
        bottom_brightness = np.mean(gray[h//2:, :])
        
        # Calculate gaze angle (0 = center, -90 to 90 = horizontal, -45 to 45 = diagonal)
        gaze_x = (right_brightness - left_brightness) / max(brightness, 1) * 30
        gaze_y = (bottom_brightness - top_brightness) / max(brightness, 1) * 30
        gaze_angle = np.arctan2(gaze_y, gaze_x) * 180 / np.pi
        
        return {
            'center': center,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'brightness': brightness,
            'gaze_direction': gaze_angle,
            'gaze_vector': (gaze_x, gaze_y)
        }
    
    def estimate_head_pose(self, face_box, left_eye_box, right_eye_box):
        """
        Estimate 3-DOF head pose (roll, pitch, yaw) from face and eye positions.
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            left_eye_box: Left eye bounding box
            right_eye_box: Right eye bounding box
            
        Returns:
            Dictionary with pitch, yaw, roll angles in degrees
        """
        if not left_eye_box or not right_eye_box:
            return {'pitch': 0, 'yaw': 0, 'roll': 0}
        
        fx, fy, fw, fh = face_box
        lx, ly, lw, lh = left_eye_box
        rx, ry, rw, rh = right_eye_box
        
        # Calculate face center
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2
        
        # Calculate eye centers relative to face
        left_eye_center = (lx + lw // 2, ly + lh // 2)
        right_eye_center = (rx + rw // 2, ry + rh // 2)
        
        # Roll: rotation around Z-axis (tilt left/right)
        eye_line_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], 
                                     right_eye_center[0] - left_eye_center[0])
        roll = np.degrees(eye_line_angle)
        
        # Yaw: rotation around Y-axis (left/right turn)
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        face_normal_x = face_center_x
        yaw_offset = (eye_center_x - face_normal_x) / fw * 30
        yaw = yaw_offset
        
        # Pitch: rotation around X-axis (up/down tilt)
        eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
        face_normal_y = face_center_y - fh * 0.1  # Eyes typically higher than center
        pitch_offset = (eye_center_y - face_normal_y) / fh * 30
        pitch = pitch_offset
        
        return {'pitch': pitch, 'yaw': yaw, 'roll': roll}
    
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and extract all eye information.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Dictionary containing:
            - faces: List of detected faces
            - eyes: List of detected eyes with features
            - frame_with_detections: Annotated frame
        """
        result = {
            'faces': [],
            'eyes': [],
            'frame_with_detections': frame.copy()
        }
        
        # Detect faces
        faces = self.detect_face(frame)
        
        if faces is not None:
            for (x, y, w, h) in faces:
                # Extract face region (no visual outline)
                face_roi = frame[y:y+h, x:x+w]
                
                # Detect eyes in face region
                left_eye, right_eye = self.detect_eyes(face_roi)
                
                eye_data = {'face_box': (x, y, w, h), 'left_eye': None, 'right_eye': None}
                
                left_eye_box = None
                right_eye_box = None
                
                # Process left eye
                if left_eye is not None:
                    lx, ly, lw, lh = left_eye
                    eye_roi = face_roi[ly:ly+lh, lx:lx+lw]
                    eye_features = self.extract_eye_features(eye_roi)
                    
                    if eye_features:
                        eye_center = (x+lx+lw//2, y+ly+lh//2)
                        left_eye_box = (x+lx, y+ly, lw, lh)
                        eye_data['left_eye'] = {
                            'box': left_eye_box,
                            'features': eye_features,
                            'center': eye_center
                        }
                        # No visualization - data only
                
                # Process right eye
                if right_eye is not None:
                    rx, ry, rw, rh = right_eye
                    eye_roi = face_roi[ry:ry+rh, rx:rx+rw]
                    eye_features = self.extract_eye_features(eye_roi)
                    
                    if eye_features:
                        eye_center = (x+rx+rw//2, y+ry+rh//2)
                        right_eye_box = (x+rx, y+ry, rw, rh)
                        eye_data['right_eye'] = {
                            'box': right_eye_box,
                            'features': eye_features,
                            'center': eye_center
                        }
                        # No visualization - data only
                
                # Calculate head pose angles (data only, no visualization)
                if left_eye_box and right_eye_box:
                    head_pose = self.estimate_head_pose((x, y, w, h), left_eye_box, right_eye_box)
                    # Scale angles to -180 to 180 range
                    head_pose['pitch'] = np.clip(head_pose['pitch'] * 6, -180, 180)
                    head_pose['yaw'] = np.clip(head_pose['yaw'] * 6, -180, 180)
                    head_pose['roll'] = np.clip(head_pose['roll'] * 6, -180, 180)
                    
                    # Store head pose data for mannequin widget
                    eye_data['head_pose'] = head_pose
                
                if eye_data['left_eye'] or eye_data['right_eye']:
                    result['eyes'].append(eye_data)
                    result['faces'].append({'box': (x, y, w, h)})
        
        return result


if __name__ == "__main__":
    # Example usage
    print("Eye Detection Module loaded successfully!")
    print("Import this module in your application to use.")
