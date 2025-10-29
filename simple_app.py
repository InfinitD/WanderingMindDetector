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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    person_id: int
    pose: Dict[str, float]  # pitch, yaw, roll
    eyes: List[Tuple[int, int, int, int]]  # eye bounding boxes
    quality_score: float

class WanderingMindDetector:
    """wandering-mind-detector using OpenCV Haar Cascades."""
    
    def __init__(self, max_persons: int = 3):
        self.max_persons = max_persons
        self.confidence_threshold = 0.3
        
        # Load Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
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
            
            # Simple pose estimation (placeholder)
            pose = {
                'pitch': 0.0,
                'yaw': 0.0,
                'roll': 0.0
            }
            
            detection = FaceDetection(
                bbox=(x, y, w, h),
                confidence=confidence,
                person_id=self.person_counter + i,
                pose=pose,
                eyes=eye_boxes,
                quality_score=confidence
            )
            
            detections.append(detection)
        
        # Update person counter
        self.person_counter = (self.person_counter + len(detections)) % 1000
        
        # Store in history
        self.detection_history.append(detections)
        
        return detections

class SimpleUI:
    """Simple UI for displaying face detection results."""
    
    def __init__(self, panel_width: int = 300):
        self.panel_width = panel_width
        self.colors = {
            'background': (240, 240, 240),
            'panel': (250, 250, 250),
            'text': (50, 50, 50),
            'face_box': (0, 255, 0),
            'eye_box': (255, 0, 0),
            'text_bg': (255, 255, 255)
        }
    
    def render(self, frame: np.ndarray, detections: List[FaceDetection], fps: float) -> np.ndarray:
        """Render the UI on the frame."""
        h, w = frame.shape[:2]
        
        # Create side panel
        panel = np.full((h, self.panel_width, 3), self.colors['panel'], dtype=np.uint8)
        
        # Draw header
        cv2.putText(panel, "wandering-mind-detector", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Draw FPS
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Draw detection count
        cv2.putText(panel, f"Faces: {len(detections)}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Draw face information
        y_offset = 110
        for i, detection in enumerate(detections):
            # Person ID
            cv2.putText(panel, f"Person {detection.person_id}:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Confidence
            cv2.putText(panel, f"Confidence: {detection.confidence:.2f}", (10, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Eyes count
            cv2.putText(panel, f"Eyes: {len(detection.eyes)}", (10, y_offset + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            y_offset += 70
        
        # Draw face bounding boxes on main frame
        for detection in detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face_box'], 2)
            
            # Draw eyes
            for ex, ey, ew, eh in detection.eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), self.colors['eye_box'], 1)
        
        # Combine frame and panel
        combined = np.hstack([frame, panel])
        return combined

class WanderingMindApp:
    """Main wandering-mind-detector application."""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.detector = WanderingMindDetector()
        self.ui = SimpleUI()
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
