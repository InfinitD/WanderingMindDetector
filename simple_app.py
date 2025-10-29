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
        card_height = 140
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
