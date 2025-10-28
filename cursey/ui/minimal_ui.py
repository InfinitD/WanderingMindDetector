"""
Minimal Clean UI for Face Detection
Simple, functional interface with proper layout

Author: Cursey Team
Date: 2025
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
from yolo_face_detector import FaceDetection

logger = logging.getLogger(__name__)


class MinimalUI:
    """Minimal clean UI with proper layout and simple design."""
    
    def __init__(self, camera_width: int = 640, camera_height: int = 480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.panel_width = 300
        self.total_width = camera_width + self.panel_width
        self.total_height = max(camera_height, 500)
        
        # Session tracking
        self.session_start = time.time()
        self.frame_count = 0
        self.total_detections = 0
        
        # Clean minimal color scheme
        self.colors = {
            'bg_primary': (20, 20, 25),       # Dark background
            'bg_secondary': (30, 30, 35),     # Secondary background
            'bg_panel': (25, 25, 30),          # Panel background
            'text_primary': (255, 255, 255),  # White text
            'text_secondary': (200, 200, 200), # Light gray text
            'text_muted': (150, 150, 150),    # Muted text
            'border': (80, 80, 80),           # Border color
            'face_box': (255, 255, 255),      # White face bounding box
            'eye_circle': (255, 255, 255),    # White eye circle
            'table_header': (60, 60, 65),      # Table header background
            'table_row': (35, 35, 40),         # Table row background
        }
        
        # Layout sections with proper margins
        margin = 15
        self.sections = {
            'header': (margin, margin, self.panel_width - 2*margin, 40),
            'pose_table': (margin, margin + 50, self.panel_width - 2*margin, 200),
            'stats': (margin, margin + 260, self.panel_width - 2*margin, 80),
            'footer': (margin, self.total_height - 40, self.panel_width - 2*margin, 25)
        }
    
    def _draw_face_bounding_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Draw white thin bounding box around face."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face_box'], 2)
    
    def _draw_eye_circles(self, frame: np.ndarray, eyes: Dict[str, any]) -> None:
        """Draw white circles around detected eyes."""
        if not eyes:
            return
        
        for eye_name, eye_detection in eyes.items():
            if eye_detection and hasattr(eye_detection, 'center'):
                center = eye_detection.center
                radius = 8  # Fixed radius for eye circles
                cv2.circle(frame, center, radius, self.colors['eye_circle'], 2)
    
    def _draw_pose_table(self, canvas: np.ndarray, persons: Dict[int, Dict]) -> None:
        """Draw clean table for pose values."""
        x, y, w, h = self.sections['pose_table']
        
        # Table background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['bg_panel'], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['border'], 1)
        
        # Table title
        cv2.putText(canvas, "POSE VALUES", (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_primary'], 2)
        
        if not persons:
            cv2.putText(canvas, "No persons detected", (x + 10, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_muted'], 1)
            return
        
        # Table header
        header_y = y + 40
        cv2.rectangle(canvas, (x + 5, header_y - 15), (x + w - 5, header_y + 5), 
                     self.colors['table_header'], -1)
        
        # Header text
        cv2.putText(canvas, "ID", (x + 15, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Pitch", (x + 50, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Roll", (x + 100, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Yaw", (x + 150, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Conf", (x + 200, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        
        # Table rows
        row_height = 20
        max_rows = min(len(persons), 7)  # Limit to 7 rows
        
        for i, (person_id, person) in enumerate(list(persons.items())[:max_rows]):
            row_y = header_y + 15 + (i * row_height)
            
            # Alternate row background
            if i % 2 == 0:
                cv2.rectangle(canvas, (x + 5, row_y - 12), (x + w - 5, row_y + 8), 
                             self.colors['table_row'], -1)
            
            # Get pose data
            if 'face_detection' in person and person['face_detection'].pose_angles:
                pose = person['face_detection'].pose_angles
                pitch = pose.get('pitch', 0)
                roll = pose.get('roll', 0)
                yaw = pose.get('yaw', 0)
                confidence = person['face_detection'].confidence
            else:
                pitch = roll = yaw = 0
                confidence = 0
            
            # Row data
            cv2.putText(canvas, str(person_id), (x + 15, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
            cv2.putText(canvas, f"{pitch:.1f}°", (x + 50, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{roll:.1f}°", (x + 100, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{yaw:.1f}°", (x + 150, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{confidence:.2f}", (x + 200, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
    
    def _draw_stats(self, canvas: np.ndarray) -> None:
        """Draw minimal statistics."""
        x, y, w, h = self.sections['stats']
        
        # Stats background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['bg_panel'], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['border'], 1)
        
        # Stats title
        cv2.putText(canvas, "STATISTICS", (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_primary'], 2)
        
        # Duration
        duration = time.time() - self.session_start
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        cv2.putText(canvas, f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}", 
                   (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   self.colors['text_secondary'], 1)
        
        # Frame count
        cv2.putText(canvas, f"Frames: {self.frame_count:,}", 
                   (x + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   self.colors['text_secondary'], 1)
    
    def _draw_footer(self, canvas: np.ndarray) -> None:
        """Draw minimal footer."""
        x, y, w, h = self.sections['footer']
        
        # Footer background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['bg_secondary'], -1)
        
        # Status text
        cv2.putText(canvas, "Face Detection Active", (x + 10, y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
    
    def render(self, camera_frame: np.ndarray, persons: Dict[int, Dict]) -> np.ndarray:
        """Render the minimal UI with camera feed and side panel."""
        # Create main canvas
        canvas = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
        canvas.fill(20)  # Dark background
        
        # Place camera feed
        if camera_frame is not None:
            resized_camera = cv2.resize(camera_frame, (self.camera_width, self.camera_height))
            canvas[:self.camera_height, :self.camera_width] = resized_camera
            
            # Draw face bounding boxes and eye circles on camera feed
            for person in persons.values():
                if 'face_detection' in person:
                    face_det = person['face_detection']
                    self._draw_face_bounding_box(canvas, face_det.bbox)
                
                if 'eyes' in person:
                    self._draw_eye_circles(canvas, person['eyes'])
        
        # Create side panel
        panel = np.zeros((self.total_height, self.panel_width, 3), dtype=np.uint8)
        panel.fill(25)  # Panel background
        
        # Draw UI sections
        self._draw_pose_table(panel, persons)
        self._draw_stats(panel)
        self._draw_footer(panel)
        
        # Place panel on canvas
        canvas[:, self.camera_width:] = panel
        
        # Update frame count
        self.frame_count += 1
        
        return canvas


if __name__ == "__main__":
    print("Minimal UI loaded successfully!")
