"""
Enhanced Minimal UI with AOI System
Clean interface with Area of Interest functionality

Author: Cursey Team
Date: 2025
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
from high_performance_detector import FastFaceDetection

logger = logging.getLogger(__name__)


class EnhancedMinimalUI:
    """Enhanced minimal UI with AOI system and improved stability."""
    
    def __init__(self, camera_width: int = 640, camera_height: int = 480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.panel_width = 320
        self.total_width = camera_width + self.panel_width
        self.total_height = max(camera_height, 520)
        
        # Session tracking
        self.session_start = time.time()
        self.frame_count = 0
        self.total_detections = 0
        
        # AOI system
        self.aoi_reference = None
        self.aoi_active = False
        
        # Clean minimal color scheme
        self.colors = {
            'bg_primary': (20, 20, 25),       # Dark background
            'bg_secondary': (30, 30, 35),     # Secondary background
            'bg_panel': (25, 25, 30),          # Panel background
            'text_primary': (255, 255, 255),  # White text
            'text_secondary': (200, 200, 200), # Light gray text
            'text_muted': (150, 150, 150),    # Muted text
            'text_accent': (100, 150, 255),    # Blue accent
            'border': (80, 80, 80),           # Border color
            'face_box': (255, 255, 255),      # White face bounding box
            'eye_circle': (255, 255, 255),    # White eye circle
            'table_header': (60, 60, 65),      # Table header background
            'table_row': (35, 35, 40),         # Table row background
            'table_row_alt': (40, 40, 45),     # Alternate table row
            'button_bg': (50, 50, 55),         # Button background
            'button_hover': (70, 70, 75),       # Button hover
            'button_active': (90, 90, 95),     # Button active
            'aoi_active': (0, 255, 100),       # Green for AOI active
            'aoi_inactive': (255, 100, 100),   # Red for AOI inactive
        }
        
        # Button definitions
        self.buttons = {
            'reset_aoi': {'rect': (15, 15, 120, 30), 'text': 'Reset AOI', 'active': False}
        }
        
        # Layout sections with proper margins
        margin = 15
        self.sections = {
            'header': (margin, margin, self.panel_width - 2*margin, 50),
            'pose_table': (margin, margin + 60, self.panel_width - 2*margin, 220),
            'aoi_status': (margin, margin + 290, self.panel_width - 2*margin, 60),
            'stats': (margin, margin + 360, self.panel_width - 2*margin, 80),
            'footer': (margin, self.total_height - 30, self.panel_width - 2*margin, 20)
        }
        
        # Performance optimizations
        self.render_cache = None
        self.cache_valid = False
        self.last_persons_count = 0
        self.last_aoi_state = None
    
    def _draw_button(self, canvas: np.ndarray, button_name: str) -> None:
        """Draw clean button."""
        button = self.buttons[button_name]
        x, y, w, h = button['rect']
        
        # Button background
        if button['active']:
            color = self.colors['button_active']
            border_color = self.colors['text_accent']
        else:
            color = self.colors['button_bg']
            border_color = self.colors['border']
        
        # Draw button
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)
        
        # Button text
        text_size = cv2.getTextSize(button['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        text_color = self.colors['text_primary'] if button['active'] else self.colors['text_secondary']
        cv2.putText(canvas, button['text'], (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    def _draw_face_bounding_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                              aoi_compliant: bool = False) -> None:
        """Draw white thin bounding box around face with AOI indication."""
        x, y, w, h = bbox
        
        # Choose color based on AOI compliance
        color = self.colors['aoi_active'] if aoi_compliant else self.colors['face_box']
        thickness = 3 if aoi_compliant else 2
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    def _draw_eye_circles(self, frame: np.ndarray, eyes: Dict[str, any]) -> None:
        """Draw white circles around detected eyes."""
        if not eyes:
            return
        
        for eye_name, eye_detection in eyes.items():
            if eye_detection and hasattr(eye_detection, 'center'):
                center = eye_detection.center
                radius = 8
                cv2.circle(frame, center, radius, self.colors['eye_circle'], 2)
    
    def _draw_pose_table(self, canvas: np.ndarray, persons: Dict[int, Dict]) -> None:
        """Draw enhanced table for pose values with AOI compliance."""
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
        cv2.putText(canvas, "Pitch", (x + 40, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Roll", (x + 85, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Yaw", (x + 130, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Conf", (x + 175, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "AOI", (x + 220, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Stab", (x + 250, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
        
        # Table rows
        row_height = 18
        max_rows = min(len(persons), 9)  # Increased to 9 rows
        
        for i, (person_id, person) in enumerate(list(persons.items())[:max_rows]):
            row_y = header_y + 15 + (i * row_height)
            
            # Alternate row background
            if i % 2 == 0:
                row_color = self.colors['table_row']
            else:
                row_color = self.colors['table_row_alt']
            
            cv2.rectangle(canvas, (x + 5, row_y - 12), (x + w - 5, row_y + 6), 
                         row_color, -1)
            
            # Get pose data
            if 'face_detection' in person and person['face_detection'].pose_angles:
                pose = person['face_detection'].pose_angles
                pitch = pose.get('pitch', 0)
                roll = pose.get('roll', 0)
                yaw = pose.get('yaw', 0)
                confidence = person['face_detection'].confidence
                stability = person['face_detection'].stability_score
                aoi_compliant = person.get('aoi_compliant', False)
            else:
                pitch = roll = yaw = 0
                confidence = 0
                stability = 0
                aoi_compliant = False
            
            # Row data
            cv2.putText(canvas, str(person_id), (x + 15, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
            cv2.putText(canvas, f"{pitch:.1f}°", (x + 40, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{roll:.1f}°", (x + 85, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{yaw:.1f}°", (x + 130, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{confidence:.2f}", (x + 175, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            
            # AOI compliance indicator
            aoi_color = self.colors['aoi_active'] if aoi_compliant else self.colors['aoi_inactive']
            aoi_text = "✓" if aoi_compliant else "✗"
            cv2.putText(canvas, aoi_text, (x + 220, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, aoi_color, 1)
            
            # Stability indicator
            stability_color = self.colors['aoi_active'] if stability > 0.7 else self.colors['aoi_inactive']
            cv2.putText(canvas, f"{stability:.2f}", (x + 250, row_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, stability_color, 1)
    
    def _draw_aoi_status(self, canvas: np.ndarray) -> None:
        """Draw AOI status section."""
        x, y, w, h = self.sections['aoi_status']
        
        # AOI status background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['bg_panel'], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['border'], 1)
        
        # AOI status title
        cv2.putText(canvas, "AOI STATUS", (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_primary'], 2)
        
        # AOI status
        if self.aoi_reference:
            status_text = "AOI Active"
            status_color = self.colors['aoi_active']
            
            # Show reference values
            ref_text = f"Ref: P={self.aoi_reference['pitch']:.1f}° R={self.aoi_reference['roll']:.1f}° Y={self.aoi_reference['yaw']:.1f}°"
            cv2.putText(canvas, ref_text, (x + 10, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
        else:
            status_text = "AOI Inactive"
            status_color = self.colors['aoi_inactive']
        
        cv2.putText(canvas, status_text, (x + 10, y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
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
        
        # Detection count
        cv2.putText(canvas, f"Detections: {self.total_detections:,}", 
                   (x + 10, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   self.colors['text_secondary'], 1)
    
    def _draw_footer(self, canvas: np.ndarray) -> None:
        """Draw minimal footer."""
        x, y, w, h = self.sections['footer']
        
        # Footer background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['bg_secondary'], -1)
        
        # Status text
        cv2.putText(canvas, "Enhanced Face Detection Active", (x + 10, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
    
    def handle_mouse_click(self, x: int, y: int) -> str:
        """Handle mouse clicks on UI elements."""
        # Check button clicks
        for button_name, button in self.buttons.items():
            bx, by, bw, bh = button['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return button_name
        return None
    
    def toggle_button(self, button_name: str) -> None:
        """Toggle button state."""
        if button_name in self.buttons:
            self.buttons[button_name]['active'] = not self.buttons[button_name]['active']
    
    def set_aoi_reference(self, pose_angles: Dict[str, float]) -> None:
        """Set AOI reference values."""
        self.aoi_reference = pose_angles.copy()
        self.aoi_active = True
        logger.info(f"AOI reference set: {self.aoi_reference}")
    
    def reset_aoi(self) -> None:
        """Reset AOI reference."""
        self.aoi_reference = None
        self.aoi_active = False
        logger.info("AOI reference reset")
    
    def render(self, camera_frame: np.ndarray, persons: Dict[int, Dict]) -> np.ndarray:
        """Render the enhanced minimal UI with camera feed and side panel (optimized)."""
        # Check if we can use cached panel
        current_persons_count = len(persons)
        current_aoi_state = self.aoi_reference is not None
        
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
                    aoi_compliant = person.get('aoi_compliant', False)
                    self._draw_face_bounding_box(canvas, face_det.bbox, aoi_compliant)
                
                if 'eyes' in person:
                    self._draw_eye_circles(canvas, person['eyes'])
        
        # Check if we need to redraw panel
        if (not self.cache_valid or 
            current_persons_count != self.last_persons_count or 
            current_aoi_state != self.last_aoi_state):
            
            # Create side panel
            panel = np.zeros((self.total_height, self.panel_width, 3), dtype=np.uint8)
            panel.fill(25)  # Panel background
            
            # Draw UI sections
            self._draw_button(panel, 'reset_aoi')
            self._draw_pose_table(panel, persons)
            self._draw_aoi_status(panel)
            self._draw_stats(panel)
            self._draw_footer(panel)
            
            # Cache the panel
            self.render_cache = panel.copy()
            self.cache_valid = True
            self.last_persons_count = current_persons_count
            self.last_aoi_state = current_aoi_state
        
        # Use cached panel
        canvas[:, self.camera_width:] = self.render_cache
        
        # Update frame count
        self.frame_count += 1
        
        return canvas


if __name__ == "__main__":
    print("Enhanced Minimal UI loaded successfully!")
