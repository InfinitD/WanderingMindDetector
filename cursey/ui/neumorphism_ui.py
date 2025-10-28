"""
Modern Neumorphism UI for Cursey Face Detection
Clean, modern interface with neumorphism design principles

Author: Cursey Team
Date: 2025
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
from detectron_detector import DetectronFaceDetection

logger = logging.getLogger(__name__)


class NeumorphismUI:
    """Modern neumorphism UI with clean design and smooth animations."""
    
    def __init__(self, camera_width: int = 640, camera_height: int = 480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.panel_width = 380
        self.total_width = camera_width + self.panel_width
        self.total_height = max(camera_height, 600)
        
        # Session tracking
        self.session_start = time.time()
        self.frame_count = 0
        self.total_detections = 0
        
        # AOI system
        self.aoi_reference = None
        self.aoi_active = False
        
        # Neumorphism color scheme
        self.colors = {
            # Base colors (soft, muted tones)
            'bg_primary': (240, 240, 245),      # Light gray background
            'bg_secondary': (230, 230, 235),    # Slightly darker gray
            'bg_panel': (235, 235, 240),        # Panel background
            'bg_card': (245, 245, 250),         # Card background
            
            # Text colors
            'text_primary': (60, 60, 70),       # Dark gray text
            'text_secondary': (100, 100, 110),  # Medium gray text
            'text_muted': (140, 140, 150),      # Light gray text
            'text_accent': (80, 120, 200),      # Blue accent
            
            # Neumorphism shadows and highlights
            'shadow_dark': (200, 200, 210),     # Dark shadow
            'shadow_light': (255, 255, 255),    # Light highlight
            'border_subtle': (220, 220, 225),   # Subtle border
            
            # Status colors
            'success': (76, 175, 80),           # Green
            'warning': (255, 193, 7),           # Yellow
            'error': (244, 67, 54),             # Red
            'info': (33, 150, 243),             # Blue
            
            # Face detection colors
            'face_box_active': (80, 120, 200),  # Blue for active faces
            'face_box_inactive': (150, 150, 160), # Gray for inactive faces
            'eye_circle': (100, 100, 110),      # Dark gray for eyes
            
            # Button colors
            'button_bg': (235, 235, 240),       # Button background
            'button_pressed': (225, 225, 230),  # Pressed button
            'button_hover': (240, 240, 245),    # Hover button
        }
        
        # Button definitions with neumorphism styling
        self.buttons = {
            'reset_aoi': {
                'rect': (20, 20, 140, 40), 
                'text': 'Reset AOI', 
                'active': False,
                'pressed': False
            },
            'toggle_detection': {
                'rect': (180, 20, 160, 40),
                'text': 'Detection ON',
                'active': True,
                'pressed': False
            }
        }
        
        # Layout sections with proper spacing
        margin = 20
        self.sections = {
            'header': (margin, margin, self.panel_width - 2*margin, 80),
            'detection_stats': (margin, margin + 100, self.panel_width - 2*margin, 120),
            'pose_table': (margin, margin + 240, self.panel_width - 2*margin, 200),
            'aoi_status': (margin, margin + 460, self.panel_width - 2*margin, 80),
            'footer': (margin, self.total_height - 40, self.panel_width - 2*margin, 30)
        }
        
        # Performance optimizations
        self.render_cache = None
        self.cache_valid = False
        self.last_persons_count = 0
        self.last_aoi_state = None
        
        # Animation states
        self.animation_time = 0
        self.pulse_animation = 0
    
    def _draw_neumorphism_button(self, canvas: np.ndarray, button_name: str) -> None:
        """Draw neumorphism-style button with depth and shadows."""
        button = self.buttons[button_name]
        x, y, w, h = button['rect']
        
        # Button background
        bg_color = self.colors['button_bg']
        if button['pressed']:
            bg_color = self.colors['button_pressed']
        
        # Draw button base
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
        
        # Neumorphism effect: draw shadows and highlights
        if not button['pressed']:
            # Light highlight (top and left edges)
            cv2.line(canvas, (x, y), (x + w, y), self.colors['shadow_light'], 2)
            cv2.line(canvas, (x, y), (x, y + h), self.colors['shadow_light'], 2)
            
            # Dark shadow (bottom and right edges)
            cv2.line(canvas, (x, y + h), (x + w, y + h), self.colors['shadow_dark'], 2)
            cv2.line(canvas, (x + w, y), (x + w, y + h), self.colors['shadow_dark'], 2)
        else:
            # Inverted shadows for pressed effect
            cv2.line(canvas, (x, y), (x + w, y), self.colors['shadow_dark'], 2)
            cv2.line(canvas, (x, y), (x, y + h), self.colors['shadow_dark'], 2)
            cv2.line(canvas, (x, y + h), (x + w, y + h), self.colors['shadow_light'], 2)
            cv2.line(canvas, (x + w, y), (x + w, y + h), self.colors['shadow_light'], 2)
        
        # Button text
        text_size = cv2.getTextSize(button['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        text_color = self.colors['text_primary'] if button['active'] else self.colors['text_muted']
        cv2.putText(canvas, button['text'], (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    def _draw_neumorphism_card(self, canvas: np.ndarray, rect: Tuple[int, int, int, int], 
                              title: str = None) -> None:
        """Draw neumorphism-style card with depth."""
        x, y, w, h = rect
        
        # Card background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['bg_card'], -1)
        
        # Neumorphism effect
        # Light highlight (top and left edges)
        cv2.line(canvas, (x, y), (x + w, y), self.colors['shadow_light'], 1)
        cv2.line(canvas, (x, y), (x, y + h), self.colors['shadow_light'], 1)
        
        # Dark shadow (bottom and right edges)
        cv2.line(canvas, (x, y + h), (x + w, y + h), self.colors['shadow_dark'], 1)
        cv2.line(canvas, (x + w, y), (x + w, y + h), self.colors['shadow_dark'], 1)
        
        # Title if provided
        if title:
            cv2.putText(canvas, title, (x + 15, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_primary'], 2)
    
    def _draw_face_bounding_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                              aoi_compliant: bool = False) -> None:
        """Draw modern face bounding box with neumorphism styling."""
        x, y, w, h = bbox
        
        # Choose color based on AOI compliance
        if aoi_compliant:
            color = self.colors['face_box_active']
            thickness = 3
        else:
            color = self.colors['face_box_inactive']
            thickness = 2
        
        # Draw rounded rectangle effect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Add subtle corner highlights
        corner_size = 15
        cv2.line(frame, (x, y), (x + corner_size, y), self.colors['shadow_light'], 1)
        cv2.line(frame, (x, y), (x, y + corner_size), self.colors['shadow_light'], 1)
        cv2.line(frame, (x + w, y), (x + w - corner_size, y), self.colors['shadow_light'], 1)
        cv2.line(frame, (x + w, y), (x + w, y + corner_size), self.colors['shadow_light'], 1)
        cv2.line(frame, (x, y + h), (x + corner_size, y + h), self.colors['shadow_dark'], 1)
        cv2.line(frame, (x, y + h), (x, y + h - corner_size), self.colors['shadow_dark'], 1)
        cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), self.colors['shadow_dark'], 1)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), self.colors['shadow_dark'], 1)
    
    def _draw_eye_circles(self, frame: np.ndarray, eyes: Dict[str, any]) -> None:
        """Draw modern eye circles with neumorphism styling."""
        if not eyes:
            return
        
        for eye_name, eye_detection in eyes.items():
            if eye_detection and hasattr(eye_detection, 'center'):
                center = eye_detection.center
                radius = 10
                
                # Draw outer circle (shadow)
                cv2.circle(frame, center, radius + 1, self.colors['shadow_dark'], 1)
                
                # Draw main circle
                cv2.circle(frame, center, radius, self.colors['eye_circle'], 2)
                
                # Draw inner highlight
                cv2.circle(frame, center, radius - 3, self.colors['shadow_light'], 1)
    
    def _draw_detection_stats(self, canvas: np.ndarray, persons: Dict[int, Dict]) -> None:
        """Draw modern detection statistics with neumorphism styling."""
        x, y, w, h = self.sections['detection_stats']
        
        # Draw card
        self._draw_neumorphism_card(canvas, (x, y, w, h), "DETECTION STATS")
        
        # Active detections
        active_count = len(persons)
        cv2.putText(canvas, f"Active: {active_count}", (x + 15, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_primary'], 2)
        
        # Total detections
        cv2.putText(canvas, f"Total: {self.total_detections:,}", (x + 15, y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_secondary'], 2)
        
        # Detection quality indicator
        if persons:
            avg_confidence = sum(p.get('face_detection', {}).confidence or 0 for p in persons.values()) / len(persons)
            quality_text = f"Quality: {avg_confidence:.1%}"
            quality_color = self.colors['success'] if avg_confidence > 0.7 else self.colors['warning']
            cv2.putText(canvas, quality_text, (x + 15, y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        else:
            cv2.putText(canvas, "Quality: --", (x + 15, y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_muted'], 2)
    
    def _draw_pose_table(self, canvas: np.ndarray, persons: Dict[int, Dict]) -> None:
        """Draw modern pose table with neumorphism styling."""
        x, y, w, h = self.sections['pose_table']
        
        # Draw card
        self._draw_neumorphism_card(canvas, (x, y, w, h), "POSE VALUES")
        
        if not persons:
            cv2.putText(canvas, "No persons detected", (x + 15, y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_muted'], 1)
            return
        
        # Table header with neumorphism styling
        header_y = y + 50
        header_height = 25
        
        # Header background
        cv2.rectangle(canvas, (x + 10, header_y - 5), (x + w - 10, header_y + header_height), 
                     self.colors['bg_secondary'], -1)
        
        # Header shadows
        cv2.line(canvas, (x + 10, header_y - 5), (x + w - 10, header_y - 5), self.colors['shadow_light'], 1)
        cv2.line(canvas, (x + 10, header_y - 5), (x + 10, header_y + header_height), self.colors['shadow_light'], 1)
        cv2.line(canvas, (x + 10, header_y + header_height), (x + w - 10, header_y + header_height), self.colors['shadow_dark'], 1)
        cv2.line(canvas, (x + w - 10, header_y - 5), (x + w - 10, header_y + header_height), self.colors['shadow_dark'], 1)
        
        # Header text
        cv2.putText(canvas, "ID", (x + 20, header_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Pitch", (x + 50, header_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Roll", (x + 100, header_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Yaw", (x + 150, header_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Conf", (x + 200, header_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
        cv2.putText(canvas, "AOI", (x + 250, header_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
        cv2.putText(canvas, "Stab", (x + 290, header_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
        
        # Table rows
        row_height = 20
        max_rows = min(len(persons), 8)
        
        for i, (person_id, person) in enumerate(list(persons.items())[:max_rows]):
            row_y = header_y + header_height + 5 + (i * row_height)
            
            # Row background with alternating colors
            if i % 2 == 0:
                row_color = self.colors['bg_card']
            else:
                row_color = self.colors['bg_secondary']
            
            cv2.rectangle(canvas, (x + 10, row_y - 2), (x + w - 10, row_y + row_height - 2), 
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
            cv2.putText(canvas, str(person_id), (x + 20, row_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_primary'], 1)
            cv2.putText(canvas, f"{pitch:.1f}°", (x + 50, row_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{roll:.1f}°", (x + 100, row_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{yaw:.1f}°", (x + 150, row_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            cv2.putText(canvas, f"{confidence:.2f}", (x + 200, row_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            
            # AOI compliance indicator
            aoi_color = self.colors['success'] if aoi_compliant else self.colors['error']
            aoi_text = "✓" if aoi_compliant else "✗"
            cv2.putText(canvas, aoi_text, (x + 250, row_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, aoi_color, 1)
            
            # Stability indicator
            stability_color = self.colors['success'] if stability > 0.7 else self.colors['warning']
            cv2.putText(canvas, f"{stability:.2f}", (x + 290, row_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, stability_color, 1)
    
    def _draw_aoi_status(self, canvas: np.ndarray) -> None:
        """Draw AOI status with neumorphism styling."""
        x, y, w, h = self.sections['aoi_status']
        
        # Draw card
        self._draw_neumorphism_card(canvas, (x, y, w, h), "AOI STATUS")
        
        # AOI status
        if self.aoi_reference:
            status_text = "AOI Active"
            status_color = self.colors['success']
            
            # Show reference values
            ref_text = f"Ref: P={self.aoi_reference['pitch']:.1f}° R={self.aoi_reference['roll']:.1f}° Y={self.aoi_reference['yaw']:.1f}°"
            cv2.putText(canvas, ref_text, (x + 15, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
        else:
            status_text = "AOI Inactive"
            status_color = self.colors['error']
        
        cv2.putText(canvas, status_text, (x + 15, y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    def _draw_footer(self, canvas: np.ndarray) -> None:
        """Draw modern footer with neumorphism styling."""
        x, y, w, h = self.sections['footer']
        
        # Footer background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['bg_secondary'], -1)
        
        # Footer shadows
        cv2.line(canvas, (x, y), (x + w, y), self.colors['shadow_light'], 1)
        cv2.line(canvas, (x, y), (x, y + h), self.colors['shadow_light'], 1)
        cv2.line(canvas, (x, y + h), (x + w, y + h), self.colors['shadow_dark'], 1)
        cv2.line(canvas, (x + w, y), (x + w, y + h), self.colors['shadow_dark'], 1)
        
        # Status text
        cv2.putText(canvas, "Detectron2 Face Detection Active", (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
    
    def handle_mouse_click(self, x: int, y: int) -> str:
        """Handle mouse clicks on UI elements."""
        # Check button clicks
        for button_name, button in self.buttons.items():
            bx, by, bw, bh = button['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                # Toggle button pressed state
                button['pressed'] = not button['pressed']
                return button_name
        return None
    
    def toggle_button(self, button_name: str) -> None:
        """Toggle button state."""
        if button_name in self.buttons:
            self.buttons[button_name]['active'] = not self.buttons[button_name]['active']
            if button_name == 'toggle_detection':
                text = 'Detection ON' if self.buttons[button_name]['active'] else 'Detection OFF'
                self.buttons[button_name]['text'] = text
    
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
        """Render the neumorphism UI with camera feed and side panel."""
        # Update animation time
        self.animation_time += 0.1
        self.pulse_animation = np.sin(self.animation_time) * 0.1 + 0.9
        
        # Check if we can use cached panel
        current_persons_count = len(persons)
        current_aoi_state = self.aoi_reference is not None
        
        # Create main canvas
        canvas = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
        canvas.fill(240)  # Light background
        
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
            panel.fill(235)  # Panel background
            
            # Draw UI sections
            self._draw_neumorphism_button(panel, 'reset_aoi')
            self._draw_neumorphism_button(panel, 'toggle_detection')
            self._draw_detection_stats(panel, persons)
            self._draw_pose_table(panel, persons)
            self._draw_aoi_status(panel)
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
    print("Modern Neumorphism UI loaded successfully!")
    print("Features:")
    print("- Clean, modern neumorphism design")
    print("- Smooth animations and transitions")
    print("- Enhanced visual hierarchy")
    print("- Improved user experience")
