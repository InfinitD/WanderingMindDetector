"""
Enhanced Main Application - wandering-mind-detector Face Detection
Improved stability, accuracy, and AOI functionality
"""

import cv2
import numpy as np
import sys
import time
import logging
from typing import Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cursey.detectors.high_performance_detector import HighPerformanceDetector
from cursey.ui.enhanced_ui import EnhancedMinimalUI
from cursey.utils.constants import AppConstants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WanderingMindApp:
    """Enhanced wandering-mind-detector application."""
    
    def __init__(self, camera_index: int = 0, max_persons: int = 3):
        # Input validation
        if not isinstance(camera_index, int) or camera_index < 0:
            raise ValueError("Camera index must be non-negative integer")
        if not isinstance(max_persons, int) or not (AppConstants.MIN_PERSONS <= max_persons <= AppConstants.MAX_PERSONS):
            raise ValueError(f"Max persons must be between {AppConstants.MIN_PERSONS} and {AppConstants.MAX_PERSONS}")
        
        self.camera_index = camera_index
        self.max_persons = max_persons
        self.cap = None
        
        # Initialize components
        self.detector = HighPerformanceDetector(max_persons=max_persons)
        self.ui = EnhancedMinimalUI()
        
        # Application state
        self.running = False
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Performance optimization
        self.ui_update_frequency = 3  # Update UI every 3rd frame
        self.frame_count = 0
        
    def initialize_camera(self) -> bool:
        """Initialize camera with optimal settings."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set optimal camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
            
            logger.info("Camera initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for UI interaction."""
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.debug(f"Mouse click at: ({x}, {y})")
            # Check if click is in UI panel area
            if x >= self.ui.camera_width:
                # Convert to panel coordinates
                panel_x = x - self.ui.camera_width
                panel_y = y
                logger.debug(f"Panel coordinates: ({panel_x}, {panel_y})")
                
                # Handle UI interaction
                button_clicked = self.ui.handle_mouse_click(panel_x, panel_y)
                if button_clicked:
                    logger.info(f"Button clicked: {button_clicked}")
                    self.handle_button_click(button_clicked)
    
    def handle_button_click(self, button_name: str):
        """Handle button click events."""
        if button_name == 'reset_aoi':
            # Get current pose from first detected person
            persons = self.detector.tracking_persons
            if persons:
                for person_data in persons.values():
                    if 'face_detection' in person_data and person_data['face_detection'].pose_angles:
                        pose_angles = person_data['face_detection'].pose_angles
                        self.detector.set_aoi_reference(pose_angles)
                        self.ui.set_aoi_reference(pose_angles)
                        logger.info(f"AOI reference set: {pose_angles}")
                        break
            else:
                logger.warning("No person detected to set AOI reference")
    
    def calculate_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
            return fps
        
        return 0.0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return UI-rendered output (optimized)."""
        try:
            # Process frame with high-performance detector
            result = self.detector.process_frame(frame)
            persons = result['persons']
            
            # Update UI detection count
            self.ui.total_detections = result['detection_count']
            
            # Update frame counter
            self.frame_count += 1
            
            # Only update UI every few frames for performance
            if self.frame_count % self.ui_update_frequency == 0:
                return self.ui.render(frame, persons)
            else:
                # Return minimal rendering for performance
                canvas = np.zeros((self.ui.total_height, self.ui.total_width, 3), dtype=np.uint8)
                canvas.fill(20)
                
                # Place camera feed
                if frame is not None:
                    resized_camera = cv2.resize(frame, (self.ui.camera_width, self.ui.camera_height))
                    canvas[:self.ui.camera_height, :self.ui.camera_width] = resized_camera
                    
                    # Draw face bounding boxes and eye circles on camera feed
                    for person in persons.values():
                        if 'face_detection' in person:
                            face_det = person['face_detection']
                            aoi_compliant = person.get('aoi_compliant', False)
                            self.ui._draw_face_bounding_box(canvas, face_det.bbox, aoi_compliant)
                        
                        if 'eyes' in person:
                            self.ui._draw_eye_circles(canvas, person['eyes'])
                
                # Use cached panel if available
                if self.ui.render_cache is not None:
                    canvas[:, self.ui.camera_width:] = self.ui.render_cache
                
                return canvas
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            # Return frame with error message
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Error: {str(e)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return self.ui.render(error_frame, {})
    
    def run(self):
        """Main application loop."""
        logger.info("wandering-mind-detector Face Detection System")
        logger.info("=" * 40)
        
        # Initialize camera
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return
        
        # Set up window with mouse callback
        cv2.namedWindow('wandering-mind-detector Face Detection', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('wandering-mind-detector Face Detection', self.mouse_callback)
        
        logger.info("\nControls:")
        logger.info("- Press 'q' to quit")
        logger.info("- Press 'r' to reset session")
        logger.info("- Click 'Reset AOI' button to set Area of Interest")
        logger.info("\nStarting application...")
        
        self.running = True
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                output_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('wandering-mind-detector Face Detection', output_frame)
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                if fps > 0:
                    print(f"\rFPS: {fps:.1f} | Detections: {self.ui.total_detections}", end="", flush=True)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset session
                    self.ui.session_start = time.time()
                    self.ui.frame_count = 0
                    self.ui.total_detections = 0
                    self.ui.reset_aoi()
                    logger.info("Session reset")
        
        except KeyboardInterrupt:
            logger.warning("Application interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("\nApplication closed successfully")


def main():
    """Main entry point."""
    # Parse command line arguments
    camera_index = 0
    max_persons = AppConstants.DEFAULT_MAX_PERSONS
    
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            logger.warning("Invalid camera index. Using default camera 0.")
    
    if len(sys.argv) > 2:
        try:
            max_persons = int(sys.argv[2])
            max_persons = max(AppConstants.MIN_PERSONS, min(max_persons, AppConstants.MAX_PERSONS))
        except ValueError:
            logger.warning(f"Invalid max persons. Using default {AppConstants.DEFAULT_MAX_PERSONS}.")
    
    logger.info(f"Starting with camera {camera_index}, max persons: {max_persons}")
    
    # Create and run application
    try:
        app = WanderingMindApp(camera_index=camera_index, max_persons=max_persons)
        app.run()
    except ValueError as e:
        logger.error(f"Invalid parameters: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
