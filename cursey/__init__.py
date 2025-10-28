"""
Cursey - Multi-Person Face & Eye Tracking System

A state-of-the-art multi-person face and eye tracking system with Facebook Detectron2 
integration and modern neumorphism UI, designed for real-time monitoring and analysis.

Author: Cursey Development Team
License: MIT
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Cursey Development Team"
__license__ = "MIT"
__description__ = "Multi-Person Face & Eye Tracking System with Detectron2"

# Import main components
from .detectors.detectron_detector import DetectronFaceDetector
from .detectors.enhanced_detector import EnhancedFaceDetector
from .detectors.high_performance_detector import HighPerformanceDetector
from .detectors.yolo_face_detector import YOLOFaceDetector
from .detectors.eye_detector import EyeDetector
from .detectors.gaze_analyzer import GazeAnalyzer

from .ui.neumorphism_ui import NeumorphismUI
from .ui.enhanced_ui import EnhancedMinimalUI
from .ui.minimal_ui import MinimalUI

from .utils.constants import AppConstants, UIConstants, DetectionConstants

__all__ = [
    # Detectors
    "DetectronFaceDetector",
    "EnhancedFaceDetector", 
    "HighPerformanceDetector",
    "YOLOFaceDetector",
    "EyeDetector",
    "GazeAnalyzer",
    
    # UI Components
    "NeumorphismUI",
    "EnhancedMinimalUI",
    "MinimalUI",
    
    # Utilities
    "AppConstants",
    "UIConstants", 
    "DetectionConstants",
    
    # Package info
    "__version__",
    "__author__",
    "__license__",
    "__description__",
]
