"""
Detectors package for Cursey face and eye detection systems.
"""

# Optional Detectron2 import
try:
    from .detectron_detector import DetectronFaceDetector
    DETECTRON_AVAILABLE = True
except ImportError:
    DetectronFaceDetector = None
    DETECTRON_AVAILABLE = False

from .enhanced_detector import EnhancedFaceDetector
from .high_performance_detector import HighPerformanceDetector
from .yolo_face_detector import YOLOFaceDetector
from .eye_detector import EyeDetector
from .gaze_analyzer import GazeAnalyzer

__all__ = [
    "DetectronFaceDetector",
    "EnhancedFaceDetector",
    "HighPerformanceDetector", 
    "YOLOFaceDetector",
    "EyeDetector",
    "GazeAnalyzer",
]
