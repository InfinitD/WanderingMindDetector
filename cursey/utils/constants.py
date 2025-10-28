"""
Constants and configuration values for Cursey application.
Extracted magic numbers and hardcoded values for better maintainability.

Author: Cursey Team
Date: 2025
"""

# UI Constants
class UIConstants:
    """UI-related constants."""
    
    # Background colors (BGR format)
    BACKGROUND_COLOR = (18, 18, 18)
    PANEL_BACKGROUND_COLOR = (25, 25, 25)
    BG_SECONDARY = (28, 28, 28)
    BG_TERTIARY = (38, 38, 38)
    
    # Text colors (BGR format)
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (200, 200, 200)
    TEXT_MUTED = (150, 150, 150)
    TEXT_ACCENT = (100, 200, 255)
    
    # Status colors (BGR format)
    SUCCESS = (76, 175, 80)
    WARNING = (255, 193, 7)
    ERROR = (244, 67, 54)
    INFO = (33, 150, 243)
    
    # UI element colors
    BUTTON_BG = (45, 45, 45)
    BUTTON_HOVER = (55, 55, 55)
    BUTTON_ACTIVE = (65, 65, 65)
    BORDER = (60, 60, 60)
    ACCENT = (0, 150, 255)
    
    # Status bar colors
    STATUS_BG = (30, 30, 30)
    STATUS_FILL = (0, 150, 255)
    STATUS_SUCCESS = (76, 175, 80)
    STATUS_WARNING = (255, 193, 7)
    STATUS_ERROR = (244, 67, 54)
    
    # Font settings
    FONT_SCALE_TITLE = 0.7
    FONT_SCALE_SUBTITLE = 0.6
    FONT_SCALE_NORMAL = 0.5
    FONT_SCALE_SMALL = 0.4
    FONT_SCALE_LARGE = 1.2
    FONT_THICKNESS_NORMAL = 1
    FONT_THICKNESS_BOLD = 2
    FONT_THICKNESS_EXTRA_BOLD = 3
    
    # Layout dimensions
    PANEL_WIDTH = 350
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    TOTAL_HEIGHT = 700
    
    # Button dimensions
    BUTTON_HEIGHT = 35
    BUTTON_SPACING = 10
    
    # Animation settings
    PERSON_COUNTER_ANIMATION_SPEED = 0.1
    MAX_ANIMATION_HISTORY = 5


# Detection Constants
class DetectionConstants:
    """Face and eye detection related constants."""
    
    # Face detection parameters
    FACE_SCALE_FACTOR = 1.1
    FACE_MIN_NEIGHBORS = 8
    FACE_MIN_SIZE = (40, 40)
    
    # Eye detection parameters
    EYE_SCALE_FACTOR = 1.1
    EYE_MIN_NEIGHBORS = 5
    EYE_MIN_SIZE = (15, 15)
    
    # Adaptive detection parameters
    DARK_SCALE_FACTOR = 1.05
    DARK_MIN_NEIGHBORS = 3
    DARK_MIN_SIZE = (20, 20)
    
    BRIGHT_SCALE_FACTOR = 1.15
    BRIGHT_MIN_NEIGHBORS = 7
    BRIGHT_MIN_SIZE = (40, 40)
    
    LOW_CONTRAST_SCALE_FACTOR = 1.1
    LOW_CONTRAST_MIN_NEIGHBORS = 4
    LOW_CONTRAST_MIN_SIZE = (25, 25)
    
    # Lighting thresholds
    DARK_BRIGHTNESS_THRESHOLD = 80
    BRIGHT_BRIGHTNESS_THRESHOLD = 200
    LOW_CONTRAST_STD_THRESHOLD = 30
    
    # Duplicate detection
    OVERLAP_THRESHOLD = 0.3
    PERSON_ASSIGNMENT_DISTANCE_THRESHOLD = 100
    
    # Tracking
    MAX_PERSON_AGE = 2.0  # seconds
    MAX_TRACKING_HISTORY = 30


# Camera Constants
class CameraConstants:
    """Camera and video related constants."""
    
    # Camera properties
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_FRAME_WIDTH = 640
    DEFAULT_FRAME_HEIGHT = 480
    DEFAULT_FPS = 30
    
    # Camera settings
    AUTO_EXPOSURE_VALUE = 0.25
    EXPOSURE_VALUE = -6
    
    # Frame processing
    HORIZONTAL_FLIP = 1


# Application Constants
class AppConstants:
    """Application-level constants."""
    
    # Person limits
    MIN_PERSONS = 1
    MAX_PERSONS = 5
    DEFAULT_MAX_PERSONS = 3
    
    # FPS calculation
    FPS_CALCULATION_INTERVAL = 1.0
    
    # Session tracking
    QUALITY_CALCULATION_FRAMES = 10
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4


# Mannequin Constants
class MannequinConstants:
    """Geometric mannequin visualization constants."""
    
    # Proportions (normalized)
    HEAD_RADIUS = 0.08
    NECK_LENGTH = 0.05
    TORSO_WIDTH = 0.12
    TORSO_HEIGHT = 0.25
    ARM_LENGTH = 0.15
    LEG_LENGTH = 0.3
    SHOULDER_WIDTH = 0.2
    
    # Colors (BGR format)
    HEAD_COLOR = (100, 150, 255)
    NECK_COLOR = (80, 120, 200)
    TORSO_COLOR = (60, 100, 180)
    ARMS_COLOR = (40, 80, 160)
    LEGS_COLOR = (20, 60, 140)
    JOINTS_COLOR = (255, 255, 255)
    GAZE_LINE_COLOR = (0, 255, 255)
    POSE_AXES_COLOR = (255, 0, 0)
    
    # Animation
    SMOOTHING_FACTOR = 0.3
    GAZE_LENGTH_MULTIPLIER = 1.5
    
    # Layout
    DEFAULT_WIDTH = 200
    DEFAULT_HEIGHT = 300
    MAX_PERSONS_DISPLAY = 6
