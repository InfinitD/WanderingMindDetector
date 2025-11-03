"""
Cursey package (legacy namespace).

This package currently only exposes versioning and constants for backward compatibility.
All runnable code lives in `simple_app.py` and modularized sources under `src/`.
"""

__version__ = "2.1.0"
__author__ = "Cursey Development Team"
__license__ = "MIT"
__description__ = "Wandering Mind Detector - lightweight face/pose/gaze demo"

# Optional constants (available if module layout is present)
try:
    from .utils.constants import AppConstants, UIConstants, DetectionConstants
except Exception:
    AppConstants = None
    UIConstants = None
    DetectionConstants = None

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    "AppConstants",
    "UIConstants",
    "DetectionConstants",
]
