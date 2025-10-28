"""
UI components package for Cursey user interfaces.
"""

# Optional Neumorphism UI import
try:
    from .neumorphism_ui import NeumorphismUI
    NEUMORPHISM_AVAILABLE = True
except ImportError:
    NeumorphismUI = None
    NEUMORPHISM_AVAILABLE = False

from .enhanced_ui import EnhancedMinimalUI
from .minimal_ui import MinimalUI

__all__ = [
    "NeumorphismUI",
    "EnhancedMinimalUI",
    "MinimalUI",
]
