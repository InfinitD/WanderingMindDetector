"""
Simple test for eye detector module
"""

import sys
import os
sys.path.append('../src')

from eye_detector import EyeDetector

def test_eye_detector():
    """Test eye detector initialization"""
    detector = EyeDetector()
    print("Eye detector initialized successfully!")
    print(f"Face cascade loaded: {not detector.face_cascade.empty()}")
    print(f"Eye cascade loaded: {not detector.eye_cascade.empty()}")

if __name__ == "__main__":
    test_eye_detector()
