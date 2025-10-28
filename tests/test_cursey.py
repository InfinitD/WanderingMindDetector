#!/usr/bin/env python3
"""
Comprehensive Test Suite for Cursey
Tests all components and provides detailed feedback

Author: Cursey Team
Date: 2025
"""

import sys
import os
import time
import logging
import traceback
from typing import Dict, Any
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDetectronDetector(unittest.TestCase):
    """Test DetectronFaceDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from cursey.detectors.detectron_detector import DetectronFaceDetector
            self.detector = DetectronFaceDetector(max_persons=3, use_gpu=False)
        except ImportError as e:
            self.skipTest(f"Detectron2 not available: {e}")
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.max_persons, 3)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
    
    def test_performance_stats(self):
        """Test performance statistics."""
        stats = self.detector.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_detections', stats)
        self.assertIn('detection_time', stats)
    
    def test_dof_limits(self):
        """Test DOF limits."""
        limits = self.detector.get_dof_limits()
        self.assertIsInstance(limits, dict)
        self.assertIn('pitch', limits)
        self.assertIn('yaw', limits)
        self.assertIn('roll', limits)
    
    def test_frame_processing(self):
        """Test frame processing with dummy data."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.process_frame(dummy_frame)
        
        self.assertIsInstance(result, dict)
        self.assertIn('persons', result)
        self.assertIn('frame', result)
        self.assertIn('detection_count', result)
        self.assertIn('performance_stats', result)
        self.assertIsInstance(result['persons'], dict)


class TestNeumorphismUI(unittest.TestCase):
    """Test NeumorphismUI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from cursey.ui.neumorphism_ui import NeumorphismUI
        self.ui = NeumorphismUI()
    
    def test_initialization(self):
        """Test UI initialization."""
        self.assertEqual(self.ui.camera_width, 640)
        self.assertEqual(self.ui.camera_height, 480)
        self.assertEqual(self.ui.panel_width, 380)
    
    def test_color_scheme(self):
        """Test color scheme."""
        self.assertIsInstance(self.ui.colors, dict)
        self.assertIn('bg_primary', self.ui.colors)
        self.assertIn('text_primary', self.ui.colors)
        self.assertIn('shadow_dark', self.ui.colors)
        self.assertIn('shadow_light', self.ui.colors)
    
    def test_button_definitions(self):
        """Test button definitions."""
        self.assertIsInstance(self.ui.buttons, dict)
        self.assertIn('reset_aoi', self.ui.buttons)
        self.assertIn('toggle_detection', self.ui.buttons)
    
    def test_sections(self):
        """Test UI sections."""
        self.assertIsInstance(self.ui.sections, dict)
        self.assertIn('header', self.ui.sections)
        self.assertIn('pose_table', self.ui.sections)
        self.assertIn('aoi_status', self.ui.sections)
    
    def test_rendering(self):
        """Test UI rendering."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_persons = {}
        
        rendered_frame = self.ui.render(dummy_frame, dummy_persons)
        
        self.assertIsInstance(rendered_frame, np.ndarray)
        self.assertEqual(rendered_frame.shape[0], self.ui.total_height)
        self.assertEqual(rendered_frame.shape[1], self.ui.total_width)
        self.assertEqual(rendered_frame.shape[2], 3)


class TestEnhancedDetector(unittest.TestCase):
    """Test EnhancedFaceDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from cursey.detectors.enhanced_detector import EnhancedFaceDetector
        self.detector = EnhancedFaceDetector(max_persons=3)
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.max_persons, 3)
        self.assertEqual(self.detector.confidence_threshold, 0.6)
    
    def test_performance_stats(self):
        """Test performance statistics."""
        stats = self.detector.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_detections', stats)
    
    def test_frame_processing(self):
        """Test frame processing."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.process_frame(dummy_frame)
        
        self.assertIsInstance(result, dict)
        self.assertIn('persons', result)
        self.assertIn('detection_count', result)


class TestHighPerformanceDetector(unittest.TestCase):
    """Test HighPerformanceDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from cursey.detectors.high_performance_detector import HighPerformanceDetector
        self.detector = HighPerformanceDetector(max_persons=3)
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.max_persons, 3)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
    
    def test_performance_stats(self):
        """Test performance statistics."""
        stats = self.detector.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('fps', stats)
    
    def test_frame_processing(self):
        """Test frame processing."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.process_frame(dummy_frame)
        
        self.assertIsInstance(result, dict)
        self.assertIn('persons', result)
        self.assertIn('performance_stats', result)


class TestConstants(unittest.TestCase):
    """Test constants module."""
    
    def test_app_constants(self):
        """Test AppConstants."""
        from cursey.utils.constants import AppConstants
        
        self.assertIsInstance(AppConstants.MIN_PERSONS, int)
        self.assertIsInstance(AppConstants.MAX_PERSONS, int)
        self.assertIsInstance(AppConstants.DEFAULT_MAX_PERSONS, int)
        self.assertGreater(AppConstants.MAX_PERSONS, AppConstants.MIN_PERSONS)
    
    def test_ui_constants(self):
        """Test UIConstants."""
        from cursey.utils.constants import UIConstants
        
        self.assertIsInstance(UIConstants.BACKGROUND_COLOR, tuple)
        self.assertIsInstance(UIConstants.TEXT_PRIMARY, tuple)
        self.assertEqual(len(UIConstants.BACKGROUND_COLOR), 3)
    
    def test_detection_constants(self):
        """Test DetectionConstants."""
        from cursey.utils.constants import DetectionConstants
        
        self.assertIsInstance(DetectionConstants.FACE_SCALE_FACTOR, float)
        self.assertIsInstance(DetectionConstants.FACE_MIN_NEIGHBORS, int)
        self.assertIsInstance(DetectionConstants.FACE_MIN_SIZE, tuple)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_detectron_ui_integration(self):
        """Test Detectron2 and UI integration."""
        try:
            from cursey.detectors.detectron_detector import DetectronFaceDetector
            from cursey.ui.neumorphism_ui import NeumorphismUI
            
            detector = DetectronFaceDetector(max_persons=2, use_gpu=False)
            ui = NeumorphismUI()
            
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.process_frame(dummy_frame)
            
            rendered_frame = ui.render(dummy_frame, result['persons'])
            
            self.assertIsInstance(rendered_frame, np.ndarray)
            self.assertEqual(rendered_frame.shape[0], ui.total_height)
            self.assertEqual(rendered_frame.shape[1], ui.total_width)
            
        except ImportError as e:
            self.skipTest(f"Integration test skipped: {e}")


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_processing_speed(self):
        """Test processing speed."""
        try:
            from cursey.detectors.detectron_detector import DetectronFaceDetector
            
            detector = DetectronFaceDetector(max_persons=3, use_gpu=False)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            result = detector.process_frame(dummy_frame)
            processing_time = time.time() - start_time
            
            # Should process quickly even with dummy data
            self.assertLess(processing_time, 1.0, f"Processing too slow: {processing_time:.3f}s")
            
        except ImportError as e:
            self.skipTest(f"Performance test skipped: {e}")


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDetectronDetector,
        TestNeumorphismUI,
        TestEnhancedDetector,
        TestHighPerformanceDetector,
        TestConstants,
        TestIntegration,
        TestPerformance,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if total_tests > 0:
        success_rate = (passed / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🎉 Overall Result: EXCELLENT - Ready for production!")
        elif success_rate >= 60:
            print("\n✅ Overall Result: GOOD - Minor issues to address")
        else:
            print("\n⚠️ Overall Result: NEEDS WORK - Major issues to fix")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
