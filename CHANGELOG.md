# Changelog

All notable changes to Cursey will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-XX

### Added
- **Facebook Detectron2 Integration**: State-of-the-art object detection with instance segmentation
- **Modern Neumorphism UI**: Clean, modern interface with depth and shadows
- **GPU Acceleration**: CUDA support with automatic CPU fallback
- **Enhanced Pose Estimation**: 3-DOF head pose with attention vectors
- **Interactive Controls**: Modern buttons with tactile feedback
- **Comprehensive Statistics**: Real-time performance monitoring
- **Automated Installation**: `install_detectron2.py` script for easy setup
- **Advanced Error Handling**: Graceful fallbacks and comprehensive error reporting
- **Performance Optimization**: Optimized for 30+ FPS real-time processing

### Changed
- **Architecture**: Complete redesign with modular components
- **UI Design**: Modernized interface with neumorphism principles
- **Detection Pipeline**: Enhanced with Detectron2 and multiple fallback options
- **Documentation**: Comprehensive README and contributing guidelines
- **Project Structure**: Organized codebase with clear separation of concerns

### Fixed
- **Performance Issues**: Optimized detection algorithms for better FPS
- **UI Responsiveness**: Improved interface responsiveness and visual feedback
- **Error Handling**: Better error recovery and user feedback
- **Memory Management**: Optimized memory usage for real-time processing

### Removed
- **Legacy Documentation**: Cleaned up outdated documentation files
- **Deprecated Components**: Removed unused legacy code
- **Redundant Files**: Streamlined project structure

## [1.0.0] - 2024-XX-XX

### Added
- **Multi-person Face Detection**: Support for 2-3 people simultaneously
- **Eye Tracking**: Robust eye detection with occlusion handling
- **Head Pose Estimation**: 3 degrees of freedom (pitch, yaw, roll)
- **Gaze Analysis**: 2 degrees of freedom gaze direction
- **Real-time Visualization**: Live camera feed with detection overlays
- **Session Tracking**: Duration, frames, detections, and FPS monitoring
- **Adaptive Detection**: Multi-scale detection for various lighting conditions
- **Temporal Smoothing**: Stable tracking with reduced jitter
- **Quality Metrics**: Confidence scoring and stability assessment

### Technical Details
- **OpenCV Integration**: Haar cascade classifiers for face and eye detection
- **Multi-threading**: Optimized processing pipeline
- **Camera Management**: Automatic camera initialization and configuration
- **Cross-platform Support**: Windows, macOS, and Linux compatibility

## [Unreleased]

### Planned Features
- **Web Interface**: Browser-based UI for remote access
- **Mobile Support**: iOS and Android applications
- **Cloud Integration**: Remote processing and data storage
- **Advanced Analytics**: Detailed gaze pattern analysis
- **API Support**: RESTful API for integration with other systems
- **Plugin System**: Extensible architecture for custom detectors
- **Machine Learning**: Custom trained models for specific use cases

### Known Issues
- Detectron2 model download may take time on first run
- GPU memory requirements may be high for some systems
- Camera permissions required for full functionality

---

## Version Numbering

- **Major** (X.0.0): Breaking changes or major feature additions
- **Minor** (0.X.0): New features or significant improvements
- **Patch** (0.0.X): Bug fixes and minor improvements

## Release Process

1. Update version numbers in relevant files
2. Update CHANGELOG.md with new version
3. Create release branch
4. Run comprehensive tests
5. Create GitHub release with release notes
6. Update documentation if needed
