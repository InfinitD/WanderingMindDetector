# Cursey Documentation

Welcome to the Cursey documentation! This guide will help you understand, install, and use the Cursey multi-person face and eye tracking system.

## 📚 Table of Contents

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [User Guide](user-guide.md)
- [API Reference](api-reference.md)
- [Development Guide](development.md)
- [Troubleshooting](troubleshooting.md)
- [Contributing](contributing.md)

## 🚀 Quick Links

- **Installation**: [Install Detectron2 and dependencies](installation.md#installation)
- **Basic Usage**: [Run your first detection](quickstart.md#basic-usage)
- **API Reference**: [Complete API documentation](api-reference.md)
- **Examples**: [Code examples and tutorials](examples/)

## 🎯 What is Cursey?

Cursey is a state-of-the-art multi-person face and eye tracking system that combines:

- **Facebook Detectron2** for cutting-edge object detection
- **Modern Neumorphism UI** for an intuitive user experience
- **Real-time Performance** optimized for 30+ FPS
- **Multi-person Support** for tracking 2-3 people simultaneously
- **Advanced Pose Estimation** with 3-DOF head pose tracking

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Input  │───▶│  Detectron2      │───▶│  Neumorphism UI │
│                 │    │  Detection       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Pose Estimation │
                       │  & Eye Tracking  │
                       └──────────────────┘
```

## 📦 Package Structure

```
cursey/
├── detectors/          # Detection algorithms
│   ├── detectron_detector.py
│   ├── enhanced_detector.py
│   ├── high_performance_detector.py
│   └── ...
├── ui/                 # User interface components
│   ├── neumorphism_ui.py
│   ├── enhanced_ui.py
│   └── ...
├── utils/              # Utilities and constants
│   ├── constants.py
│   └── ...
└── __init__.py
```

## 🔧 Key Features

### Detection Capabilities
- **State-of-the-art Detection**: Facebook Detectron2 integration
- **Multi-person Tracking**: Support for 2-3 people simultaneously
- **Robust Eye Tracking**: Handles occlusion and various lighting conditions
- **Pose Estimation**: 3 degrees of freedom (pitch, yaw, roll)
- **Gaze Analysis**: Real-time gaze direction estimation

### User Interface
- **Modern Design**: Neumorphism UI with depth and shadows
- **Real-time Statistics**: Live performance monitoring
- **Interactive Controls**: Intuitive button interactions
- **Responsive Layout**: Optimized for different screen sizes

### Performance
- **Real-time Processing**: 30+ FPS target performance
- **GPU Acceleration**: CUDA support with CPU fallback
- **Memory Optimization**: Efficient resource management
- **Cross-platform**: Windows, macOS, and Linux support

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   python install_detectron2.py
   ```

2. **Run the Application**
   ```bash
   python examples/detectron_main_app.py
   ```

3. **Explore Examples**
   ```bash
   # Basic usage
   python examples/detectron_main_app.py
   
   # Legacy version
   python examples/main_app.py
   ```

## 📖 Documentation Sections

### Installation
Complete installation guide including Detectron2 setup, dependencies, and troubleshooting.

### Quick Start
Get up and running quickly with basic examples and common use cases.

### User Guide
Comprehensive guide covering all features, controls, and advanced usage.

### API Reference
Complete API documentation for all classes, methods, and parameters.

### Development Guide
Information for developers including architecture, contributing, and extending the system.

### Troubleshooting
Common issues, solutions, and performance optimization tips.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Facebook AI Research for Detectron2
- OpenCV community for computer vision tools
- Contributors and users of the Cursey project

---

For more information, visit our [GitHub repository](https://github.com/cursey-team/cursey) or [open an issue](https://github.com/cursey-team/cursey/issues) for support.
