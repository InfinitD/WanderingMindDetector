# Cursey - Multi-Person Face & Eye Tracking System

A state-of-the-art multi-person face and eye tracking system with Facebook Detectron2 integration and modern neumorphism UI, designed for real-time monitoring and analysis.

## 🚀 Features

### SOTA Detection Algorithm
- **Facebook Detectron2** integration for state-of-the-art object detection
- **Multi-person tracking** (2-3 people optimal performance)
- **Adaptive detection** for various lighting conditions
- **Robust eye tracking** with occlusion handling
- **Pose-invariant detection** for different face angles
- **Auto-scaling features** for optimal performance
- **Temporal smoothing** for stable tracking
- **GPU acceleration** support for real-time performance

### Modern Neumorphism UI
- **Clean, modern interface** with neumorphism design principles
- **Smooth animations** and visual transitions
- **Enhanced visual hierarchy** with depth and shadows
- **Interactive buttons** with tactile feedback
- **Real-time statistics** with modern card layouts
- **Status indicators** with intuitive color coding
- **Responsive design** optimized for different screen sizes

### Abstract Geometric Visualization
- **Real-time mannequin widgets** with 3D pose representation
- **Head pose mapping** (pitch, yaw, roll degrees of freedom)
- **Gaze direction visualization** with angle indicators
- **Multi-person grid layout** for simultaneous tracking
- **Smooth animation** with interpolation
- **Color-coded elements** for intuitive understanding

### Mini User Interface
- **Side panel layout** with camera feed and controls
- **Session tracking** (duration, frames, detections, FPS)
- **Real-time values** display for each person
- **Interactive controls** (start/stop, reset, toggle displays)
- **Status indicators** (tracking quality, system health)
- **Performance monitoring** with FPS and quality metrics

## 📁 Project Structure

```
Cursey/
├── src/
│   ├── detectron_main_app.py       # Main application with Detectron2
│   ├── detectron_detector.py       # Facebook Detectron2 integration
│   ├── neumorphism_ui.py          # Modern neumorphism UI
│   ├── main_app.py                # Enhanced main application
│   ├── enhanced_ui.py             # Enhanced UI with AOI system
│   ├── enhanced_detector.py       # Enhanced detection system
│   ├── high_performance_detector.py # High-performance detector
│   ├── yolo_face_detector.py      # YOLO-style detector
│   ├── eye_detector.py            # Eye detection module
│   ├── gaze_analyzer.py           # Gaze analysis algorithms
│   └── constants.py               # Application constants
├── models/                         # Haar cascade classifiers
├── requirements.txt               # Python dependencies
├── install_detectron2.py          # Detectron2 installation script
├── README.md                      # This file
└── COMPREHENSIVE_REDESIGN.md      # Detailed implementation guide
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Camera access permissions
- Adequate lighting for face detection
- CUDA support (recommended for Detectron2)

### Quick Setup
```bash
# Clone or navigate to Cursey directory
cd Cursey

# Install Detectron2 and dependencies (recommended)
python install_detectron2.py

# Or install manually
pip install -r requirements.txt

# Run the Detectron2 application
cd src
python detectron_main_app.py
```

### Advanced Usage
```bash
# Specify camera index, max persons, and GPU usage
python detectron_main_app.py 0 3 true

# Camera 0, max 3 people, GPU enabled
python detectron_main_app.py 1 2 false

# Camera 1, max 2 people, CPU only
```

## 🎮 Controls

### Mouse Controls
- Click buttons in the side panel for:
  - **Reset AOI**: Set Area of Interest reference
  - **Detection ON/OFF**: Toggle detection system
  - **Interactive elements**: Modern neumorphism buttons

### Keyboard Shortcuts
- `q` - Quit application
- `r` - Reset session
- `d` - Toggle detection on/off

## 📊 Technical Specifications

### Performance
- **Max Persons**: 2-3 (optimal performance)
- **Target FPS**: 30+ FPS with Detectron2
- **Resolution**: 640x480 (camera), 380px side panel
- **Memory**: Optimized for real-time processing
- **GPU**: CUDA acceleration support
- **CPU**: Multi-threaded detection pipeline

### Detection Capabilities
- **Face Detection**: Detectron2 with fallback to OpenCV
- **Eye Tracking**: Robust with occlusion handling
- **Head Pose**: 3 degrees of freedom (pitch, yaw, roll)
- **Gaze Angles**: 2 degrees of freedom (horizontal, vertical)
- **Lighting Adaptation**: Dark, bright, low-contrast, normal conditions
- **Instance Segmentation**: Detectron2 instance-level detection

### Visualization Features
- **Neumorphism Design**: Modern UI with depth and shadows
- **Real-time Animation**: Smooth interpolation and movement
- **Color Coding**: Intuitive element identification
- **Multi-person Support**: Grid layout for multiple detections
- **Pose Information**: Real-time angle displays
- **Quality Metrics**: Confidence and stability scores

## 🔧 Architecture

### Core Components

1. **DetectronFaceDetector** (`detectron_detector.py`)
   - Facebook Detectron2 integration
   - State-of-the-art object detection
   - Instance segmentation capabilities
   - GPU acceleration support
   - Fallback to OpenCV when needed

2. **NeumorphismUI** (`neumorphism_ui.py`)
   - Modern neumorphism design
   - Clean, intuitive interface
   - Smooth animations
   - Enhanced visual hierarchy
   - Interactive controls

3. **CurseyDetectronApp** (`detectron_main_app.py`)
   - Main application integration
   - Camera management and optimization
   - Event handling and user interaction
   - Performance monitoring and error handling
   - Detectron2 and UI integration

4. **Enhanced Components** (legacy support)
   - EnhancedDetector for advanced detection
   - HighPerformanceDetector for 60 FPS
   - YOLOFaceDetector for YOLO-style detection

## 📈 Use Cases

- **Research**: Multi-person gaze tracking studies with SOTA detection
- **Education**: Attention monitoring in classrooms
- **Accessibility**: Eye-controlled interfaces
- **Psychology**: Behavioral analysis and research
- **Human-Computer Interaction**: Gaze-based interaction systems
- **Computer Vision**: Object detection and instance segmentation research

## 🚀 Getting Started

1. **Installation**:
   ```bash
   python install_detectron2.py
   ```

2. **Basic Usage**:
   ```bash
   cd src
   python detectron_main_app.py
   ```

3. **First Run**:
   - Grant camera permissions when prompted
   - Detection starts automatically
   - Position yourself in front of the camera
   - Observe the modern UI responding to your movements

4. **Multi-Person**:
   - Have 2-3 people in front of the camera
   - Each person gets tracked with Detectron2
   - Real-time values display for each person
   - Session statistics track all detections

## 🔍 Troubleshooting

### Common Issues
- **Detectron2 installation fails**: Run `python install_detectron2.py` for automated setup
- **CUDA not available**: Detectron2 will fallback to CPU mode automatically
- **Camera not detected**: Check camera permissions and connections
- **Poor tracking quality**: Ensure adequate lighting
- **Low FPS**: Close other applications, check system resources

### Performance Tips
- Use good lighting conditions
- Position camera at eye level
- Avoid extreme angles or occlusions
- Close unnecessary applications
- Enable GPU acceleration if available

### Detectron2 Specific
- **Model loading**: First run may take longer to download models
- **Memory usage**: Detectron2 requires more RAM than OpenCV
- **GPU memory**: Ensure sufficient VRAM for GPU acceleration

## 📝 License

This project is part of the Cursey development initiative for advanced multi-person tracking systems with state-of-the-art computer vision.

## 🤝 Contributing

This is a research and development project. For questions or contributions, please refer to the development team.

---

**Cursey Multi-Person Tracking System** - Advanced face and eye tracking with Facebook Detectron2 and modern neumorphism UI
