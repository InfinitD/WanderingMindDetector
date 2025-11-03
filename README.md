# wandering-mind-detector - Lightweight Head Pose & Gaze Demo

Lightweight, camera-based head pose (pitch, yaw, roll) and gaze direction demo with real-time overlay and optional accuracy metrics. Detectron2 and legacy UIs were removed to keep the stack simple and fast.

## Features

### Core Features
- Real-time head pose estimation (pitch, yaw, roll)
- Gaze direction visualization with smoothing
- Accuracy metrics (reprojection error, MAE, angular error)
- Always-on 3D pose mesh preview (matplotlib)

### Visualization
- Inline UI panel with cards for FPS, counts, and pose values
- 3D mesh (axes, head sphere, gaze vector)

## Project Structure

```
wandering-mind-detector/
├── simple_app.py                  # Main application (head pose + gaze + UI)
├── src/
│   └── metrics/
│       ├── __init__.py
│       └── accuracy.py            # Reprojection error, MAE, angular error
├── models/                        # Haar cascade classifiers
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── Dockerfile, docker-compose.yml # Optional containerization
```

## Installation

### Prerequisites
- Python 3.8+
- Camera access permissions
- Adequate lighting for face detection

### Quick Setup
```bash
cd wandering-mind-detector/Cursey
pip install -r requirements.txt
python3 simple_app.py
```

### Options
- Press `q` to quit, `r` to reset smoothing/metrics.

## Controls

### Keyboard Shortcuts
- `q` - Quit application
- `r` - Reset session

## Technical Specifications

### Performance
- Resolution: 640x480 (camera), ~320px side panel
- Target: 30 FPS on typical CPU

### Capabilities
- Head Pose: 3 DOF (pitch, yaw, roll)
- Gaze: 2 DOF projected vector
- Metrics: reprojection error, MAE, angular error

### Visualization
- UI cards for FPS, counts, pose values
- 3D mesh window (axes, head sphere, gaze)

## Architecture

- `simple_app.py`: Captures frames, detects faces (OpenCV), estimates pose/gaze, renders UI and 3D mesh.
- `src/metrics/accuracy.py`: Accuracy metrics computation (reprojection error, MAE, angular error).

## Use Cases

- **Research**: Multi-person gaze tracking studies with SOTA detection
- **Education**: Attention monitoring in classrooms
- **Accessibility**: Eye-controlled interfaces
- **Psychology**: Behavioral analysis and research
- **Human-Computer Interaction**: Gaze-based interaction systems
- **Computer Vision**: Object detection and instance segmentation research

## Getting Started

1. Install deps and run:
   ```bash
   cd Cursey
   pip install -r requirements.txt
   python3 simple_app.py
   ```

2. First run tips:
   - Grant camera permissions
   - Ensure decent lighting
   - Use `r` to reset smoothing/metrics

## Troubleshooting

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

### Notes
- If matplotlib is missing, 3D mesh will be skipped automatically.

## License

This project is part of the wandering-mind-detector development initiative for advanced multi-person tracking systems with state-of-the-art computer vision.

## Contributing

This is a research and development project. For questions or contributions, please refer to the development team.

---

**wandering-mind-detector Multi-Person Tracking System** - Advanced face and eye tracking with Facebook Detectron2 for attention monitoring
