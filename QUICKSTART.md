# Cursey Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
cd src
python3 main_app.py
```

### 3. Start Tracking
- Click **"Start"** button in the side panel
- Position yourself in front of the camera
- Watch your geometric mannequin respond to your movements!

## 🎮 Quick Controls

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Start/Stop | Click "Start" button | `s` |
| Toggle Mannequins | Click "Mannequins" button | `m` |
| Show/Hide Values | Click "Show Values" button | `v` |
| Reset Session | Click "Reset" button | `r` |
| Quit | - | `q` |

## 📊 What You'll See

### Camera Feed (Left Side)
- Live video from your camera
- No visual overlays (clean interface)

### Side Panel (Right Side)
- **Control Buttons**: Start, Reset, Mannequins, Values, Record
- **Session Info**: Duration, frames, detections, FPS
- **Real-time Values**: Head pose and gaze angles
- **Geometric Mannequins**: Abstract 3D pose visualization
- **Status Indicators**: Tracking quality and system health

## �� Multi-Person Mode

1. Have 2-3 people in front of the camera
2. Each person gets their own mannequin widget
3. Real-time values display for each person
4. Session statistics track all detections

## 🔧 Troubleshooting

### Camera Issues
- **"Could not open camera"**: Check camera permissions in System Settings
- **No video feed**: Ensure camera is not being used by another application

### Tracking Issues
- **Poor detection**: Improve lighting conditions
- **Low quality**: Position camera at eye level
- **No mannequins**: Click "Mannequins" button to enable

### Performance Issues
- **Low FPS**: Close other applications
- **Laggy response**: Check system resources

## 📈 Understanding the Display

### Geometric Mannequin
- **Head**: Light blue circle with pose axes
- **Body**: Geometric rectangles and lines
- **Gaze**: Cyan arrows showing direction
- **Pose Info**: Real-time angle displays

### Real-time Values
- **Pitch**: Head up/down angle (-180° to 180°)
- **Yaw**: Head left/right angle (-180° to 180°)
- **Roll**: Head tilt angle (-180° to 180°)
- **Gaze H**: Horizontal gaze direction (-180° to 180°)
- **Gaze V**: Vertical gaze direction (-180° to 180°)

## 🎨 Customization

### Button States
- **Active buttons**: Highlighted when enabled
- **Inactive buttons**: Grayed out when disabled
- **Status colors**: Green (good), Yellow (warning), Red (error)

### Display Options
- Toggle mannequins on/off
- Show/hide real-time values
- Start/stop recording mode
- Reset session data

## 🚀 Advanced Usage

### Command Line Options
```bash
# Specify camera and max persons
python3 main_app.py 0 3

# Camera 0, max 3 people
```

### Performance Monitoring
- Watch FPS counter in session info
- Monitor tracking quality indicator
- Check detection count statistics

---

**Ready to track!** 🎯
