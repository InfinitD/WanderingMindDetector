# wandering-mind-detector Docker Documentation

## Overview
This Docker setup allows you to run the wandering-mind-detector application in a containerized environment with proper camera access and GUI support.

## Files Created

### 1. Dockerfile
- **Purpose**: Defines the Docker image for the wandering-mind-detector
- **Base Image**: Ubuntu 20.04
- **Features**:
  - OpenCV and computer vision libraries
  - Python 3 environment
  - Camera access support
  - Non-root user for security
  - Optimized for face and eye detection

### 2. docker-compose.yml
- **Purpose**: Orchestrates the Docker container with proper configuration
- **Services**:
  - `wandering-mind-detector`: GUI mode with X11 forwarding
  - `wandering-mind-detector-headless`: Headless mode for server deployment
- **Features**:
  - Camera device mapping
  - X11 forwarding for GUI
  - Network configuration
  - Volume mounting

### 3. requirements.txt
- **Purpose**: Python dependencies for the container
- **Dependencies**:
  - opencv-python>=4.8.0
  - numpy>=1.24.0
  - opencv-contrib-python>=4.8.0

### 4. docker_setup.sh
- **Purpose**: Automated setup and run script
- **Features**:
  - Docker installation check
  - Image building
  - Camera access verification
  - Multiple run modes
  - User-friendly interface

## Usage

### Quick Start
```bash
# Make script executable and run
chmod +x docker_setup.sh
./docker_setup.sh
```

### Manual Docker Commands

#### Build Image
```bash
docker build -t wandering-mind-detector .
```

#### Run GUI Mode (Linux/macOS with X11)
```bash
# Enable X11 forwarding
xhost +local:docker

# Run container
docker run -it --rm \
    --device=/dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    wandering-mind-detector
```

#### Run Headless Mode
```bash
docker run -it --rm \
    --device=/dev/video0:/dev/video0 \
    -p 8080:8080 \
    wandering-mind-detector
```

#### Using Docker Compose
```bash
# GUI mode
docker-compose up wandering-mind-detector

# Headless mode
docker-compose up wandering-mind-detector-headless
```

## Requirements

### System Requirements
- Docker and Docker Compose installed
- Camera device accessible at `/dev/video0`
- For GUI mode: X11 server running

### Camera Access
The container needs access to your camera device. The setup includes:
- Device mapping: `/dev/video0:/dev/video0`
- Privileged mode for hardware access
- Volume mounting for camera devices

## Troubleshooting

### Camera Not Found
```bash
# Check available video devices
ls /dev/video*

# Update docker-compose.yml with correct device path
# Change /dev/video0 to your actual camera device
```

### GUI Not Working
```bash
# Enable X11 forwarding
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY

# For macOS, you might need XQuartz
# Install: brew install --cask xquartz
```

### Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout and login again
```

## Security Notes
- Container runs as non-root user (`appuser`)
- Camera access is limited to specific devices
- X11 forwarding is restricted to local connections
- No unnecessary privileges granted

## Performance
- Optimized for real-time face and eye detection
- Efficient OpenCV installation
- Minimal base image for faster startup
- Proper resource allocation for computer vision tasks
