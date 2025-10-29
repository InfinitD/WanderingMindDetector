#!/bin/bash

# Mind Wandering Detector Docker Setup Script
# This script helps you build and run the Mind Wandering Detector in Docker

set -e

echo "Mind Wandering Detector - Docker Setup"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t mind-wandering-detector .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Check camera access
echo "📷 Checking camera access..."
if [ -e /dev/video0 ]; then
    echo "✅ Camera device found at /dev/video0"
else
    echo "⚠️  Camera device not found. Make sure your camera is connected."
fi

# Ask user for run mode
echo ""
echo "Choose run mode:"
echo "1) GUI mode (requires X11 forwarding)"
echo "2) Headless mode (no GUI)"
echo "3) Docker Compose (recommended)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🚀 Starting GUI mode..."
        xhost +local:docker
        docker run -it --rm \
            --device=/dev/video0:/dev/video0 \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            mind-wandering-detector
        ;;
    2)
        echo "🚀 Starting headless mode..."
        docker run -it --rm \
            --device=/dev/video0:/dev/video0 \
            -p 8080:8080 \
            mind-wandering-detector
        ;;
    3)
        echo "🚀 Starting with Docker Compose..."
        docker-compose up mind-wandering-detector
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "✅ Mind Wandering Detector Docker setup complete!"
