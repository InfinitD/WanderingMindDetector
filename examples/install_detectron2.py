#!/usr/bin/env python3
"""
Detectron2 Installation Script for Cursey
Automated setup for Facebook Detectron2 dependencies

Author: Cursey Team
Date: 2025
"""

import subprocess
import sys
import os
import platform


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA is not available. CPU-only mode will be used.")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet. CUDA check will be performed after installation.")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n" + "="*60)
    print("Installing Cursey Dependencies")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install PyTorch (with CUDA support if available)
    system = platform.system().lower()
    if system == "linux" or system == "darwin":  # Linux or macOS
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:  # Windows
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA support"):
        # Fallback to CPU-only
        print("Falling back to CPU-only PyTorch...")
        if not run_command("pip install torch torchvision torchaudio", "Installing PyTorch (CPU-only)"):
            return False
    
    # Install Detectron2
    if not run_command("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html", 
                      "Installing Detectron2"):
        # Fallback to CPU-only Detectron2
        print("Falling back to CPU-only Detectron2...")
        if not run_command("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html", 
                          "Installing Detectron2 (CPU-only)"):
            return False
    
    # Install other dependencies
    dependencies = [
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "numpy>=1.24.0",
        "fvcore>=0.1.5",
        "iopath>=0.1.7",
        "pycocotools>=2.0.2",
        "matplotlib>=3.5.0",
        "pillow>=8.3.0",
        "pytest>=7.0.0",
        "jupyter>=1.0.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    return True


def verify_installation():
    """Verify that all components are properly installed."""
    print("\n" + "="*60)
    print("Verifying Installation")
    print("="*60)
    
    try:
        # Test PyTorch
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        
        # Test CUDA
        check_cuda()
        
        # Test Detectron2
        import detectron2
        print(f"✓ Detectron2 {detectron2.__version__} installed")
        
        # Test OpenCV
        import cv2
        print(f"✓ OpenCV {cv2.__version__} installed")
        
        # Test other dependencies
        import numpy as np
        print(f"✓ NumPy {np.__version__} installed")
        
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} installed")
        
        print("\n✓ All dependencies installed successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def create_test_script():
    """Create a test script to verify Detectron2 functionality."""
    test_script = '''#!/usr/bin/env python3
"""
Test script for Detectron2 installation
"""

import cv2
import numpy as np
import torch

def test_detectron2():
    """Test Detectron2 functionality."""
    try:
        import detectron2
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        
        print("Testing Detectron2...")
        
        # Create config
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        # Set device
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
            print("✓ Using GPU")
        else:
            cfg.MODEL.DEVICE = "cpu"
            print("✓ Using CPU")
        
        # Create predictor
        predictor = DefaultPredictor(cfg)
        print("✓ Detectron2 predictor created successfully")
        
        # Test with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        outputs = predictor(dummy_image)
        print("✓ Detectron2 inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Detectron2 test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_detectron2()
    if success:
        print("\\n✓ Detectron2 installation verified!")
    else:
        print("\\n✗ Detectron2 installation has issues")
'''
    
    with open("test_detectron2.py", "w") as f:
        f.write(test_script)
    
    print("✓ Test script created: test_detectron2.py")


def main():
    """Main installation process."""
    print("Cursey Detectron2 Installation Script")
    print("="*60)
    print("This script will install Facebook Detectron2 and all required dependencies.")
    print("The installation process may take several minutes.")
    
    # Ask for confirmation
    response = input("\nDo you want to continue? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Installation failed. Please check the error messages above.")
        return
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed.")
        return
    
    # Create test script
    create_test_script()
    
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("You can now run the Cursey Detectron2 application:")
    print("python src/detectron_main_app.py")
    print("\nTo test Detectron2 installation:")
    print("python test_detectron2.py")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
