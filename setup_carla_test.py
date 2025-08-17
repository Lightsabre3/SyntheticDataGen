#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for CARLA synthetic data testing environment
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_carla_installation():
    """Check if CARLA is installed and accessible"""
    logger.info("Checking CARLA installation...")
    
    # Common CARLA installation paths
    carla_paths = [
        "/opt/carla-simulator",
        "/usr/local/carla",
        "C:/CARLA_0.9.15",  # Windows
        os.path.expanduser("~/CARLA_0.9.15")
    ]
    
    carla_found = False
    for path in carla_paths:
        if os.path.exists(path):
            logger.info(f"CARLA found at: {path}")
            carla_found = True
            break
    
    if not carla_found:
        logger.warning("CARLA installation not found in common paths")
        logger.info("Please ensure CARLA is installed and accessible")
        logger.info("Download from: https://github.com/carla-simulator/carla/releases")
    
    return carla_found

def install_carla_package():
    """Install CARLA package using the dedicated installer"""
    logger.info("Installing CARLA Python API...")
    
    try:
        # Run the dedicated CARLA installer
        result = subprocess.run([sys.executable, "install_carla_api.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("[OK] CARLA Python API installed successfully")
            return True
        else:
            logger.warning("CARLA installer completed with warnings")
            logger.info("Output: " + result.stdout)
            
            # Check if CARLA is actually available despite warnings
            try:
                import carla
                logger.info("[OK] CARLA Python API is available")
                return True
            except ImportError:
                logger.error("[FAIL] CARLA Python API not available")
                return False
                
    except Exception as e:
        logger.error(f"Failed to run CARLA installer: {e}")
        
        # Fallback: check if CARLA is already available
        try:
            import carla
            logger.info("[OK] CARLA Python API already available")
            return True
        except ImportError:
            logger.error("[FAIL] CARLA Python API not found")
            logger.info("Please run manually: python install_carla_api.py")
            return False

def install_requirements():
    """Install required Python packages"""
    logger.info("Installing required packages...")
    
    requirements = [
        "numpy<2.0",  # Pin to NumPy 1.x for compatibility
        "opencv-python>=4.8.0", 
        "matplotlib",
        "pillow",
        "torch",
        "torchvision"
    ]
    
    # Install standard packages first
    all_success = True
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"[OK] {package} installed successfully")
        except subprocess.CalledProcessError:
            logger.error(f"[FAIL] Failed to install {package}")
            all_success = False
    
    # Try to install CARLA
    carla_success = install_carla_package()
    
    return all_success and carla_success

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        "carla_test_results",
        "models",
        "logs",
        "camera_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def check_synthetic_data():
    """Check if synthetic data is available"""
    logger.info("Checking synthetic data...")
    
    synthetic_data_path = "rural_driving_10k\images"
    if os.path.exists(synthetic_data_path):
        image_count = len([f for f in os.listdir(synthetic_data_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        logger.info(f"Found {image_count} synthetic images in {synthetic_data_path}")
        return True
    else:
        logger.warning(f"Synthetic data directory not found: {synthetic_data_path}")
        logger.info("Please ensure your generated 10k images are in the 'generated_images' folder")
        return False

def create_model_placeholders():
    """Create placeholder model files for testing"""
    logger.info("Creating model placeholders...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create placeholder model files
    placeholder_models = [
        "rural_driving_with_synthetic.pth",
        "rural_driving_baseline.pth"
    ]
    
    for model_name in placeholder_models:
        model_path = models_dir / model_name
        if not model_path.exists():
            # Create a simple placeholder file
            with open(model_path, 'w') as f:
                f.write(f"# Placeholder for {model_name}\n")
                f.write("# Replace this with your actual trained model\n")
            logger.info(f"Created placeholder: {model_path}")
    
    return True

def test_carla_connection():
    """Test connection to CARLA server"""
    logger.info("Testing CARLA connection...")
    
    try:
        import carla
        
        # Try to connect
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        
        # Test connection
        world = client.get_world()
        logger.info(f"[OK] Successfully connected to CARLA world: {world.get_map().name}")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Failed to connect to CARLA: {e}")
        logger.info("Make sure CARLA server is running:")
        logger.info("  Linux/Mac: ./CarlaUE4.sh")
        logger.info("  Windows: CarlaUE4.exe")
        return False

def create_run_script():
    """Create convenient run scripts for different platforms"""
    logger.info("Creating run scripts...")
    
    # Unix/Linux/Mac script
    unix_script_content = '''#!/bin/bash
# CARLA Synthetic Data Test Runner

echo "Starting CARLA Synthetic Data Test..."

# Check if CARLA is running
if ! pgrep -f "CarlaUE4" > /dev/null; then
    echo "[FAIL] CARLA server is not running!"
    echo "Please start CARLA server first:"
    echo "  Linux/Mac: ./CarlaUE4.sh"
    echo "  Windows: CarlaUE4.exe"
    exit 1
fi

echo "[OK] CARLA server detected"

# Run the test
python carla_synthetic_data_test.py

echo "Test completed!"
'''
    
    # Windows batch script
    windows_script_content = '''@echo off
REM CARLA Synthetic Data Test Runner

echo Starting CARLA Synthetic Data Test...

REM Check if CARLA is running
tasklist /FI "IMAGENAME eq CarlaUE4.exe" 2>NUL | find /I /N "CarlaUE4.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo [FAIL] CARLA server is not running!
    echo Please start CARLA server first:
    echo   Windows: CarlaUE4.exe
    echo   Linux/Mac: ./CarlaUE4.sh
    pause
    exit /b 1
)

echo [OK] CARLA server detected

REM Run the test
python carla_synthetic_data_test.py

echo Test completed!
pause
'''
    
    # Create Unix script
    with open("run_carla_test.sh", 'w', encoding='utf-8') as f:
        f.write(unix_script_content)
    
    # Create Windows script
    with open("run_carla_test.bat", 'w', encoding='utf-8') as f:
        f.write(windows_script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("run_carla_test.sh", 0o755)
    
    logger.info("Created run_carla_test.sh (Unix/Linux/Mac)")
    logger.info("Created run_carla_test.bat (Windows)")
    
    return True

def main():
    """Main setup function"""
    logger.info("="*50)
    logger.info("CARLA SYNTHETIC DATA TEST SETUP")
    logger.info("="*50)
    
    # Setup steps
    steps = [
        ("Checking CARLA installation", check_carla_installation),
        ("Installing requirements", install_requirements),
        ("Setting up directories", setup_directories),
        ("Checking synthetic data", check_synthetic_data),
        ("Creating model placeholders", create_model_placeholders),
        ("Creating run script", create_run_script),
        ("Testing CARLA connection", test_carla_connection)
    ]
    
    results = {}
    for step_name, step_func in steps:
        logger.info(f"\n--- {step_name} ---")
        try:
            results[step_name] = step_func()
        except Exception as e:
            logger.error(f"Failed: {e}")
            results[step_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SETUP SUMMARY")
    logger.info("="*50)
    
    for step_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        logger.info(f"{step_name}: {status}")
    
    if all(results.values()):
        logger.info("\n[SUCCESS] Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start CARLA server: ./CarlaUE4.sh (or CarlaUE4.exe on Windows)")
        logger.info("2. Run test: python carla_synthetic_data_test.py")
        logger.info("   Or use: ./run_carla_test.sh")
    else:
        logger.warning("\n[WARNING] Setup completed with some issues")
        logger.info("Please resolve the failed steps before running tests")

if __name__ == "__main__":
    main()