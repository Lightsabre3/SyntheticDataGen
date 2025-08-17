#!/usr/bin/env python3
"""
CARLA Python API Installation Script
Handles multiple installation methods and versions
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import zipfile
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CARLAInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.carla_versions = ["0.9.14", "0.9.15", "0.9.13"]
        
    def try_pip_installation(self):
        """Try installing CARLA via pip with multiple versions"""
        logger.info("Attempting pip installation...")
        
        pip_packages = [
            "carla==0.9.14",
            "carla==0.9.15",
            "carla>=0.9.13",
            "carla"
        ]
        
        for package in pip_packages:
            try:
                logger.info(f"Trying: pip install {package}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"[SUCCESS] Installed {package}")
                return self.verify_installation()
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install {package}")
                continue
        
        return False
    
    def download_carla_egg(self, version="0.9.14"):
        """Download CARLA .egg file directly"""
        logger.info(f"Downloading CARLA {version} .egg file...")
        
        # Determine the correct .egg file based on system and Python version
        if self.system == "windows":
            egg_name = f"carla-{version}-py{self.python_version}-win-amd64.egg"
        elif self.system == "linux":
            egg_name = f"carla-{version}-py{self.python_version}-linux-x86_64.egg"
        else:
            logger.error(f"Unsupported system: {self.system}")
            return False
        
        # GitHub releases URL
        base_url = f"https://github.com/carla-simulator/carla/releases/download/{version}"
        egg_url = f"{base_url}/{egg_name}"
        
        try:
            logger.info(f"Downloading from: {egg_url}")
            response = requests.get(egg_url, stream=True)
            response.raise_for_status()
            
            # Save to site-packages
            import site
            site_packages = site.getsitepackages()[0]
            egg_path = Path(site_packages) / egg_name
            
            with open(egg_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded to: {egg_path}")
            
            # Add to Python path
            if str(egg_path) not in sys.path:
                sys.path.insert(0, str(egg_path))
            
            return self.verify_installation()
            
        except Exception as e:
            logger.error(f"Failed to download .egg file: {e}")
            return False
    
    def install_from_local_carla(self):
        """Try to find and install from local CARLA installation"""
        logger.info("Looking for local CARLA installation...")
        
        # Common CARLA installation paths
        carla_paths = [
            "C:/CARLA_0.9.14",
            "C:/CARLA_0.9.15", 
            "/opt/carla-simulator",
            "/usr/local/carla",
            os.path.expanduser("~/CARLA_0.9.14"),
            os.path.expanduser("~/CARLA_0.9.15"),
            "./CARLA_0.9.14",
            "./CARLA_0.9.15"
        ]
        
        for carla_path in carla_paths:
            carla_dir = Path(carla_path)
            if carla_dir.exists():
                logger.info(f"Found CARLA installation at: {carla_dir}")
                
                # Look for Python API
                python_api_dir = carla_dir / "PythonAPI" / "carla" / "dist"
                if python_api_dir.exists():
                    # Find .egg files
                    egg_files = list(python_api_dir.glob("*.egg"))
                    if egg_files:
                        egg_file = egg_files[0]  # Use first .egg file found
                        logger.info(f"Found CARLA .egg file: {egg_file}")
                        
                        # Add to Python path
                        if str(egg_file) not in sys.path:
                            sys.path.insert(0, str(egg_file))
                        
                        # Also try to copy to site-packages
                        try:
                            import site
                            import shutil
                            site_packages = site.getsitepackages()[0]
                            dest_path = Path(site_packages) / egg_file.name
                            shutil.copy2(egg_file, dest_path)
                            logger.info(f"Copied .egg to site-packages: {dest_path}")
                        except Exception as e:
                            logger.warning(f"Could not copy to site-packages: {e}")
                        
                        return self.verify_installation()
        
        logger.warning("No local CARLA installation found")
        return False
    
    def verify_installation(self):
        """Verify that CARLA can be imported"""
        try:
            import carla
            version = getattr(carla, '__version__', 'unknown')
            logger.info(f"[SUCCESS] CARLA Python API imported successfully (version: {version})")
            return True
        except ImportError as e:
            logger.error(f"[FAIL] Cannot import CARLA: {e}")
            return False
    
    def install(self):
        """Try all installation methods"""
        logger.info("="*50)
        logger.info("CARLA PYTHON API INSTALLER")
        logger.info("="*50)
        logger.info(f"System: {self.system}")
        logger.info(f"Python: {self.python_version}")
        
        # Check if already installed
        if self.verify_installation():
            logger.info("CARLA is already installed and working!")
            return True
        
        # Method 1: Try pip installation
        logger.info("\n--- Method 1: Pip Installation ---")
        if self.try_pip_installation():
            return True
        
        # Method 2: Try local CARLA installation
        logger.info("\n--- Method 2: Local CARLA Installation ---")
        if self.install_from_local_carla():
            return True
        
        # Method 3: Download .egg file
        logger.info("\n--- Method 3: Download .egg File ---")
        for version in self.carla_versions:
            if self.download_carla_egg(version):
                return True
        
        # All methods failed
        logger.error("\n[FAIL] All installation methods failed")
        self.print_manual_instructions()
        return False
    
    def print_manual_instructions(self):
        """Print manual installation instructions"""
        logger.info("\n" + "="*50)
        logger.info("MANUAL INSTALLATION REQUIRED")
        logger.info("="*50)
        logger.info("1. Download CARLA from:")
        logger.info("   https://github.com/carla-simulator/carla/releases")
        logger.info("2. Extract the CARLA package")
        logger.info("3. Find the Python API .egg file:")
        if self.system == "windows":
            logger.info(f"   CARLA_ROOT/PythonAPI/carla/dist/carla-*-py{self.python_version}-win-amd64.egg")
        else:
            logger.info(f"   CARLA_ROOT/PythonAPI/carla/dist/carla-*-py{self.python_version}-linux-x86_64.egg")
        logger.info("4. Copy the .egg file to your Python site-packages directory")
        logger.info("5. Or add the .egg file to your PYTHONPATH environment variable")

def main():
    """Main installation function"""
    installer = CARLAInstaller()
    success = installer.install()
    
    if success:
        logger.info("\n[SUCCESS] CARLA Python API installation completed!")
        logger.info("You can now run CARLA-based scripts.")
    else:
        logger.error("\n[FAIL] CARLA Python API installation failed.")
        logger.info("Please follow the manual installation instructions above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)