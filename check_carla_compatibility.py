#!/usr/bin/env python3
"""
CARLA 0.9.14 Compatibility Checker
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_carla_compatibility():
    """Check CARLA 0.9.14 compatibility"""
    logger.info("Checking CARLA 0.9.14 compatibility...")
    
    try:
        import carla
        logger.info(f"[OK] CARLA Python API imported successfully")
        
        # Try to connect
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        
        # Get version info
        try:
            client_version = client.get_client_version()
            server_version = client.get_server_version()
            
            logger.info(f"Client API version: {client_version}")
            logger.info(f"Server version: {server_version}")
            
            if "0.9.14" in server_version:
                logger.info("[OK] CARLA 0.9.14 server detected")
            else:
                logger.warning(f"[WARNING] Expected CARLA 0.9.14, found: {server_version}")
                
        except Exception as version_error:
            logger.warning(f"Could not get version info: {version_error}")
        
        # Test basic functionality
        world = client.get_world()
        current_map = world.get_map().name
        logger.info(f"[OK] Current map: {current_map}")
        
        # Check available maps
        try:
            available_maps = client.get_available_maps()
            rural_maps = [m for m in available_maps if 'Town07' in m or 'Town06' in m]
            logger.info(f"[OK] Rural maps available: {len(rural_maps)}")
            for map_name in rural_maps[:3]:  # Show first 3
                logger.info(f"  - {map_name.split('/')[-1]}")
        except Exception as maps_error:
            logger.warning(f"Could not list maps: {maps_error}")
        
        # Test blueprint library
        blueprint_library = world.get_blueprint_library()
        vehicles = blueprint_library.filter('vehicle.*')
        logger.info(f"[OK] Available vehicles: {len(vehicles)}")
        
        # Test Tesla Model 3 (commonly used)
        tesla_bp = blueprint_library.filter('vehicle.tesla.model3')
        if tesla_bp:
            logger.info("[OK] Tesla Model 3 blueprint available")
        else:
            logger.warning("[WARNING] Tesla Model 3 not found, will use alternative vehicle")
        
        logger.info("\n[SUCCESS] CARLA 0.9.14 compatibility check passed!")
        return True
        
    except ImportError:
        logger.error("[FAIL] CARLA Python API not installed")
        logger.info("Install with: pip install carla==0.9.14")
        return False
        
    except Exception as e:
        logger.error(f"[FAIL] CARLA connection failed: {e}")
        logger.info("Make sure CARLA server is running:")
        logger.info("  ./CarlaUE4.sh (Linux/Mac)")
        logger.info("  CarlaUE4.exe (Windows)")
        return False

def check_dependencies():
    """Check other dependencies"""
    logger.info("\nChecking dependencies...")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('matplotlib.pyplot', 'matplotlib'),
        ('torch', 'torch'),
    ]
    
    all_good = True
    for module, package in dependencies:
        try:
            __import__(module)
            logger.info(f"[OK] {package}")
        except ImportError:
            logger.error(f"[FAIL] {package} not installed")
            all_good = False
    
    return all_good

def main():
    """Main compatibility check"""
    logger.info("="*50)
    logger.info("CARLA 0.9.14 COMPATIBILITY CHECK")
    logger.info("="*50)
    
    deps_ok = check_dependencies()
    carla_ok = check_carla_compatibility()
    
    if deps_ok and carla_ok:
        logger.info("\n[SUCCESS] All compatibility checks passed!")
        logger.info("You can now run: python carla_synthetic_data_test.py")
    else:
        logger.error("\n[FAIL] Some compatibility issues found")
        if not deps_ok:
            logger.info("Run: python setup_carla_test.py")
        if not carla_ok:
            logger.info("Make sure CARLA 0.9.14 server is running")

if __name__ == "__main__":
    main()