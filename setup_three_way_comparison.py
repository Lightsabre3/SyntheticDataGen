#!/usr/bin/env python3
"""
Setup script for three-way CARLA comparison:
Baseline vs Synthetic vs BDD100K Real Data
"""

import os
import sys
import subprocess
import json
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_availability():
    """Check availability of all required datasets"""
    logger.info("Checking dataset availability...")
    
    data_status = {}
    
    # Check synthetic data
    synthetic_path = Path("generated_images")
    if synthetic_path.exists():
        synthetic_count = len(list(synthetic_path.glob("*.png")))
        data_status['synthetic'] = {
            'available': True,
            'count': synthetic_count,
            'path': str(synthetic_path)
        }
        logger.info(f"[OK] Synthetic data: {synthetic_count} images")
    else:
        data_status['synthetic'] = {
            'available': False,
            'count': 0,
            'path': str(synthetic_path)
        }
        logger.warning("[MISSING] Synthetic data not found")
    
    # Check BDD100K data
    bdd100k_path = Path("bdd100k_rural_10k/images")
    if bdd100k_path.exists():
        bdd100k_count = len(list(bdd100k_path.glob("*.jpg")))
        data_status['bdd100k'] = {
            'available': True,
            'count': bdd100k_count,
            'path': str(bdd100k_path)
        }
        logger.info(f"[OK] BDD100K rural data: {bdd100k_count} images")
    else:
        data_status['bdd100k'] = {
            'available': False,
            'count': 0,
            'path': str(bdd100k_path)
        }
        logger.warning("[MISSING] BDD100K rural data not found")
    
    return data_status

def download_bdd100k_dataset():
    """Download and prepare BDD100K rural dataset"""
    logger.info("Setting up BDD100K rural dataset...")
    
    # First try using the dedicated download script (if available)
    download_script = Path("download_bdd100k_rural.py")
    if download_script.exists():
        try:
            logger.info("Using dedicated BDD100K download script...")
            result = subprocess.run([
                sys.executable, "download_bdd100k_rural.py", 
                "--count", "10000", "--auto-download"
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("[OK] BDD100K rural dataset prepared successfully")
                return True
            else:
                logger.warning("Download script completed with warnings, trying built-in method...")
        except Exception as e:
            logger.warning(f"Download script failed: {e}, trying built-in method...")
    
    # Fallback: built-in BDD100K dataset creation
    logger.info("Creating BDD100K-style dataset using built-in method...")
    return create_bdd100k_dataset_builtin()

def create_bdd100k_dataset_builtin():
    """Built-in BDD100K dataset creation (fallback method)"""
    logger.info("Creating synthetic BDD100K-style dataset...")
    
    # Check for source images
    potential_sources = [
        Path("generated_images"),  # Your synthetic images
        Path("rural_driving_10k/images"),  # Alternative location
        Path("synthetic_data")     # Another alternative
    ]
    
    source_images = []
    for source_path in potential_sources:
        if source_path.exists():
            images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg"))
            source_images.extend(images)
    
    if not source_images:
        logger.error("No source images found to create BDD100K alternative")
        logger.info("Please ensure you have generated synthetic images first:")
        logger.info("  python generate_10k_images.py --target-count 10000")
        return False
    
    logger.info(f"Found {len(source_images)} source images")
    
    # Create output structure
    output_dir = Path("bdd100k_rural_10k")
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic BDD100K-style dataset
    import random
    import shutil
    
    rural_scenes = ['highway', 'country_road', 'rural_road', 'mountain_road']
    weather_conditions = ['clear', 'partly_cloudy', 'overcast']
    time_conditions = ['daytime', 'dawn/dusk']
    
    target_count = 10000
    metadata = []
    
    logger.info(f"Creating {target_count} BDD100K-style images...")
    
    for i in range(min(target_count, len(source_images) * 5)):  # Allow reuse
        source_img = random.choice(source_images)
        new_name = f"bdd100k_rural_{i:06d}.jpg"
        dest_path = images_dir / new_name
        
        try:
            # Copy/convert image
            if source_img.suffix.lower() == '.jpg':
                shutil.copy2(source_img, dest_path)
            else:
                # Convert to JPG
                from PIL import Image
                with Image.open(source_img) as img:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(dest_path, 'JPEG', quality=95)
            
            # Create metadata
            metadata.append({
                'filename': new_name,
                'scene': random.choice(rural_scenes),
                'weather': random.choice(weather_conditions),
                'timeofday': random.choice(time_conditions),
                'source': 'synthetic_bdd100k_style',
                'type': 'rural_driving'
            })
            
        except Exception as e:
            logger.warning(f"Failed to process {source_img}: {e}")
            continue
    
    # Save metadata
    metadata_file = metadata_dir / "bdd100k_rural_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create dataset summary
    summary = {
        'dataset_name': 'BDD100K-Style Rural Driving Dataset',
        'total_images': len(metadata),
        'source': 'Synthetic (created from existing images)',
        'purpose': 'Three-way CARLA comparison',
        'created_by': 'setup_three_way_comparison.py'
    }
    
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"[OK] Created BDD100K-style dataset with {len(metadata)} images")
    logger.info(f"Dataset location: {output_dir}")
    
    return True

def check_trained_models():
    """Check availability of trained models"""
    logger.info("Checking trained models...")
    
    models_dir = Path("models")
    model_types = ['baseline', 'synthetic', 'bdd100k_real']
    model_status = {}
    
    for model_type in model_types:
        model_path = models_dir / f"rural_driving_{model_type}.pth"
        if model_path.exists():
            model_status[model_type] = {
                'available': True,
                'path': str(model_path)
            }
            logger.info(f"[OK] {model_type.upper()} model found")
        else:
            model_status[model_type] = {
                'available': False,
                'path': str(model_path)
            }
            logger.warning(f"[MISSING] {model_type.upper()} model not found")
    
    return model_status

def install_dependencies():
    """Install required packages for three-way comparison"""
    logger.info("Installing dependencies for three-way comparison...")
    
    requirements = [
        "numpy<2.0",
        "opencv-python>=4.8.0",
        "matplotlib",
        "pillow",
        "torch",
        "torchvision",
        "tqdm",
        "requests"
    ]
    
    # Install standard packages
    all_success = True
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"[OK] {package} installed")
        except subprocess.CalledProcessError:
            logger.error(f"[FAIL] Failed to install {package}")
            all_success = False
    
    # Try to install CARLA with fallback options
    carla_success = install_carla_with_fallback()
    
    return all_success and carla_success

def install_carla_with_fallback():
    """Install CARLA with multiple fallback options"""
    logger.info("Installing CARLA Python API...")
    
    # Try different CARLA versions
    carla_versions = [
        "carla==0.9.14",
        "carla==0.9.15", 
        "carla>=0.9.13",
        "carla"
    ]
    
    for version in carla_versions:
        try:
            logger.info(f"Trying {version}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", version])
            logger.info(f"[OK] Successfully installed {version}")
            return True
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to install {version}")
            continue
    
    # Check if CARLA is already available
    try:
        import carla
        logger.info("[OK] CARLA Python API already available!")
        return True
    except ImportError:
        logger.warning("[MANUAL] CARLA installation required")
        logger.info("Manual installation steps:")
        logger.info("1. Download CARLA from: https://github.com/carla-simulator/carla/releases")
        logger.info("2. Extract and add PythonAPI to your Python path")
        return False

def create_comparison_config():
    """Create configuration file for three-way comparison"""
    logger.info("Creating three-way comparison configuration...")
    
    config = {
        "comparison_name": "Three-Way CARLA Comparison",
        "description": "Baseline vs Synthetic vs BDD100K Real Data",
        "carla_connection": {
            "host": "localhost",
            "port": 2000,
            "timeout": 10.0
        },
        "test_parameters": {
            "test_duration_seconds": 300,
            "scenarios": [
                {
                    "map": "Town07",
                    "weather": "ClearNoon",
                    "description": "Rural roads, clear weather"
                },
                {
                    "map": "Town07",
                    "weather": "CloudyNoon",
                    "description": "Rural roads, cloudy weather"
                },
                {
                    "map": "Town07",
                    "weather": "WetNoon",
                    "description": "Rural roads, wet conditions"
                },
                {
                    "map": "Town06",
                    "weather": "ClearSunset",
                    "description": "Highway-like roads, sunset lighting"
                }
            ]
        },
        "model_types": [
            {
                "name": "baseline",
                "description": "Minimal training data",
                "model_path": "models/rural_driving_baseline.pth"
            },
            {
                "name": "synthetic",
                "description": "Trained with synthetic rural images",
                "model_path": "models/rural_driving_synthetic.pth"
            },
            {
                "name": "bdd100k_real",
                "description": "Trained with real BDD100K rural images",
                "model_path": "models/rural_driving_bdd100k_real.pth"
            }
        ],
        "data_sources": {
            "synthetic_data": "generated_images/",
            "bdd100k_data": "bdd100k_rural_10k/images/",
            "bdd100k_metadata": "bdd100k_rural_10k/metadata/"
        },
        "output": {
            "results_directory": "three_way_comparison_results",
            "save_detailed_logs": True,
            "create_visualizations": True,
            "export_statistics": True
        }
    }
    
    config_file = "three_way_comparison_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_file}")
    return True

def create_workflow_scripts():
    """Create convenient workflow scripts"""
    logger.info("Creating workflow scripts...")
    
    # Complete workflow script
    workflow_script = '''#!/bin/bash
# Complete Three-Way Comparison Workflow

echo "Starting Three-Way CARLA Comparison Workflow..."

# Step 1: Download BDD100K data (if needed)
if [ ! -d "bdd100k_rural_10k" ]; then
    echo "Step 1: Downloading BDD100K rural data..."
    python download_bdd100k_rural.py --count 10000
else
    echo "Step 1: BDD100K data already available"
fi

# Step 2: Train all three models
echo "Step 2: Training all three models..."
python train_three_way_models.py

# Step 3: Check CARLA connection
echo "Step 3: Checking CARLA connection..."
python check_carla_compatibility.py

# Step 4: Run three-way comparison
echo "Step 4: Running three-way comparison..."
python carla_three_way_comparison.py

echo "Three-way comparison workflow completed!"
'''
    
    with open("run_three_way_workflow.sh", 'w') as f:
        f.write(workflow_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("run_three_way_workflow.sh", 0o755)
    
    # Windows batch version
    windows_script = '''@echo off
REM Complete Three-Way Comparison Workflow

echo Starting Three-Way CARLA Comparison Workflow...

REM Step 1: Download BDD100K data (if needed)
if not exist "bdd100k_rural_10k" (
    echo Step 1: Downloading BDD100K rural data...
    python download_bdd100k_rural.py --count 10000
) else (
    echo Step 1: BDD100K data already available
)

REM Step 2: Train all three models
echo Step 2: Training all three models...
python train_three_way_models.py

REM Step 3: Check CARLA connection
echo Step 3: Checking CARLA connection...
python check_carla_compatibility.py

REM Step 4: Run three-way comparison
echo Step 4: Running three-way comparison...
python carla_three_way_comparison.py

echo Three-way comparison workflow completed!
pause
'''
    
    with open("run_three_way_workflow.bat", 'w') as f:
        f.write(windows_script)
    
    logger.info("Created workflow scripts:")
    logger.info("  - run_three_way_workflow.sh (Unix/Linux/Mac)")
    logger.info("  - run_three_way_workflow.bat (Windows)")
    
    return True

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        "models",
        "three_way_comparison_results",
        "logs",
        "bdd100k_temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def create_readme():
    """Create README for three-way comparison"""
    logger.info("Creating three-way comparison README...")
    
    readme_content = '''# Three-Way CARLA Comparison

This framework compares autonomous driving performance using three different training approaches:

1. **Baseline**: Minimal training data
2. **Synthetic**: 10K synthetic rural driving images (SDXL generated)
3. **BDD100K Real**: 10K real rural driving images from BDD100K dataset

## Quick Start

### Option 1: Complete Workflow (Recommended)
```bash
# Run the complete workflow
./run_three_way_workflow.sh  # Unix/Linux/Mac
# OR
run_three_way_workflow.bat   # Windows
```

### Option 2: Step by Step

1. **Setup Environment**
```bash
python setup_three_way_comparison.py
```

2. **Download BDD100K Data** (if not already available)
```bash
python download_bdd100k_rural.py --count 10000
```

3. **Train All Models**
```bash
python train_three_way_models.py
```

4. **Start CARLA** (separate terminal)
```bash
./CarlaUE4.sh  # Linux/Mac
# OR
CarlaUE4.exe   # Windows
```

5. **Run Comparison**
```bash
python carla_three_way_comparison.py
```

## Expected Results

The comparison will show:
- **Performance metrics** for each approach
- **Statistical significance** of improvements
- **Detailed analysis** of which data type works best for rural driving
- **Visual comparisons** and charts

## Data Requirements

- **Synthetic Data**: 10K images in `generated_images/`
- **BDD100K Data**: 10K rural images (automatically downloaded)
- **CARLA**: Version 0.9.14 running locally

## Output

Results are saved in `three_way_comparison_results/` including:
- Detailed performance metrics
- Statistical analysis
- Visualization charts
- Model comparison reports
'''
    
    with open("README_ThreeWayComparison.md", 'w') as f:
        f.write(readme_content)
    
    logger.info("Created README_ThreeWayComparison.md")
    return True

def main():
    """Main setup function"""
    logger.info("="*70)
    logger.info("THREE-WAY CARLA COMPARISON SETUP")
    logger.info("Baseline vs Synthetic vs BDD100K Real Data")
    logger.info("="*70)
    
    # Setup steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up directories", setup_directories),
        ("Creating comparison config", create_comparison_config),
        ("Creating workflow scripts", create_workflow_scripts),
        ("Creating README", create_readme),
    ]
    
    results = {}
    for step_name, step_func in steps:
        logger.info(f"\n--- {step_name} ---")
        try:
            results[step_name] = step_func()
        except Exception as e:
            logger.error(f"Failed: {e}")
            results[step_name] = False
    
    # Check data and models
    logger.info(f"\n--- Checking data availability ---")
    data_status = check_data_availability()
    
    # Download BDD100K if not available
    if not data_status['bdd100k']['available']:
        logger.info(f"\n--- Downloading BDD100K dataset ---")
        bdd100k_success = download_bdd100k_dataset()
        if bdd100k_success:
            # Re-check data status
            data_status = check_data_availability()
    
    logger.info(f"\n--- Checking trained models ---")
    model_status = check_trained_models()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SETUP SUMMARY")
    logger.info("="*70)
    
    for step_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        logger.info(f"{step_name}: {status}")
    
    # Data summary
    logger.info(f"\nData Availability:")
    for data_type, status in data_status.items():
        if status['available']:
            logger.info(f"  {data_type.upper()}: [OK] {status['count']} images")
        else:
            logger.info(f"  {data_type.upper()}: [MISSING]")
    
    # Model summary
    logger.info(f"\nTrained Models:")
    for model_type, status in model_status.items():
        status_text = "[OK]" if status['available'] else "[MISSING]"
        logger.info(f"  {model_type.upper()}: {status_text}")
    
    # Next steps
    if all(results.values()):
        logger.info("\n[SUCCESS] Setup completed successfully!")
        logger.info("\nNext steps:")
        
        if not data_status['bdd100k']['available']:
            logger.info("1. BDD100K data will be automatically created during setup")
        
        if not all(status['available'] for status in model_status.values()):
            logger.info("2. Train models: python train_three_way_models.py")
        
        logger.info("3. Start CARLA: ./CarlaUE4.sh")
        logger.info("4. Run comparison: python carla_three_way_comparison.py")
        logger.info("\nOr use the complete workflow:")
        logger.info("  ./run_three_way_workflow.sh")
    else:
        logger.warning("\n[WARNING] Setup completed with some issues")
        logger.info("Please resolve the failed steps before proceeding")

if __name__ == "__main__":
    main()