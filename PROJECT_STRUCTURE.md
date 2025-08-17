# 🗂️ Project Structure

## 📁 Current Clean Structure

```
sdxl-rural-driving-generator/
├── 📊 Data Generation (Core)
│   ├── sdxl_rural_driving_gen.ipynb      # Interactive generation notebook
│   ├── generate_10k_images.py            # Production-scale generation (10K+ images)
│   ├── monitor_generation.py             # Real-time generation monitoring
│   ├── test_first_batch.py               # Quick testing and diagnostics
│   ├── config_10k.json                   # Production generation configuration
│   └── requirements_10k.txt              # Production dependencies
│
├── 🚗 CARLA Testing Framework
│   ├── carla_synthetic_data_test.py      # Two-way comparison (baseline vs synthetic)
│   ├── carla_three_way_comparison.py     # Three-way comparison (+ BDD100K)
│   ├── carla_test_config.json            # CARLA testing configuration
│   ├── setup_carla_test.py               # CARLA environment setup
│   ├── check_carla_compatibility.py      # CARLA 0.9.14 compatibility checker
│   └── README_CARLA_Testing.md           # CARLA testing documentation
│
├── 🤖 Model Training
│   ├── train_with_synthetic_data.py      # Basic model training example
│   ├── train_three_way_models.py         # Three-way model training
│   └── download_bdd100k_rural.py         # BDD100K dataset extraction
│
├── 🔧 Setup & Utilities
│   ├── setup_three_way_comparison.py     # Complete three-way setup
│   ├── install_carla_api.py              # Robust CARLA installation
│   ├── fix_diffusers_dependencies.py     # Dependency conflict resolver
│   ├── fix_numpy_opencv.py               # NumPy 2.0 compatibility fix
│   └── carla_integration.ipynb           # Interactive CARLA integration
│
├── 📖 Documentation
│   ├── README.md                         # Main project documentation
│   ├── README_CARLA_Testing.md          # CARLA testing guide
│   └── PROJECT_STRUCTURE.md             # This file
│
└── 📁 Generated Data & Results
    ├── generated_images/                 # Your synthetic images
    ├── bdd100k_rural_10k/               # BDD100K rural subset
    ├── models/                          # Trained model files
    ├── carla_test_results/              # CARLA testing results
    └── three_way_comparison_results/    # Three-way comparison results
```

## 🎯 Main Workflows

### 1. **Data Generation Workflow**
```bash
# Interactive development
jupyter notebook sdxl_rural_driving_gen.ipynb

# Production generation
python generate_10k_images.py --target-count 10000

# Monitor progress
python monitor_generation.py

# Quick testing
python test_first_batch.py
```

### 2. **CARLA Two-Way Testing**
```bash
# Setup
python setup_carla_test.py

# Start CARLA
./CarlaUE4.sh

# Run comparison
python carla_synthetic_data_test.py
```

### 3. **CARLA Three-Way Testing**
```bash
# Complete setup
python setup_three_way_comparison.py

# Download BDD100K data
python download_bdd100k_rural.py

# Train models
python train_three_way_models.py

# Run three-way comparison
python carla_three_way_comparison.py
```

## 🧹 Removed Obsolete Files

The following files were removed during cleanup:
- `BLACK_IMAGE_FIX.md` → Information integrated into main README
- `check_dependencies.py` → Functionality in setup scripts
- `debug_generation.py` → Replaced by better diagnostic tools
- `diagnose_black_images.py` → Functionality in test_first_batch.py
- `launch_10k_generation.py` → Functionality in main generation script
- `quick_test_generation.py` → Replaced by test_first_batch.py
- `run_diagnostic_tests.sh` → Replaced by Python scripts
- `setup_environment.py` → Functionality in main setup scripts
- `test_config.json` → Replaced by config_10k.json
- `test_first_batch_README.md` → Information in main README
- `test_fix.py` → Issues resolved in main codebase
- `test_quality_assessment.py` → Functionality in main scripts

## 🎨 File Categories

### **Core Production Files** (Keep)
- Generation: `generate_10k_images.py`, `monitor_generation.py`
- Testing: `carla_synthetic_data_test.py`, `carla_three_way_comparison.py`
- Training: `train_three_way_models.py`
- Setup: `setup_carla_test.py`, `setup_three_way_comparison.py`

### **Development/Interactive Files** (Keep)
- Notebooks: `sdxl_rural_driving_gen.ipynb`, `carla_integration.ipynb`
- Quick testing: `test_first_batch.py`
- Compatibility: `check_carla_compatibility.py`

### **Utility/Fix Files** (Keep)
- Installation: `install_carla_api.py`
- Fixes: `fix_diffusers_dependencies.py`, `fix_numpy_opencv.py`
- Data: `download_bdd100k_rural.py`

### **Configuration Files** (Keep)
- `config_10k.json`, `carla_test_config.json`
- `requirements_10k.txt`

### **Documentation** (Keep)
- `README.md`, `README_CARLA_Testing.md`
- `PROJECT_STRUCTURE.md`

## 🚀 Next Steps

The workspace is now clean and organized with:
- ✅ **12 obsolete files removed**
- ✅ **Clear separation of concerns**
- ✅ **Streamlined workflows**
- ✅ **Comprehensive documentation**

You can now focus on the core functionality without clutter!