# 🚀 Simplified Workflow Guide

## 🎯 **Recommended Workflow (Simplest)**

### **Option 1: Complete Automated Setup (Recommended)**
```bash
# This handles EVERYTHING automatically
python setup_three_way_comparison.py
```

**What it does:**
- ✅ Installs all dependencies (including CARLA)
- ✅ Creates necessary directories
- ✅ Automatically creates BDD100K-style dataset from your synthetic images
- ✅ Sets up all configurations
- ✅ Creates workflow scripts

**Requirements:** You need your synthetic images in `generated_images/` folder

---

## 🔧 **Alternative Workflows (Advanced Users)**

### **Option 2: Step-by-Step Manual Control**
```bash
# 1. Setup environment only
python setup_three_way_comparison.py

# 2. Create custom BDD100K dataset (optional)
python download_bdd100k_rural.py --count 10000 --auto-download

# 3. Train models
python train_three_way_models.py

# 4. Run comparison
python carla_three_way_comparison.py
```

### **Option 3: With Real BDD100K Dataset**
```bash
# If you have access to the real BDD100K dataset:
# 1. Download and extract BDD100K to 'bdd100k_full/' directory
# 2. Run the extraction script
python download_bdd100k_rural.py --count 10000

# 3. Run setup
python setup_three_way_comparison.py
```

---

## 📁 **File Usage Guide**

### **Essential Files (Always Needed)**
- `setup_three_way_comparison.py` - **Main setup script**
- `carla_three_way_comparison.py` - **Main testing script**
- `train_three_way_models.py` - **Model training**

### **Optional Files (Advanced Use Cases)**
- `download_bdd100k_rural.py` - **Optional**: For custom BDD100K handling
- `setup_carla_test.py` - **Optional**: For two-way comparison only
- `carla_synthetic_data_test.py` - **Optional**: For two-way comparison only

### **Utility Files**
- `install_carla_api.py` - **Auto-used**: CARLA installation
- `fix_numpy_opencv.py` - **As-needed**: Fix compatibility issues
- `check_carla_compatibility.py` - **As-needed**: Troubleshooting

---

## 🤔 **When to Use Which Files**

### **For Most Users (Recommended)**
```bash
# Just run this one command
python setup_three_way_comparison.py

# Then start CARLA and run comparison
./CarlaUE4.sh
python carla_three_way_comparison.py
```

### **For Research/Advanced Users**
- Use `download_bdd100k_rural.py` if you want custom BDD100K handling
- Use individual setup scripts if you need fine-grained control
- Use utility scripts for troubleshooting specific issues

### **For Two-Way Comparison Only**
```bash
# If you only want baseline vs synthetic (no BDD100K)
python setup_carla_test.py
python carla_synthetic_data_test.py
```

---

## 📊 **File Dependency Map**

```
setup_three_way_comparison.py (MAIN)
├── Uses: install_carla_api.py (auto)
├── Uses: download_bdd100k_rural.py (optional)
├── Creates: Configuration files
└── Prepares: Environment for carla_three_way_comparison.py

carla_three_way_comparison.py (MAIN)
├── Requires: Trained models (from train_three_way_models.py)
├── Requires: BDD100K data (created by setup)
└── Requires: CARLA running

train_three_way_models.py
├── Uses: generated_images/ (your synthetic data)
├── Uses: bdd100k_rural_10k/ (created by setup)
└── Creates: Model files for comparison
```

---

## 🎯 **Bottom Line**

**For 90% of users:** Just run `python setup_three_way_comparison.py` and you're done!

**The download script is kept for:**
- Advanced users who want custom BDD100K handling
- Users with access to the real BDD100K dataset
- Debugging and development purposes

**You can safely ignore it** if you just want to run the comparison with your synthetic data.