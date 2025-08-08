#!/usr/bin/env python3
"""
Quick dependency checker for 10K image generation
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed and working"""
    if import_name is None:
        import_name = package_name
    
    try:
        # Try to import
        result = subprocess.run([sys.executable, "-c", f"import {import_name}; print(f'{package_name}: OK')"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name}: {e.stderr.strip() if e.stderr else 'Import failed'}")
        return False

def check_torch_details():
    """Check PyTorch specific details"""
    try:
        result = subprocess.run([sys.executable, "-c", """
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"""], capture_output=True, text=True, check=True)
        print("🎮 PyTorch Details:")
        for line in result.stdout.strip().split('\n'):
            print(f"   {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch details check failed: {e.stderr}")
        return False

def main():
    print("🔍 Dependency Check for 10K Image Generation")
    print("=" * 50)
    
    # Core packages
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"), 
        ("torchaudio", "torchaudio"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("PIL", "PIL"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm")
    ]
    
    print("📦 Checking required packages:")
    working_count = 0
    for package, import_name in packages:
        if check_package(package, import_name):
            working_count += 1
    
    print(f"\n📊 Status: {working_count}/{len(packages)} packages working")
    
    # Check PyTorch details if available
    if working_count > 0:
        print("\n" + "="*30)
        check_torch_details()
    
    # Optional packages
    print("\n🔧 Checking optional packages:")
    optional = [("xformers", "xformers")]
    for package, import_name in optional:
        check_package(package, import_name)
    
    if working_count == len(packages):
        print("\n🎉 All dependencies are working!")
        print("You can proceed with image generation.")
    else:
        print(f"\n⚠️ {len(packages) - working_count} packages need to be installed/fixed")
        print("Run: python setup_environment.py")

if __name__ == "__main__":
    main()