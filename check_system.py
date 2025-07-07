#!/usr/bin/env python3
import sys
import importlib

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: Not installed")
        return False

def main():
    print("🔍 Enhanced System Check")
    print("=" * 30)
    
    # Check Python version
    version = sys.version_info
    if version >= (3, 9):
        print(f"✅ Python: {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python: {version.major}.{version.minor}.{version.micro} (Need 3.9+)")
        return False
    
    # Core packages
    core_ok = True
    core_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('torch', 'torch'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('optuna', 'optuna'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly')
    ]
    
    print("\n📦 Core Packages:")
    for name, import_name in core_packages:
        if not check_package(name, import_name):
            core_ok = False
    
    # Enhanced packages (optional)
    print("\n🚀 Enhanced Packages (Optional):")
    enhanced_packages = [
        ('pytorch-lightning', 'pytorch_lightning'),
        ('transformers', 'transformers'),
        ('vectorbt', 'vectorbt'),
        ('backtrader', 'backtrader'),
        ('pandas-ta', 'pandas_ta')
    ]
    
    for name, import_name in enhanced_packages:
        check_package(name, import_name)
    
    # CUDA check
    print("\n🖥️  System:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available ({torch.cuda.get_device_name()})")
        else:
            print("⚠️  CUDA: Not available (CPU only)")
    except:
        print("❌ CUDA: Cannot check")
    
    print("\n" + "=" * 30)
    if core_ok:
        print("✅ System ready!")
        return True
    else:
        print("❌ System not ready - install missing packages")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1)
