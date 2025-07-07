#!/bin/bash
# quick_install_enhanced.sh - Quick Installation Script

echo "🚀 Quick Enhanced System Installation"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Please run this script in the project root directory (where main.py exists)"
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Step 1: Create enhanced virtual environment
echo "🐍 Creating enhanced virtual environment..."
if [ ! -d "venv_enhanced" ]; then
    python3 -m venv venv_enhanced
    echo "✅ Enhanced virtual environment created"
else
    echo "ℹ️  Enhanced virtual environment already exists"
fi

# Activate environment
source venv_enhanced/bin/activate
echo "✅ Enhanced virtual environment activated"

# Step 2: Upgrade pip and install core packages
echo "📦 Installing core enhanced packages..."
pip install --upgrade pip setuptools wheel

# Core ML packages
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 scipy>=1.11.0
pip install torch>=2.0.0 xgboost>=2.0.0 lightgbm>=4.0.0

# Optimization packages
pip install optuna>=3.3.0 hyperopt>=0.2.7

# Visualization
pip install matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.15.0

# Data sources
pip install yfinance>=0.2.18 requests>=2.31.0 aiohttp>=3.8.0

# Configuration
pip install PyYAML>=6.0 python-dotenv>=1.0.0

echo "✅ Core enhanced packages installed"

# Step 3: Install optional advanced packages (with error handling)
echo "🚀 Installing advanced packages..."

# Try to install advanced packages one by one
advanced_packages=(
    "pytorch-lightning>=2.0.0"
    "transformers>=4.30.0" 
    "vectorbt>=0.25.0"
    "backtrader>=1.9.78"
    "pandas-ta>=0.3.14b"
    "psutil>=5.9.0"
    "joblib>=1.3.0"
    "numba>=0.57.0"
)

for package in "${advanced_packages[@]}"; do
    echo "📦 Installing $package..."
    if pip install "$package"; then
        echo "✅ $package installed successfully"
    else
        echo "⚠️  Failed to install $package (skipping)"
    fi
done

# Step 4: Create enhanced directory structure
echo "📁 Creating enhanced directory structure..."
directories=(
    "data/external"
    "features"
    "optimization" 
    "models/advanced"
    "results/hyperopt"
    "results/features"
    "results/models"
    "saved_models/enhanced_models"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "📁 Created: $dir"
done

# Create __init__.py files
init_dirs=("data" "features" "optimization" "models")
for dir in "${init_dirs[@]}"; do
    touch "$dir/__init__.py"
    echo "📄 Created: $dir/__init__.py"
done

echo "✅ Enhanced directory structure created"

# Step 5: Create enhanced config
echo "⚙️  Creating enhanced configuration..."
cat > config_enhanced.yaml << 'EOF'
# Enhanced Configuration
system:
  name: "Enhanced Ultimate Auto Trading System"
  version: "2.0.0"
  enhanced_features: true

gpu:
  enabled: true
  memory_fraction: 0.7

targets:
  ai_win_rate: 90
  backtest_return: 85
  paper_trade_return: 100

data:
  multi_timeframe: true
  timeframes: ["1m", "5m", "15m", "1h"]
  symbols: ["BTC/USDT", "ETH/USDT", "LTC/USDT"]
  enhanced_features: true

hyperparameter_optimization:
  enabled: true
  method: "optuna" 
  n_trials: 50
  models_to_optimize: ["xgboost", "lightgbm"]

advanced_models:
  transformer:
    enabled: true
  ensemble:
    enabled: true
EOF

echo "✅ Enhanced configuration created: config_enhanced.yaml"

# Step 6: Create startup scripts
echo "🚀 Creating startup scripts..."

# Enhanced startup script
cat > start_enhanced.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Enhanced Ultimate Auto Trading System"

# Activate enhanced environment
if [ -f "venv_enhanced/bin/activate" ]; then
    source venv_enhanced/bin/activate
    echo "✅ Enhanced environment activated"
else
    echo "❌ Enhanced environment not found"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=1

# Run enhanced system
if [ -f "main_enhanced.py" ]; then
    echo "🔥 Running Enhanced System..."
    python main_enhanced.py
elif [ -f "main.py" ]; then
    echo "⚠️  Running Original System (Enhanced not found)..."
    python main.py
else
    echo "❌ No main file found"
    exit 1
fi
EOF

chmod +x start_enhanced.sh

# Original startup script (fallback)
cat > start_original.sh << 'EOF'
#!/bin/bash
echo "🔄 Starting Original Ultimate Auto Trading System"

# Activate original environment  
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv_enhanced/bin/activate" ]; then
    source venv_enhanced/bin/activate
else
    echo "❌ No virtual environment found"
    exit 1
fi

# Run original system
if [ -f "main.py" ]; then
    python main.py
else
    echo "❌ main.py not found"
    exit 1
fi
EOF

chmod +x start_original.sh

echo "✅ Startup scripts created"

# Step 7: Create check script
echo "🔍 Creating requirements check script..."
cat > check_system.py << 'EOF'
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
EOF

echo "✅ Check script created: check_system.py"

# Step 8: Final setup
echo "🎯 Final setup..."

# Create a simple enhanced main file if it doesn't exist
if [ ! -f "main_enhanced.py" ]; then
    echo "📝 Creating simple enhanced main file..."
    cat > main_enhanced.py << 'EOF'
#!/usr/bin/env python3
"""
main_enhanced.py - Enhanced Ultimate Auto Trading System
This is a placeholder. Replace with the actual enhanced main file.
"""

import asyncio
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def main():
    print("🚀 Enhanced Ultimate Auto Trading System")
    print("=" * 50)
    print("⚠️  This is a placeholder enhanced main file.")
    print("📝 Please replace with the actual enhanced implementation.")
    print("🔄 Falling back to original system...")
    
    # Try to import and run original system
    try:
        original_main = __import__('main')
        if hasattr(original_main, 'main'):
            return await original_main.main()
        else:
            print("❌ Original main function not found")
            return 1
    except ImportError:
        print("❌ Original main.py not found")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
EOF
    echo "✅ Placeholder enhanced main created"
fi

echo ""
echo "🎉 Enhanced System Installation Complete!"
echo "========================================"
echo ""
echo "📋 Next Steps:"
echo "1. Check system: python check_system.py"
echo "2. Replace main_enhanced.py with actual enhanced implementation"
echo "3. Place enhanced files in correct directories:"
echo "   - data/enhanced_data_manager.py"
echo "   - features/advanced_feature_engineering.py" 
echo "   - optimization/hyperparameter_optimization.py"
echo "   - models/advanced_architectures.py"
echo "4. Configure: config_enhanced.yaml"
echo "5. Start enhanced system: ./start_enhanced.sh"
echo "   OR start original system: ./start_original.sh"
echo ""
echo "🔧 Available Commands:"
echo "   python check_system.py          # Check requirements"
echo "   ./start_enhanced.sh             # Start enhanced system"
echo "   ./start_original.sh             # Start original system"
echo "   source venv_enhanced/bin/activate  # Activate enhanced environment"
echo ""
echo "🚀 Ready for enhanced trading!"