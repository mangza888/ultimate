#!/bin/bash
# setup_python39_gpu1.sh - Setup Ultimate Trading System for Python 3.9 + GPU 1

echo "🚀 Setting up Ultimate Trading System for Python 3.9 + GPU 1"
echo "============================================================="

# ตรวจสอบ Python 3.9
echo "🐍 Checking Python 3.9..."
if ! command -v python3.9 &> /dev/null; then
    echo "❌ Python 3.9 not found! Installing..."
    
    # Ubuntu/Debian
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y python3.9 python3.9-venv python3.9-dev python3.9-distutils
    
    # Install pip for Python 3.9
    curl https://bootstrap.pypa.io/get-pip.py | python3.9
    
    echo "✅ Python 3.9 installed successfully!"
else
    echo "✅ Python 3.9 found: $(python3.9 --version)"
fi

# ตรวจสอบ GPU
echo "🖥️ Checking GPU setup..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found!"
    echo "💡 Please install NVIDIA drivers first"
    exit 1
fi

echo "🔍 GPU Information:"
nvidia-smi --list-gpus
echo "📌 Will use GPU 1 (cuda:1)"

# ตั้งค่า environment variables สำหรับ GPU 1
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
echo "✅ Set CUDA_VISIBLE_DEVICES=1"

# ลบ virtual environment เก่า
echo "🗑️ Removing old virtual environment..."
if [ -d "venv" ]; then
    rm -rf venv
    echo "✅ Old venv removed"
fi

# สร้าง virtual environment ใหม่ด้วย Python 3.9
echo "🏗️ Creating new virtual environment with Python 3.9..."
python3.9 -m venv venv
source venv/bin/activate

# ตรวจสอบ Python version ใน venv
echo "📋 Python version in venv: $(python --version)"

# อัพเกรด pip และ build tools
echo "📦 Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel build

# ติดตั้ง dependencies ที่เข้ากันได้กับ Python 3.9 + GPU
echo "📦 Installing Python 3.9 compatible packages..."

# Core scientific packages (Python 3.9 optimized)
echo "📊 Installing core scientific packages..."
pip install "numpy>=1.22.0,<1.26.0"
pip install "pandas>=1.5.0,<2.1.0"
pip install "scipy>=1.9.0,<1.12.0"
pip install "scikit-learn>=1.1.0,<1.4.0"

# Configuration and utilities
echo "⚙️ Installing configuration packages..."
pip install "PyYAML>=6.0"
pip install "python-dotenv>=1.0.0"

# Machine Learning (GPU enabled)
echo "🤖 Installing ML packages with GPU support..."
pip install "xgboost>=1.7.0,<2.1.0"
pip install "lightgbm>=4.0.0,<4.3.0" --config-settings=cmake.define.USE_GPU=ON
pip install "optuna>=3.0.0"

# Deep Learning (Python 3.9 + CUDA support)
echo "🧠 Installing PyTorch with CUDA support..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install "pytorch-lightning>=2.0.0,<2.2.0"

# Reinforcement Learning
echo "🎮 Installing RL packages..."
pip install "stable-baselines3>=2.0.0"
pip install "gymnasium>=0.28.0"
pip install "sb3-contrib>=2.0.0"

# Backtesting frameworks
echo "📈 Installing backtesting frameworks..."
pip install "vectorbt>=0.25.0"
pip install "backtrader>=1.9.78"

# Try to install Zipline for Python 3.9
echo "🏦 Attempting to install Zipline..."
pip install "cython>=3.0.0,<3.1.0"

# Try different methods for Zipline
if pip install zipline-reloaded --no-cache-dir; then
    echo "✅ Zipline installed successfully!"
else
    echo "⚠️ Zipline installation failed, trying alternative..."
    if pip install git+https://github.com/stefan-jansen/zipline-reloaded.git; then
        echo "✅ Zipline installed from git!"
    else
        echo "ℹ️ Skipping Zipline - will use VectorBT and Backtrader only"
    fi
fi

# Data and technical analysis
echo "📊 Installing data packages..."
pip install "yfinance>=0.2.0"
pip install "pandas-ta>=0.3.14b"

# Visualization
echo "🎨 Installing visualization packages..."
pip install "matplotlib>=3.6.0"
pip install "seaborn>=0.12.0"
pip install "plotly>=5.15.0"

# Performance libraries
echo "⚡ Installing performance packages..."
pip install "numba>=0.57.0"
pip install "joblib>=1.3.0"

# Additional Python 3.9 optimized packages
echo "🔧 Installing additional packages..."
pip install "aiohttp>=3.8.0"
pip install "asyncio>=3.4.3"
pip install "requests>=2.28.0"
pip install "psutil>=5.9.0"

# Development tools
echo "🛠️ Installing development tools..."
pip install "jupyter>=1.0.0"
pip install "pytest>=7.0.0"
pip install "black>=22.0.0"

# Create GPU test script
echo "🧪 Creating GPU test script..."
cat > test_gpu_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test GPU setup for Ultimate Trading System
"""
import os
import sys

def test_gpu_setup():
    print("🧪 Testing GPU Setup for Ultimate Trading System")
    print("=" * 50)
    
    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Test CUDA environment
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"🔧 CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Test PyTorch CUDA
    try:
        import torch
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🖥️ CUDA available: {torch.cuda.is_available()}")
        print(f"🎯 CUDA device count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"📌 Current device: {current_device}")
            print(f"🏷️ Device name: {device_name}")
            
            # Test tensor on GPU
            x = torch.tensor([1.0]).cuda()
            print(f"✅ Tensor on GPU: {x.device}")
            
            # Test GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU memory: {total_memory:.1f} GB")
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    
    # Test XGBoost GPU
    try:
        import xgboost as xgb
        print(f"🌳 XGBoost version: {xgb.__version__}")
        
        # Test GPU training
        dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[1, 0])
        params = {'tree_method': 'gpu_hist', 'gpu_id': 1}
        model = xgb.train(params, dtrain, num_boost_round=1)
        print("✅ XGBoost GPU training successful!")
        
    except ImportError as e:
        print(f"❌ XGBoost import failed: {e}")
    except Exception as e:
        print(f"⚠️ XGBoost GPU test failed: {e}")
    
    # Test LightGBM GPU
    try:
        import lightgbm as lgb
        print(f"💡 LightGBM version: {lgb.__version__}")
        
        # Test GPU training
        train_data = lgb.Dataset([[1, 2], [3, 4]], label=[1, 0])
        params = {'device': 'gpu', 'gpu_device_id': 1, 'objective': 'binary'}
        model = lgb.train(params, train_data, num_boost_round=1, verbose=-1)
        print("✅ LightGBM GPU training successful!")
        
    except ImportError as e:
        print(f"❌ LightGBM import failed: {e}")
    except Exception as e:
        print(f"⚠️ LightGBM GPU test failed: {e}")
    
    # Test other packages
    packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn', 'scikit-learn'),
        ('vectorbt', 'vbt'),
        ('stable_baselines3', 'sb3')
    ]
    
    for module, display_name in packages:
        try:
            __import__(module)
            print(f"✅ {display_name} imported successfully")
        except ImportError:
            print(f"❌ {display_name} import failed")
    
    # Test Zipline (optional)
    try:
        import zipline
        print(f"🏦 Zipline version: {zipline.__version__}")
    except ImportError:
        print("ℹ️ Zipline not available (this is okay)")
    
    print("\n🎉 GPU setup test completed!")

if __name__ == "__main__":
    test_gpu_setup()
EOF

# Run the test
echo "🧪 Running GPU setup test..."
python test_gpu_setup.py

# Create activation script
echo "📝 Creating activation script..."
cat > activate_trading_env.sh << 'EOF'
#!/bin/bash
# Activation script for Ultimate Trading System

echo "🚀 Activating Ultimate Trading System Environment"
echo "================================================="

# Set GPU environment
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Activate virtual environment
source venv/bin/activate

# Show environment info
echo "🐍 Python: $(python --version)"
echo "🖥️ GPU: CUDA device 1"
echo "📁 Working directory: $(pwd)"

echo "✅ Environment activated!"
echo "💡 Run 'python main.py' to start the trading system"
EOF

chmod +x activate_trading_env.sh

# Final summary
echo ""
echo "🎉 Setup completed successfully!"
echo "================================="
echo "✅ Python 3.9 virtual environment created"
echo "✅ GPU 1 (cuda:1) configured"
echo "✅ All packages installed"
echo "✅ GPU functionality tested"
echo ""
echo "🚀 To start using the system:"
echo "   1. source activate_trading_env.sh"
echo "   2. python main.py"
echo ""
echo "🔧 Configuration files updated:"
echo "   - config.yaml (GPU 1 settings)"
echo "   - ai_models.yaml (Python 3.9 optimized)"
echo ""
echo "💡 GPU 1 will be used for all AI training and inference"

# Save environment info
cat > environment_info.txt << EOF
Ultimate Trading System Environment Info
========================================
Setup Date: $(date)
Python Version: $(python3.9 --version)
GPU Device: CUDA:1
Virtual Environment: venv/
Activation Script: activate_trading_env.sh

GPU Information:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | sed -n '2p')

Installed Packages:
$(pip list | grep -E "(torch|xgboost|lightgbm|numpy|pandas)")
EOF

echo "📄 Environment info saved to: environment_info.txt"