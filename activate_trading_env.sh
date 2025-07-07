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
