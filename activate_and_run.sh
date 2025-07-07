#!/bin/bash
echo "🚀 Activating Ultimate Trading System..."

# Set environment
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Activate venv
source venv/bin/activate

# Show info
echo "🐍 Python: $(python --version)"
echo "🖥️ GPU: Using GPU 1 (cuda:0 in code)"
echo "📁 Working directory: $(pwd)"

echo "✅ Environment ready!"
echo ""
echo "🎯 Choose an option:"
echo "1. python start_trading.py    # Run full trading system"
echo "2. python compatibility_fixes.py  # Test compatibility only"
echo "3. python main.py            # Run original main (may have warnings)"
