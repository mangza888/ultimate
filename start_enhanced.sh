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
