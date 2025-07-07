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
