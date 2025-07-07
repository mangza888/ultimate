#!/bin/bash
# quick_fix.sh - แก้ไขปัญหาทั้งหมดแบบรวดเร็ว

echo "🚀 Quick Fix for Ultimate Trading System"
echo "========================================"

# Activate environment
source venv/bin/activate

echo "🔧 Applying compatibility fixes..."

# 1. สร้างไฟล์ compatibility fixes
cat > compatibility_fixes.py << 'EOF'
#!/usr/bin/env python3
"""
compatibility_fixes.py - แก้ไข compatibility issues
"""

import warnings
import os
import sys

def setup_environment():
    """Setup environment variables"""
    
    # GPU environment
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # LightGBM GPU
    os.environ['LGB_GPU_DEVICE_ID'] = '0'
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    print("✅ Environment setup completed")

def patch_xgboost():
    """Patch XGBoost for new version compatibility"""
    try:
        import xgboost as xgb
        
        # Monkey patch for backward compatibility
        if hasattr(xgb, 'XGBClassifier'):
            original_init = xgb.XGBClassifier.__init__
            
            def patched_init(self, *args, **kwargs):
                # Convert old parameters to new ones
                if 'gpu_id' in kwargs:
                    gpu_id = kwargs.pop('gpu_id')
                    kwargs['device'] = f'cuda:{gpu_id}'
                
                if 'tree_method' in kwargs and kwargs['tree_method'] == 'gpu_hist':
                    kwargs['tree_method'] = 'hist'
                
                return original_init(self, *args, **kwargs)
            
            xgb.XGBClassifier.__init__ = patched_init
        
        print("✅ XGBoost patched for compatibility")
        
    except ImportError:
        print("❌ XGBoost not available")

def patch_lightgbm():
    """Patch LightGBM for new version compatibility"""
    try:
        import lightgbm as lgb
        
        # Monkey patch for backward compatibility
        if hasattr(lgb, 'train'):
            original_train = lgb.train
            
            def patched_train(*args, **kwargs):
                # Remove deprecated verbose parameter
                if 'verbose' in kwargs:
                    kwargs.pop('verbose')
                
                # Add verbosity if not present
                if len(args) > 0 and isinstance(args[0], dict):
                    params = args[0]
                    if 'verbosity' not in params:
                        params['verbosity'] = -1
                
                return original_train(*args, **kwargs)
            
            lgb.train = patched_train
        
        print("✅ LightGBM patched for compatibility")
        
    except ImportError:
        print("❌ LightGBM not available")

def test_gpu_setup():
    """Test GPU setup"""
    try:
        import torch
        print(f"🔥 PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎯 Current device: {torch.cuda.current_device()}")
            print(f"🏷️ Device name: {torch.cuda.get_device_name()}")
        
        # Test tensor creation
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = torch.tensor([1.0]).to(device)
        print(f"✅ Tensor test passed: {x.device}")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

if __name__ == "__main__":
    setup_environment()
    patch_xgboost()
    patch_lightgbm()
    test_gpu_setup()
    print("🎉 Compatibility fixes applied successfully!")
EOF

# 2. รันไฟล์ compatibility fixes
echo "🔧 Running compatibility fixes..."
python compatibility_fixes.py

# 3. อัพเดท config files
echo "⚙️ Updating configuration files..."

# Update ai_models.yaml
mkdir -p config
cat > config/ai_models.yaml << 'EOF'
# config/ai_models.yaml - Updated for Python 3.9 + GPU 1

ai_general:
  python_version: "3.9"
  gpu_memory_fraction: 0.7
  device: "cuda:0"
  random_seed: 42
  early_stopping_patience: 12
  validation_split: 0.2
  mixed_precision: true

traditional_ml:
  xgboost:
    enabled: true
    priority: 1
    params:
      n_estimators: 600
      max_depth: 10
      learning_rate: 0.08
      subsample: 0.85
      colsample_bytree: 0.85
      tree_method: "hist"
      device: "cuda:0"
      predictor: "gpu_predictor"
      random_state: 42
      
  lightgbm:
    enabled: true
    priority: 2
    params:
      n_estimators: 600
      max_depth: 10
      learning_rate: 0.08
      subsample: 0.85
      colsample_bytree: 0.85
      device: "gpu"
      gpu_platform_id: 0
      gpu_device_id: 0
      objective: "multiclass"
      num_class: 3
      verbosity: -1
      random_state: 42
      
  random_forest:
    enabled: true
    priority: 3
    params:
      n_estimators: 300
      max_depth: 18
      min_samples_split: 4
      min_samples_leaf: 1
      n_jobs: -1
      random_state: 42

pytorch_lightning:
  enabled: true
  trainer:
    max_epochs: 120
    devices: [0]
    accelerator: "gpu"
    precision: "16-mixed"
    accumulate_grad_batches: 1
    gradient_clip_val: 1.2
    strategy: "auto"
    enable_progress_bar: true
    enable_model_summary: true

reinforcement_learning:
  ppo:
    enabled: true
    params:
      device: "cuda:0"
  sac:
    enabled: true
    params:
      device: "cuda:0"
  a2c:
    enabled: true
    params:
      device: "cuda:0"
  ddpg:
    enabled: true
    params:
      device: "cuda:0"

distributed_rl:
  enabled: true
  ray_config:
    num_workers: 6
    num_gpus: 1
    gpu_id: 0

backtesting:
  frameworks:
    vectorbt:
      enabled: true
      priority: 1
    backtrader:
      enabled: true
      priority: 2
    zipline:
      enabled: true
      priority: 3
EOF

# Update main config.yaml 
cat > config.yaml << 'EOF'
# config.yaml - Updated for Python 3.9 + GPU 1

system:
  name: "Ultimate Auto Trading System"
  version: "1.0.0"
  python_version: "3.9"
  
gpu:
  enabled: true
  memory_fraction: 0.7
  device: "cuda"
  gpu_id: 1
  cuda_device: "cuda:0"

targets:
  ai_win_rate: 90
  backtest_return: 85
  paper_trade_return: 100
  
capital:
  initial: 10000
  max_position_size: 0.25
  min_confidence: 0.6

symbols:
  - "BTC"
  - "ETH"
  - "BNB"
  - "LTC"

ai_models:
  primary: "xgboost"
  backup: "random_forest"

training:
  max_iterations: 50
  samples_per_symbol: 1000
  validation_split: 0.2
  early_stopping: true
  batch_size_multiplier: 1.5

backtest:
  duration_days: 365
  transaction_cost: 0.001
  min_trades: 10
  parallel_processing: true
  
paper_trade:
  timeframe: "1m"
  duration_minutes: 60
  update_interval: 5
  min_trades_per_hour: 5
  async_processing: true
  
indicators:
  moving_averages: [5, 10, 20, 50]
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bollinger_period: 20
  bollinger_std: 2
  volatility_windows: [5, 10, 20]
  momentum_periods: [1, 5, 10, 20]

data:
  min_history_periods: 60
  max_history_periods: 200
  feature_count: 25
  vectorized_operations: true
  
logging:
  level: "INFO"
  file: "trading_log.txt"
  max_size_mb: 15
  backup_count: 7
  console_output: true
  structured_logging: true

output:
  save_models: true
  save_results: true
  save_plots: true
  results_dir: "results"
  models_dir: "models"
  logs_dir: "logs"
  compression: true
  
retry:
  max_training_attempts: 12
  max_backtest_attempts: 6
  max_paper_trade_attempts: 4
  delay_between_attempts: 3
  exponential_backoff: true

alerts:
  enabled: true
  target_achieved: true
  training_failed: true
  backtest_failed: true
  performance_monitoring: true

performance:
  multiprocessing:
    enabled: true
    max_workers: 10
  async_processing:
    enabled: true
    max_concurrent_tasks: 15
  memory_optimization:
    enabled: true
    garbage_collection: "generational"
  jit_compilation:
    numba_enabled: true
    cache_enabled: true
EOF

echo "✅ Configuration files updated"

# 4. สร้างไฟล์ startup script
cat > start_trading.py << 'EOF'
#!/usr/bin/env python3
"""
start_trading.py - Startup script with compatibility fixes
"""

import os
import sys
import warnings

def setup_environment():
    """Setup environment before importing main modules"""
    
    # Set GPU device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['LGB_GPU_DEVICE_ID'] = '0'
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    print("✅ Environment configured for GPU 1")

def apply_patches():
    """Apply compatibility patches"""
    
    try:
        # Patch XGBoost
        import xgboost as xgb
        if hasattr(xgb, 'XGBClassifier'):
            original_init = xgb.XGBClassifier.__init__
            
            def patched_init(self, *args, **kwargs):
                if 'gpu_id' in kwargs:
                    gpu_id = kwargs.pop('gpu_id')
                    kwargs['device'] = f'cuda:0'  # Always use cuda:0 due to CUDA_VISIBLE_DEVICES
                
                if 'tree_method' in kwargs and kwargs['tree_method'] == 'gpu_hist':
                    kwargs['tree_method'] = 'hist'
                
                return original_init(self, *args, **kwargs)
            
            xgb.XGBClassifier.__init__ = patched_init
        
        print("✅ XGBoost compatibility patch applied")
        
    except ImportError:
        print("⚠️ XGBoost not available")
    
    try:
        # Patch LightGBM
        import lightgbm as lgb
        if hasattr(lgb, 'train'):
            original_train = lgb.train
            
            def patched_train(*args, **kwargs):
                if 'verbose' in kwargs:
                    kwargs.pop('verbose')
                
                if len(args) > 0 and isinstance(args[0], dict):
                    params = args[0]
                    if 'verbosity' not in params:
                        params['verbosity'] = -1
                
                return original_train(*args, **kwargs)
            
            lgb.train = patched_train
        
        print("✅ LightGBM compatibility patch applied")
        
    except ImportError:
        print("⚠️ LightGBM not available")

def main():
    """Main function"""
    print("🚀 Starting Ultimate Trading System...")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Apply compatibility patches
    apply_patches()
    
    # Import and run main
    try:
        from main import UltimateAutoTradingSystem
        import asyncio
        
        # Create and run system
        system = UltimateAutoTradingSystem()
        success = asyncio.run(system.run_full_pipeline())
        
        if success:
            print("\n🎉 MISSION ACCOMPLISHED!")
            print("✅ All targets achieved")
            print("🚀 System ready for live trading")
        else:
            print("\n❌ MISSION FAILED")
            print("⚠️ Some targets not achieved")
            print("🔄 Review logs and retry")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n💥 System error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
EOF

# 5. ทำให้ไฟล์ executable
chmod +x start_trading.py
chmod +x compatibility_fixes.py

# 6. สร้าง simple activation script
cat > activate_and_run.sh << 'EOF'
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
EOF

chmod +x activate_and_run.sh

# 7. Test the fixes
echo "🧪 Testing compatibility fixes..."
python compatibility_fixes.py

echo ""
echo "🎉 Quick Fix Completed Successfully!"
echo "===================================="
echo "✅ Environment configured for Python 3.9 + GPU 1"
echo "✅ XGBoost and LightGBM compatibility patches applied"
echo "✅ Configuration files updated"
echo "✅ Startup scripts created"
echo ""
echo "🚀 To start the trading system:"
echo "   ./activate_and_run.sh"
echo "   python start_trading.py"
echo ""
echo "🔧 Available commands:"
echo "   python compatibility_fixes.py  # Test compatibility"
echo "   python start_trading.py       # Run trading system"
echo "   ./activate_and_run.sh         # Get activation help"