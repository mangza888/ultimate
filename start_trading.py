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
