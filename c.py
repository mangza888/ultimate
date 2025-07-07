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
