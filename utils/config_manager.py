#!/usr/bin/env python3
# utils/config_manager.py - Configuration Manager
# จัดการการตั้งค่าทั้งหมด

import yaml
import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """จัดการการตั้งค่าของระบบ"""
    
    def __init__(self, config_path="config.yaml"):
        """เริ่มต้น ConfigManager"""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """โหลด configuration จากไฟล์"""
        try:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            elif self.config_path.endswith('.json'):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {self.config_path}")
                
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {self.config_path}")
            return self._create_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """สร้าง default configuration"""
        default_config = {
            'system': {
                'name': 'Ultimate Auto Trading System',
                'version': '1.0.0'
            },
            'gpu': {
                'enabled': True,
                'memory_fraction': 0.6,
                'device': 'cuda'
            },
            'targets': {
                'ai_win_rate': 90,
                'backtest_return': 85,
                'paper_trade_return': 100
            },
            'capital': {
                'initial': 10000,
                'max_position_size': 0.25,
                'min_confidence': 0.6
            },
            'symbols': ['BTC', 'ETH', 'BNB', 'LTC'],
            'ai_models': {
                'primary': 'xgboost',
                'backup': 'random_forest'
            },
            'training': {
                'max_iterations': 50,
                'samples_per_symbol': 1000,
                'validation_split': 0.2,
                'early_stopping': True
            },
            'backtest': {
                'duration_days': 365,
                'transaction_cost': 0.001,
                'min_trades': 10
            },
            'paper_trade': {
                'timeframe': '1m',
                'duration_minutes': 60,
                'update_interval': 5,
                'min_trades_per_hour': 5
            },
            'indicators': {
                'moving_averages': [5, 10, 20, 50],
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_period': 20,
                'bollinger_std': 2,
                'volatility_windows': [5, 10, 20],
                'momentum_periods': [1, 5, 10, 20]
            },
            'data': {
                'min_history_periods': 60,
                'max_history_periods': 200,
                'feature_count': 22
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading_log.txt',
                'max_size_mb': 10,
                'backup_count': 5,
                'console_output': True
            },
            'output': {
                'save_models': True,
                'save_results': True,
                'save_plots': True,
                'results_dir': 'results',
                'models_dir': 'models',
                'logs_dir': 'logs'
            },
            'retry': {
                'max_training_attempts': 10,
                'max_backtest_attempts': 5,
                'max_paper_trade_attempts': 3,
                'delay_between_attempts': 5
            },
            'alerts': {
                'enabled': True,
                'target_achieved': True,
                'training_failed': True,
                'backtest_failed': True
            }
        }
        
        # บันทึก default config
        self.save_config(default_config)
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """ดึงค่า configuration ด้วย dot notation"""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
                    
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """ตั้งค่า configuration ด้วย dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """อัพเดท configuration แบบ batch"""
        for key, value in updates.items():
            self.set(key, value)
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """บันทึก configuration ลงไฟล์"""
        if config is None:
            config = self.config
            
        try:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif self.config_path.endswith('.json'):
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
            
            print(f"✅ Config saved to: {self.config_path}")
            
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def reload(self) -> None:
        """โหลด configuration ใหม่"""
        self.config = self._load_config()
        print(f"🔄 Config reloaded from: {self.config_path}")
    
    def create_directories(self) -> None:
        """สร้าง directories ที่จำเป็น"""
        directories = [
            self.get('output.results_dir', 'results'),
            self.get('output.models_dir', 'models'),
            self.get('output.logs_dir', 'logs')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def get_system_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลระบบ"""
        return {
            'name': self.get('system.name'),
            'version': self.get('system.version'),
            'gpu_enabled': self.get('gpu.enabled'),
            'gpu_memory_fraction': self.get('gpu.memory_fraction'),
            'initial_capital': self.get('capital.initial'),
            'symbols': self.get('symbols'),
            'targets': self.get('targets')
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """ดึงการตั้งค่าสำหรับการเทรน"""
        return {
            'max_iterations': self.get('training.max_iterations'),
            'samples_per_symbol': self.get('training.samples_per_symbol'),
            'validation_split': self.get('training.validation_split'),
            'early_stopping': self.get('training.early_stopping'),
            'target_win_rate': self.get('targets.ai_win_rate'),
            'symbols': self.get('symbols'),
            'model_config': self.get('ai_models')
        }
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """ดึงการตั้งค่าสำหรับ backtest"""
        return {
            'duration_days': self.get('backtest.duration_days'),
            'transaction_cost': self.get('backtest.transaction_cost'),
            'min_trades': self.get('backtest.min_trades'),
            'target_return': self.get('targets.backtest_return'),
            'symbols': self.get('symbols'),
            'initial_capital': self.get('capital.initial')
        }
    
    def get_paper_trade_config(self) -> Dict[str, Any]:
        """ดึงการตั้งค่าสำหรับ paper trading"""
        return {
            'timeframe': self.get('paper_trade.timeframe'),
            'duration_minutes': self.get('paper_trade.duration_minutes'),
            'update_interval': self.get('paper_trade.update_interval'),
            'min_trades_per_hour': self.get('paper_trade.min_trades_per_hour'),
            'target_return': self.get('targets.paper_trade_return'),
            'initial_capital': self.get('capital.initial'),
            'max_position_size': self.get('capital.max_position_size'),
            'min_confidence': self.get('capital.min_confidence'),
            'symbols': self.get('symbols')
        }
    
    def get_indicators_config(self) -> Dict[str, Any]:
        """ดึงการตั้งค่าสำหรับ technical indicators"""
        return self.get('indicators')
    
    def validate_config(self) -> bool:
        """ตรวจสอบความถูกต้องของ configuration"""
        try:
            # ตรวจสอบค่าที่จำเป็น
            required_keys = [
                'targets.ai_win_rate',
                'targets.backtest_return', 
                'targets.paper_trade_return',
                'capital.initial',
                'symbols'
            ]
            
            for key in required_keys:
                if self.get(key) is None:
                    print(f"❌ Missing required config: {key}")
                    return False
            
            # ตรวจสอบค่าที่สมเหตุสมผล
            if self.get('capital.initial') <= 0:
                print("❌ Initial capital must be positive")
                return False
                
            if not self.get('symbols'):
                print("❌ No symbols specified")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Config validation error: {e}")
            return False

# Singleton instance
_config_instance = None

def get_config(config_path="config.yaml"):
    """Get singleton config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance