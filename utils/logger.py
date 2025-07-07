#!/usr/bin/env python3
# utils/logger.py - Logging Module
# ระบบ logging สำหรับ Ultimate Auto Trading

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import yaml

class TradingLogger:
    """ระบบ logging สำหรับการเทรด"""
    
    def __init__(self, config_path="config.yaml"):
        """เริ่มต้น logger"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path):
        """โหลด configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'logging': {
                'level': 'INFO',
                'file': 'trading_log.txt',
                'max_size_mb': 10,
                'backup_count': 5,
                'console_output': True
            },
            'output': {
                'logs_dir': 'logs'
            }
        }
    
    def _setup_logger(self):
        """ตั้งค่า logger"""
        # สร้าง logs directory
        logs_dir = self.config['output']['logs_dir']
        os.makedirs(logs_dir, exist_ok=True)
        
        # ตั้งค่า logger
        logger = logging.getLogger('TradingSystem')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # เคลียร์ handlers เดิม
        logger.handlers.clear()
        
        # File handler (หมุนไฟล์เมื่อใหญ่เกินไป)
        log_file = os.path.join(logs_dir, self.config['logging']['file'])
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config['logging']['max_size_mb'] * 1024 * 1024,
            backupCount=self.config['logging']['backup_count']
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # เพิ่ม handlers
        logger.addHandler(file_handler)
        
        if self.config['logging']['console_output']:
            logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message, emoji="ℹ️"):
        """Log info message"""
        self.logger.info(f"{emoji} {message}")
        
    def success(self, message, emoji="✅"):
        """Log success message"""
        self.logger.info(f"{emoji} {message}")
        
    def warning(self, message, emoji="⚠️"):
        """Log warning message"""
        self.logger.warning(f"{emoji} {message}")
        
    def error(self, message, emoji="❌"):
        """Log error message"""
        self.logger.error(f"{emoji} {message}")
        
    def debug(self, message, emoji="🔍"):
        """Log debug message"""
        self.logger.debug(f"{emoji} {message}")
        
    def trading_action(self, action, symbol, price, quantity, confidence):
        """Log trading action"""
        self.logger.info(f"📊 {action} {quantity:.4f} {symbol} at ${price:.2f} (Confidence: {confidence:.2f})")
        
    def model_performance(self, model_name, win_rate, accuracy):
        """Log model performance"""
        self.logger.info(f"🤖 {model_name}: Win Rate {win_rate:.2f}%, Accuracy {accuracy:.3f}")
        
    def backtest_result(self, symbol, initial, final, return_pct, trades):
        """Log backtest result"""
        self.logger.info(f"🔙 Backtest {symbol}: ${initial:,.2f} → ${final:,.2f} ({return_pct:+.2f}%) | {trades} trades")
        
    def paper_trade_result(self, initial, final, return_pct, trades, duration):
        """Log paper trade result"""
        self.logger.info(f"📄 Paper Trade: ${initial:,.2f} → ${final:,.2f} ({return_pct:+.2f}%) | {trades} trades | {duration}")
        
    def system_status(self, status, details=""):
        """Log system status"""
        emoji = "🚀" if status == "running" else "⏹️" if status == "stopped" else "⚙️"
        self.logger.info(f"{emoji} System {status} {details}")
        
    def gpu_status(self, gpu_name, memory_used, memory_total):
        """Log GPU status"""
        self.logger.info(f"🖥️ GPU: {gpu_name} | Memory: {memory_used:.1f}GB/{memory_total:.1f}GB")
        
    def target_achieved(self, target_type, value, target):
        """Log target achievement"""
        self.logger.info(f"🎯 Target Achieved! {target_type}: {value:.2f}% (Target: {target:.2f}%)")
        
    def training_attempt(self, attempt, max_attempts):
        """Log training attempt"""
        self.logger.info(f"🔄 Training Attempt {attempt}/{max_attempts}")
        
    def save_session(self, filename):
        """Log session save"""
        self.logger.info(f"💾 Session saved: {filename}")

# Singleton instance
_logger_instance = None

def get_logger(config_path="config.yaml"):
    """Get singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingLogger(config_path)
    return _logger_instance

# Convenience functions
def log_info(message, emoji="ℹ️"):
    get_logger().info(message, emoji)

def log_success(message, emoji="✅"):
    get_logger().success(message, emoji)

def log_warning(message, emoji="⚠️"):
    get_logger().warning(message, emoji)

def log_error(message, emoji="❌"):
    get_logger().error(message, emoji)

def log_debug(message, emoji="🔍"):
    get_logger().debug(message, emoji)