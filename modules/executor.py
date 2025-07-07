from utils.config_manager import get_config
from utils.logger import get_logger

class Executor:
    """
    คำนวณ position size และตัดสินใจซื้อ–ขายในช่วง live/paper trading
    """
    def __init__(self):
        cfg = get_config()
        self.logger = get_logger()
        self.initial_capital = cfg.get('system.initial_capital')
        self.risk_frac = cfg.get('system.risk_fraction')

    def position_size(self, atr: float) -> float:
        risk_amount = self.initial_capital * self.risk_frac
        size = risk_amount / atr
        self.logger.info(f"Calculated position size: {size:.4f}")
        return size

    def decide(self, score: float, threshold: float = 0.5) -> int:
        action = 1 if score > threshold else -1
        self.logger.info(f"Decided action {action} for score {score:.3f}")
        return action
