from utils.config_manager import get_config
from utils.logger import get_logger
from backtesting.advanced_backtesting import AdvancedBacktesting

class Backtester:
    """
    รัน backtest โดยใช้ AdvancedBacktesting และ log ผลลัพธ์
    """
    def __init__(self):
        cfg = get_config()
        self.logger = get_logger()
        self.cost = cfg.get('backtest.transaction_cost')
        self.slippage = cfg.get('backtest.slippage')

    def run(self, df, signals: list[float]) -> dict:
        self.logger.info("Starting backtest run")
        bt = AdvancedBacktesting(df, signals, cost=self.cost, slippage=self.slippage)
        result = bt.run()
        self.logger.info(f"Backtest complete: Return {result['return_pct']:+.2f}% over {len(signals)} signals")
        return result
