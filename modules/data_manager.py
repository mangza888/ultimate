from utils.config_manager import get_config
from utils.logger import get_logger
import pandas as pd
import requests

class DataManager:
    """
    จัดการการโหลดข้อมูล OHLCV และคำนวณ Indicator
    """
    def __init__(self):
        cfg = get_config()
        self.symbols = cfg.get('symbols')
        self.timeframe = cfg.get('paper_trade.timeframe')
        self.history_days = cfg.get('backtest.duration_days')
        self.logger = get_logger()
        self.base_url = cfg.get('api.market_data_url', '')

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        self.logger.info(f"Fetching OHLCV for {symbol}")
        params = {
            'symbol': symbol,
            'interval': self.timeframe,
            'limit': int(self.history_days * 24 * 60 / int(self.timeframe.rstrip('m')))
        }
        resp = requests.get(self.base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame(data)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Computing technical indicators")
        ind = get_config().get('indicators')
        df['EMA20'] = df['close'].ewm(span=ind['moving_averages'][2]).mean()
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(ind['rsi_period']).mean()
        loss = -delta.clip(upper=0).rolling(ind['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df.dropna()

    def load_all(self) -> dict[str, pd.DataFrame]:
        return {s: self.compute_indicators(self.fetch_ohlcv(s)) for s in self.symbols}
