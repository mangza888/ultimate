#!/usr/bin/env python3
# data/data_manager.py - Data Management Module
# จัดการข้อมูลสำหรับการเทรดและการเทรน

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from utils.config_manager import get_config
from utils.logger import get_logger

class DataManager:
    """จัดการข้อมูลสำหรับการเทรด"""
    
    def __init__(self):
        """เริ่มต้น DataManager"""
        self.config = get_config()
        self.logger = get_logger()
        self.data_cache = {}
        
    def generate_synthetic_data(self, symbol: str, num_samples: int = 1000) -> pd.DataFrame:
        """สร้างข้อมูลจำลองที่มีคุณภาพ"""
        try:
            # ตั้งค่า random seed ตาม symbol
            np.random.seed(hash(symbol) % 2**32)
            
            # ราคาเริ่มต้น
            base_prices = {'BTC': 50000, 'ETH': 3000, 'BNB': 300, 'LTC': 100}
            start_price = base_prices.get(symbol, 100)
            
            # สร้างข้อมูลราคาที่เหมือนจริง
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(days=num_samples),
                periods=num_samples,
                freq='1H'
            )
            
            # สร้างราคา
            prices = self._generate_realistic_prices(start_price, num_samples)
            
            # สร้าง OHLCV
            data = {
                'timestamp': timestamps,
                'open': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, num_samples))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, num_samples))),
                'close': prices,
                'volume': np.random.lognormal(15, 1, num_samples),
                'symbol': symbol
            }
            
            df = pd.DataFrame(data)
            
            # ปรับแต่งให้เหมือนจริงมากขึ้น
            df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
            df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
            
            self.logger.debug(f"Generated synthetic data for {symbol}: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data for {symbol}: {e}")
            raise
    
    def _generate_realistic_prices(self, start_price: float, num_samples: int) -> np.ndarray:
        """สร้างราคาที่เหมือนจริง"""
        prices = [start_price]
        
        for i in range(1, num_samples):
            # สร้าง trend component
            trend = np.sin(i * 0.01) * 0.002  # Long-term trend
            
            # สร้าง mean reversion component
            mean_reversion = -0.1 * (prices[-1] - start_price) / start_price
            
            # สร้าง random walk component
            random_walk = np.random.normal(0, 0.02)
            
            # สร้าง volatility clustering
            volatility = 0.015 * (1 + 0.5 * abs(np.random.normal(0, 1)))
            
            # รวมทุก component
            price_change = trend + mean_reversion * 0.1 + random_walk * volatility
            
            # คำนวณราคาใหม่
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, start_price * 0.1))  # ป้องกันราคาติดลบ
        
        return np.array(prices)
    
    def load_historical_data(self, symbol: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
        """โหลดข้อมูลย้อนหลัง"""
        try:
            # ลองหาไฟล์ข้อมูล (ใช้ข้อมูลจริงของคุณ)
            possible_files = [
                f"{symbol}_yfinance_daily.csv",
                f"{symbol}_daily_fixed.csv",
                f"{symbol}_real.csv",
                f"{symbol}.csv"
            ]
            
            for filename in possible_files:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    self.logger.info(f"Loading {symbol} from {filename}")
                    
                    # โหลดไฟล์ CSV
                    df = pd.read_csv(filepath)
                    
                    # แปลงข้อมูลให้เป็นรูปแบบที่ถูกต้อง
                    df = self._process_historical_data(df, symbol)
                    
                    if len(df) > 100:  # ต้องมีข้อมูลอย่างน้อย 100 แถว
                        self.logger.success(f"Loaded {symbol}: {len(df)} rows")
                        return df
                    else:
                        self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
            
            self.logger.warning(f"No historical data found for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {e}")
            return None
    
    def _process_historical_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """ประมวลผลข้อมูลย้อนหลังให้เป็นรูปแบบมาตรฐาน"""
        try:
            # ทำสำเนาข้อมูล
            processed_df = df.copy()
            
            # ปรับชื่อคอลัมน์ให้เป็นตัวพิมพ์เล็ก
            processed_df.columns = [col.lower().strip() for col in processed_df.columns]
            
            # จัดการคอลัมน์ timestamp/date
            if 'date' in processed_df.columns:
                processed_df['timestamp'] = pd.to_datetime(processed_df['date'])
            elif 'timestamp' in processed_df.columns:
                processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
            else:
                # สร้าง timestamp ถ้าไม่มี
                processed_df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(days=len(processed_df)), 
                    periods=len(processed_df), 
                    freq='D'
                )
            
            # ตรวจสอบและแปลงคอลัมน์ OHLCV
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in required_columns:
                if col in processed_df.columns:
                    # แปลงเป็นตัวเลข
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    
                    # เติมค่าที่หายไป
                    processed_df[col] = processed_df[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # ถ้าไม่มีคอลัมน์ ให้ใช้ close แทน
                    if col in ['open', 'high', 'low'] and 'close' in processed_df.columns:
                        processed_df[col] = processed_df['close']
                    elif col == 'volume':
                        processed_df[col] = 1000000  # ค่า default
            
            # เพิ่มคอลัมน์ symbol
            processed_df['symbol'] = symbol
            
            # ลบแถวที่มีค่า NaN
            processed_df = processed_df.dropna(subset=required_columns)
            
            # เรียงลำดับตาม timestamp
            processed_df = processed_df.sort_values('timestamp').reset_index(drop=True)
            
            # ตรวจสอบความถูกต้องของข้อมูล OHLC
            processed_df = self._validate_ohlc_data(processed_df)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing historical data for {symbol}: {e}")
            raise
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบและแก้ไขข้อมูล OHLC"""
        try:
            # ตรวจสอบว่า High >= max(Open, Close) และ Low <= min(Open, Close)
            df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
            df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
            
            # ตรวจสอบว่าราคาเป็นบวก
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df[col] = np.maximum(df[col], 0.01)  # ป้องกันราคาเป็นศูนย์หรือลบ
            
            # ตรวจสอบ volume
            df['volume'] = np.maximum(df['volume'], 1)  # ป้องกัน volume เป็นศูนย์
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating OHLC data: {e}")
            return df
    
    def get_combined_data(self, symbols: List[str], data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """ดึงข้อมูลรวมสำหรับหลาย symbols"""
        try:
            combined_data = {}
            
            for symbol in symbols:
                # ลองโหลดข้อมูลจริงก่อน
                historical_data = self.load_historical_data(symbol, data_dir)
                
                if historical_data is not None and len(historical_data) > 100:
                    combined_data[symbol] = historical_data
                    self.logger.info(f"Using historical data for {symbol}")
                else:
                    # ใช้ข้อมูลจำลองถ้าไม่มีข้อมูลจริง
                    synthetic_data = self.generate_synthetic_data(symbol, 1000)
                    combined_data[symbol] = synthetic_data
                    self.logger.info(f"Using synthetic data for {symbol}")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error getting combined data: {e}")
            return {}
    
    def create_features_from_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """สร้าง technical indicators จากข้อมูล"""
        try:
            features_df = df.copy()
            
            # ดึงการตั้งค่า indicators
            indicators_config = self.config.get_indicators_config()
            
            # Moving Averages
            for period in indicators_config['moving_averages']:
                sma_col = f'sma_{period}'
                features_df[sma_col] = features_df['close'].rolling(window=period, min_periods=1).mean()
                features_df[f'price_to_{sma_col}'] = features_df['close'] / features_df[sma_col]
            
            # RSI
            rsi_period = indicators_config['rsi_period']
            delta = features_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            fast_period = indicators_config['macd_fast']
            slow_period = indicators_config['macd_slow']
            signal_period = indicators_config['macd_signal']
            
            ema_fast = features_df['close'].ewm(span=fast_period).mean()
            ema_slow = features_df['close'].ewm(span=slow_period).mean()
            features_df['macd'] = ema_fast - ema_slow
            features_df['macd_signal'] = features_df['macd'].ewm(span=signal_period).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Bollinger Bands
            bb_period = indicators_config['bollinger_period']
            bb_std = indicators_config['bollinger_std']
            
            bb_sma = features_df['close'].rolling(window=bb_period, min_periods=1).mean()
            bb_std_dev = features_df['close'].rolling(window=bb_period, min_periods=1).std()
            features_df['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
            features_df['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
            features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            
            # Volatility
            returns = features_df['close'].pct_change().fillna(0)
            for window in indicators_config['volatility_windows']:
                features_df[f'volatility_{window}'] = returns.rolling(window=window, min_periods=1).std()
            
            # Momentum
            for period in indicators_config['momentum_periods']:
                features_df[f'momentum_{period}'] = features_df['close'].pct_change(periods=period).fillna(0)
            
            # Volume indicators
            volume_sma = features_df['volume'].rolling(window=20, min_periods=1).mean()
            features_df['volume_ratio'] = features_df['volume'] / (volume_sma + 1e-8)
            
            # Price range
            features_df['price_range'] = (features_df['high'] - features_df['low']) / (features_df['close'] + 1e-8)
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return df