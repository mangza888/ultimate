#!/usr/bin/env python3
# data/data_manager.py - Data Management Module (แก้ไข imports)
# จัดการข้อมูลสำหรับการเทรดและการเทรน

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union  # ✅ เพิ่ม Any และ Union
from utils.config_manager import get_config
from utils.logger import get_logger

class DataManager:
    """จัดการข้อมูลสำหรับการเทรด"""
    
    def __init__(self):
        """เริ่มต้น DataManager"""
        self.config = get_config()
        self.logger = get_logger()
        self.data_cache = {}
        
    def load_historical_data(self, symbol: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
        """โหลดข้อมูลย้อนหลัง (ปรับปรุงให้ใช้ข้อมูลจริง)"""
        try:
            # ลำดับไฟล์ที่จะลอง (เรียงตามลำดับความสำคัญ)
            possible_files = [
                f"{symbol}_yfinance_daily.csv",           # ไฟล์หลัก
                f"{symbol}_yfinance_daily_backup.csv",    # ไฟล์สำรอง
                f"{symbol}_daily_fixed.csv",
                f"{symbol}_real.csv",
                f"{symbol}.csv"
            ]
            
            for filename in possible_files:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    self.logger.info(f"Loading {symbol} from {filename}")
                    
                    try:
                        # โหลดไฟล์ CSV
                        df = pd.read_csv(filepath)
                        
                        # ตรวจสอบและแปลงข้อมูล
                        df = self._process_historical_data(df, symbol, filename)
                        
                        if len(df) > 50:  # ต้องมีข้อมูลอย่างน้อย 50 แถว
                            self.logger.success(f"Loaded {symbol}: {len(df)} rows from {filename}")
                            return df
                        else:
                            self.logger.warning(f"Insufficient data for {symbol} in {filename}: {len(df)} rows")
                            continue
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to load {filename}: {e}")
                        continue
            
            # ถ้าไม่มีข้อมูลจริง ให้สร้างข้อมูลจำลอง
            self.logger.warning(f"No historical data found for {symbol}, generating synthetic data")
            return self.generate_synthetic_data(symbol, 1000)
            
        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {e}")
            return self.generate_synthetic_data(symbol, 1000)
    
    def _process_historical_data(self, df: pd.DataFrame, symbol: str, filename: str) -> pd.DataFrame:
        """ประมวลผลข้อมูลย้อนหลังให้เป็นรูปแบบมาตรฐาน (ปรับปรุงให้รองรับหลายรูปแบบ)"""
        try:
            processed_df = df.copy()
            
            # แปลงชื่อคอลัมน์ให้เป็นตัวพิมพ์เล็ก
            processed_df.columns = [col.lower().strip() for col in processed_df.columns]
            
            # ตรวจสอบรูปแบบไฟล์
            if 'backup' in filename:
                # ไฟล์ backup มีรูปแบบต่าง (Price, Close, High, Low, Open, Volume)
                self.logger.info(f"Processing backup format for {symbol}")
                processed_df = self._process_backup_format(processed_df)
            else:
                # ไฟล์ปกติ (Date, Open, High, Low, Close, Volume)
                self.logger.info(f"Processing standard format for {symbol}")
                processed_df = self._process_standard_format(processed_df)
            
            # จัดการ timestamp
            processed_df = self._process_timestamp(processed_df)
            
            # ตรวจสอบและแปลงคอลัมน์ OHLCV
            processed_df = self._process_ohlcv_columns(processed_df)
            
            # เพิ่มคอลัมน์ symbol
            processed_df['symbol'] = symbol
            
            # ลบแถวที่มีค่า NaN
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            final_rows = len(processed_df)
            
            if initial_rows != final_rows:
                self.logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
            
            # เรียงลำดับตาม timestamp
            processed_df = processed_df.sort_values('timestamp').reset_index(drop=True)
            
            # ตรวจสอบความถูกต้องของข้อมูล OHLC
            processed_df = self._validate_ohlc_data(processed_df)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing historical data for {symbol}: {e}")
            raise
    
    def _process_backup_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """ประมวลผลไฟล์ backup format"""
        try:
            # ไฟล์ backup: Price, Close, High, Low, Open, Volume
            # แปลงชื่อคอลัมน์
            column_mapping = {
                'price': 'close',  # ใช้ Price เป็น Close ถ้าไม่มี Close
            }
            
            # เปลี่ยนชื่อคอลัมน์
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df[new_name] = df[old_name]
            
            # ตรวจสอบและแปลงข้อมูลเป็นตัวเลข
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    # แปลงจาก string เป็น numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing backup format: {e}")
            raise
    
    def _process_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """ประมวลผลไฟล์ standard format"""
        try:
            # ไฟล์ปกติ: Date, Open, High, Low, Close, Volume
            # ข้อมูลควรจะเป็นตัวเลขอยู่แล้ว แต่ตรวจสอบอีกครั้ง
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing standard format: {e}")
            raise
    
    def _process_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """ประมวลผล timestamp"""
        try:
            # ค้นหาคอลัมน์ timestamp/date
            timestamp_columns = ['date', 'timestamp', 'time', 'datetime']
            timestamp_column = None
            
            for col in timestamp_columns:
                if col in df.columns:
                    timestamp_column = col
                    break
            
            if timestamp_column:
                # แปลงเป็น datetime
                df['timestamp'] = pd.to_datetime(df[timestamp_column], errors='coerce')
                
                # ลบคอลัมน์เดิมถ้าไม่ใช่ 'timestamp'
                if timestamp_column != 'timestamp':
                    df = df.drop(columns=[timestamp_column])
            else:
                # สร้าง timestamp ถ้าไม่มี
                self.logger.warning("No timestamp column found, creating artificial timestamps")
                df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(days=len(df)), 
                    periods=len(df), 
                    freq='D'
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing timestamp: {e}")
            # สร้าง timestamp แบบ fallback
            df['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(days=len(df)), 
                periods=len(df), 
                freq='D'
            )
            return df
    
    def _process_ohlcv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ประมวลผลคอลัมน์ OHLCV"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in required_columns:
                if col not in df.columns:
                    if col in ['open', 'high', 'low'] and 'close' in df.columns:
                        # ใช้ close แทนถ้าไม่มีคอลัมน์อื่น
                        df[col] = df['close']
                        self.logger.warning(f"Missing {col}, using close price")
                    elif col == 'volume':
                        # ตั้งค่า volume เริ่มต้น
                        df[col] = 1000000
                        self.logger.warning(f"Missing {col}, using default value")
                    else:
                        raise ValueError(f"Missing required column: {col}")
                
                # ตรวจสอบและเติมค่าที่หายไป
                if df[col].isna().any():
                    # เติมด้วยค่าก่อนหน้า แล้วค่าถัดไป แล้วค่าเฉลี่ย
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
                    
                    if df[col].isna().any():
                        # ถ้ายังมี NaN อยู่ ให้ใช้ค่าเริ่มต้น
                        default_values = {
                            'open': 100, 'high': 100, 'low': 100, 'close': 100, 'volume': 1000000
                        }
                        df[col] = df[col].fillna(default_values[col])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing OHLCV columns: {e}")
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
                # แทนที่ค่าลบหรือศูนย์ด้วยค่าเฉลี่ย
                invalid_mask = (df[col] <= 0)
                if invalid_mask.any():
                    mean_price = df[col][df[col] > 0].mean()
                    df.loc[invalid_mask, col] = mean_price
                    self.logger.warning(f"Fixed {invalid_mask.sum()} invalid {col} values")
            
            # ตรวจสอบ volume
            df['volume'] = np.maximum(df['volume'], 1)  # ป้องกัน volume เป็นศูนย์
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating OHLC data: {e}")
            return df
    
    def get_combined_data(self, symbols: List[str], data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """ดึงข้อมูลรวมสำหรับหลาย symbols (ปรับปรุงให้ใช้ข้อมูลจริง)"""
        try:
            combined_data = {}
            
            for symbol in symbols:
                self.logger.info(f"Loading data for {symbol}...")
                
                # ลองโหลดข้อมูลจริงก่อน
                historical_data = self.load_historical_data(symbol, data_dir)
                
                if historical_data is not None and len(historical_data) > 50:
                    combined_data[symbol] = historical_data
                    self.logger.success(f"Using real data for {symbol}: {len(historical_data)} rows")
                else:
                    # ใช้ข้อมูลจำลองถ้าไม่มีข้อมูลจริงหรือข้อมูลไม่เพียงพอ
                    self.logger.warning(f"Using synthetic data for {symbol}")
                    synthetic_data = self.generate_synthetic_data(symbol, 1000)
                    combined_data[symbol] = synthetic_data
            
            # แสดงสรุป
            total_real_data = sum(1 for symbol, data in combined_data.items() 
                                if len(data) > 500)  # ถือว่าเป็นข้อมูลจริงถ้ามีมากกว่า 500 แถว
            
            self.logger.info(f"Data summary: {len(combined_data)} symbols loaded")
            self.logger.info(f"Real data: {total_real_data}, Synthetic data: {len(combined_data) - total_real_data}")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error getting combined data: {e}")
            return {}
    
    def generate_synthetic_data(self, symbol: str, num_samples: int = 1000) -> pd.DataFrame:
        """สร้างข้อมูลจำลองที่มีคุณภาพ (เพิ่มคุณภาพให้ดีขึ้น)"""
        try:
            # ตั้งค่า random seed ตาม symbol
            np.random.seed(hash(symbol) % 2**32)
            
            # ราคาเริ่มต้นที่สมจริง
            base_prices = {
                'BTC': 45000, 'ETH': 2800, 'BNB': 320, 'LTC': 95,
                'ADA': 0.5, 'DOT': 7, 'LINK': 15, 'UNI': 8
            }
            start_price = base_prices.get(symbol, 100)
            
            # สร้างข้อมูลราคาที่เหมือนจริงมากขึ้น
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(days=num_samples),
                periods=num_samples,
                freq='1H'
            )
            
            # สร้างราคาด้วย sophisticated model
            prices = self._generate_realistic_prices(start_price, num_samples, symbol)
            
            # สร้าง OHLCV ที่สมจริง
            data = self._generate_realistic_ohlcv(prices, timestamps, symbol)
            
            df = pd.DataFrame(data)
            
            # ปรับแต่งให้เหมือนจริงมากขึ้น
            df = self._post_process_synthetic_data(df)
            
            self.logger.debug(f"Generated synthetic data for {symbol}: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data for {symbol}: {e}")
            raise
    
    def _generate_realistic_prices(self, start_price: float, num_samples: int, symbol: str) -> np.ndarray:
        """สร้างราคาที่เหมือนจริงมากขึ้น"""
        prices = [start_price]
        
        # ตั้งค่าตาม symbol
        volatility_params = {
            'BTC': {'base_vol': 0.035, 'trend_strength': 0.003},
            'ETH': {'base_vol': 0.045, 'trend_strength': 0.004},
            'BNB': {'base_vol': 0.040, 'trend_strength': 0.002},
            'LTC': {'base_vol': 0.050, 'trend_strength': 0.002}
        }
        
        params = volatility_params.get(symbol, {'base_vol': 0.030, 'trend_strength': 0.002})
        
        for i in range(1, num_samples):
            # Multiple time series components
            
            # 1. Long-term trend
            trend = np.sin(i * 0.005) * params['trend_strength']
            
            # 2. Medium-term cycles
            cycle = np.sin(i * 0.02) * 0.001 + np.cos(i * 0.03) * 0.0005
            
            # 3. Mean reversion
            deviation = (prices[-1] - start_price) / start_price
            mean_reversion = -0.05 * deviation if abs(deviation) > 0.1 else 0
            
            # 4. Volatility clustering (GARCH-like)
            if i > 1:
                prev_return = (prices[-1] - prices[-2]) / prices[-2]
                vol_clustering = 0.3 * abs(prev_return)
            else:
                vol_clustering = 0
            
            # 5. Random component
            base_volatility = params['base_vol'] * (1 + vol_clustering)
            random_component = np.random.normal(0, base_volatility)
            
            # 6. Jump component (rare large moves)
            if np.random.random() < 0.005:  # 0.5% chance of jump
                jump = np.random.normal(0, 0.1) * (1 if np.random.random() > 0.5 else -1)
            else:
                jump = 0
            
            # Combine all components
            total_change = trend + cycle + mean_reversion + random_component + jump
            
            # Apply change
            new_price = prices[-1] * (1 + total_change)
            
            # Ensure price stays positive and reasonable
            new_price = max(new_price, start_price * 0.1)  # Don't go below 10% of start price
            new_price = min(new_price, start_price * 10)   # Don't go above 10x start price
            
            prices.append(new_price)
        
        return np.array(prices)
    
    def _generate_realistic_ohlcv(self, prices: np.ndarray, timestamps: pd.DatetimeIndex, symbol: str) -> Dict[str, Any]:
        """สร้าง OHLCV ที่สมจริง"""
        num_samples = len(prices)
        
        # สร้าง OHLC
        opens = np.zeros(num_samples)
        highs = np.zeros(num_samples)
        lows = np.zeros(num_samples)
        closes = prices.copy()
        
        # Volume parameters ตาม symbol
        volume_params = {
            'BTC': {'base': 20000000, 'multiplier': 0.5},
            'ETH': {'base': 15000000, 'multiplier': 0.6},
            'BNB': {'base': 5000000, 'multiplier': 0.7},
            'LTC': {'base': 3000000, 'multiplier': 0.8}
        }
        
        vol_param = volume_params.get(symbol, {'base': 1000000, 'multiplier': 1.0})
        
        for i in range(num_samples):
            if i == 0:
                opens[i] = closes[i]
            else:
                # Open มักจะใกล้เคียงกับ close ของวันก่อน
                gap = np.random.normal(0, 0.002)  # Gap เล็กน้อย
                opens[i] = closes[i-1] * (1 + gap)
            
            # สร้าง intraday range
            volatility = abs(closes[i] - opens[i]) / opens[i] + 0.005  # ความผันผวนขั้นต่ำ
            
            # High และ Low
            high_factor = 1 + abs(np.random.normal(0, volatility))
            low_factor = 1 - abs(np.random.normal(0, volatility))
            
            highs[i] = max(opens[i], closes[i]) * high_factor
            lows[i] = min(opens[i], closes[i]) * low_factor
            
            # ตรวจสอบความสมเหตุสมผล
            highs[i] = max(highs[i], max(opens[i], closes[i]))
            lows[i] = min(lows[i], min(opens[i], closes[i]))
        
        # สร้าง Volume ที่สมเหตุสมผล
        volumes = []
        for i in range(num_samples):
            # Volume สูงเมื่อมีการเคลื่อนไหวมาก
            price_change = abs(closes[i] - opens[i]) / opens[i] if opens[i] > 0 else 0
            volume_multiplier = 1 + (price_change * 3)  # Volume เพิ่มตามการเคลื่อนไหว
            
            # Add some randomness
            random_factor = np.random.lognormal(0, vol_param['multiplier'])
            
            volume = vol_param['base'] * volume_multiplier * random_factor
            volumes.append(max(volume, vol_param['base'] * 0.1))  # Volume ขั้นต่ำ
        
        return {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'symbol': symbol
        }
    
    def _post_process_synthetic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ปรับแต่งข้อมูลสังเคราะห์หลังการสร้าง"""
        try:
            # ปัดราคาให้สมจริง
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if df[col].iloc[0] > 1000:  # ราคาสูง เช่น BTC
                    df[col] = np.round(df[col], 2)
                elif df[col].iloc[0] > 100:  # ราคาปานกลาง
                    df[col] = np.round(df[col], 3)
                else:  # ราคาต่ำ
                    df[col] = np.round(df[col], 4)
            
            # ปัด Volume
            df['volume'] = np.round(df['volume']).astype(int)
            
            # เพิ่ม noise เล็กน้อยให้สมจริง
            for col in price_columns:
                noise = np.random.normal(0, df[col] * 0.0001, len(df))
                df[col] += noise
                df[col] = np.maximum(df[col], 0.01)  # ป้องกันราคาติดลบ
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error post-processing synthetic data: {e}")
            return df
    
    def create_features_from_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """สร้าง technical indicators จากข้อมูล (ปรับปรุงให้ครบถ้วน)"""
        try:
            features_df = df.copy()
            
            # ดึงการตั้งค่า indicators
            indicators_config = self.config.get_indicators_config()
            
            # Basic price data
            close = features_df['close']
            high = features_df['high']
            low = features_df['low']
            volume = features_df['volume']
            
            # 1. Moving Averages
            for period in indicators_config['moving_averages']:
                sma_col = f'sma_{period}'
                ema_col = f'ema_{period}'
                
                # Simple Moving Average
                features_df[sma_col] = close.rolling(window=period, min_periods=1).mean()
                
                # Exponential Moving Average
                features_df[ema_col] = close.ewm(span=period).mean()
                
                # Price to MA ratios
                features_df[f'price_to_{sma_col}'] = close / features_df[sma_col]
                features_df[f'price_to_{ema_col}'] = close / features_df[ema_col]
            
            # 2. RSI
            rsi_period = indicators_config['rsi_period']
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # 3. MACD
            fast_period = indicators_config['macd_fast']
            slow_period = indicators_config['macd_slow']
            signal_period = indicators_config['macd_signal']
            
            ema_fast = close.ewm(span=fast_period).mean()
            ema_slow = close.ewm(span=slow_period).mean()
            features_df['macd'] = ema_fast - ema_slow
            features_df['macd_signal'] = features_df['macd'].ewm(span=signal_period).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # 4. Bollinger Bands
            bb_period = indicators_config['bollinger_period']
            bb_std = indicators_config['bollinger_std']
            
            bb_sma = close.rolling(window=bb_period, min_periods=1).mean()
            bb_std_dev = close.rolling(window=bb_period, min_periods=1).std()
            features_df['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
            features_df['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
            features_df['bb_position'] = (close - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'] + 1e-8)
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / bb_sma
            
            # 5. Volatility measures
            returns = close.pct_change().fillna(0)
            for window in indicators_config['volatility_windows']:
                features_df[f'volatility_{window}'] = returns.rolling(window=window, min_periods=1).std()
                features_df[f'volatility_{window}_annualized'] = features_df[f'volatility_{window}'] * np.sqrt(252)
            
            # 6. Momentum indicators
            for period in indicators_config['momentum_periods']:
                features_df[f'momentum_{period}'] = close.pct_change(periods=period).fillna(0)
                features_df[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)).fillna(0)
            
            # 7. Volume indicators
            volume_sma_20 = volume.rolling(window=20, min_periods=1).mean()
            features_df['volume_ratio'] = volume / (volume_sma_20 + 1e-8)
            features_df['volume_sma_ratio'] = volume_sma_20 / volume.rolling(window=50, min_periods=1).mean()
            
            # OBV (On-Balance Volume)
            obv = [0]
            for i in range(1, len(features_df)):
                if features_df['close'].iloc[i] > features_df['close'].iloc[i-1]:
                    obv.append(obv[-1] + features_df['volume'].iloc[i])
                elif features_df['close'].iloc[i] < features_df['close'].iloc[i-1]:
                    obv.append(obv[-1] - features_df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            features_df['obv'] = obv
            
            # 8. Price patterns
            features_df['hl_ratio'] = (high - low) / close
            features_df['oc_ratio'] = (close - features_df['open']) / features_df['open']
            features_df['upper_shadow'] = (high - np.maximum(features_df['open'], close)) / (high - low + 1e-8)
            features_df['lower_shadow'] = (np.minimum(features_df['open'], close) - low) / (high - low + 1e-8)
            
            # 9. Stochastic Oscillator
            stoch_period = 14
            lowest_low = low.rolling(window=stoch_period, min_periods=1).min()
            highest_high = high.rolling(window=stoch_period, min_periods=1).max()
            features_df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
            features_df['stoch_d'] = features_df['stoch_k'].rolling(window=3, min_periods=1).mean()
            
            # 10. Average True Range (ATR)
            atr_period = 14
            high_low = high - low
            high_close_prev = np.abs(high - close.shift(1))
            low_close_prev = np.abs(low - close.shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            features_df['atr'] = true_range.rolling(window=atr_period, min_periods=1).mean()
            features_df['atr_percent'] = features_df['atr'] / close
            
            # 11. Williams %R
            williams_period = 14
            highest_high_w = high.rolling(window=williams_period, min_periods=1).max()
            lowest_low_w = low.rolling(window=williams_period, min_periods=1).min()
            features_df['williams_r'] = -100 * (highest_high_w - close) / (highest_high_w - lowest_low_w + 1e-8)
            
            # 12. Commodity Channel Index (CCI)
            cci_period = 20
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=cci_period, min_periods=1).mean()
            mad = typical_price.rolling(window=cci_period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
            features_df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-8)
            
            # 13. Time-based features
            features_df['hour'] = features_df['timestamp'].dt.hour if 'timestamp' in features_df.columns else 0
            features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek if 'timestamp' in features_df.columns else 0
            features_df['month'] = features_df['timestamp'].dt.month if 'timestamp' in features_df.columns else 1
            
            # 14. Lagged features
            lag_periods = [1, 2, 3, 5, 10]
            for lag in lag_periods:
                features_df[f'close_lag_{lag}'] = close.shift(lag)
                features_df[f'return_lag_{lag}'] = returns.shift(lag)
                features_df[f'volume_lag_{lag}'] = volume.shift(lag)
            
            # 15. Rolling statistics
            rolling_windows = [5, 10, 20, 50]
            for window in rolling_windows:
                features_df[f'close_mean_{window}'] = close.rolling(window=window, min_periods=1).mean()
                features_df[f'close_std_{window}'] = close.rolling(window=window, min_periods=1).std()
                features_df[f'volume_mean_{window}'] = volume.rolling(window=window, min_periods=1).mean()
                features_df[f'high_max_{window}'] = high.rolling(window=window, min_periods=1).max()
                features_df[f'low_min_{window}'] = low.rolling(window=window, min_periods=1).min()
            
            # Fill any remaining NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            self.logger.debug(f"Created {len(features_df.columns)} features from price data")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return df
    
    def get_training_data(self, symbols: List[str], lookback_days: int = 30, 
                         prediction_horizon: int = 1, data_dir: str = "data") -> Dict[str, Any]:
        """เตรียมข้อมูลสำหรับการเทรนโมเดล"""
        try:
            # โหลดข้อมูลรวม
            combined_data = self.get_combined_data(symbols, data_dir)
            
            if not combined_data:
                self.logger.error("No data available for training")
                return {}
            
            training_data = {}
            
            for symbol, data in combined_data.items():
                self.logger.info(f"Preparing training data for {symbol}")
                
                # สร้าง features
                features_data = self.create_features_from_data(data)
                
                # สร้าง sequences สำหรับ LSTM
                sequences_data = self._create_sequences(
                    features_data, lookback_days, prediction_horizon
                )
                
                if sequences_data:
                    training_data[symbol] = sequences_data
                    self.logger.success(f"Created training data for {symbol}: {sequences_data['X'].shape}")
                else:
                    self.logger.warning(f"Failed to create sequences for {symbol}")
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return {}
    
    def _create_sequences(self, features_df: pd.DataFrame, lookback_days: int, 
                         prediction_horizon: int) -> Optional[Dict[str, Any]]:
        """สร้าง sequences สำหรับ LSTM"""
        try:
            # เลือกคอลัมน์ที่เป็น features (ไม่รวม timestamp และ symbol)
            feature_columns = [col for col in features_df.columns 
                             if col not in ['timestamp', 'symbol']]
            
            # แปลงข้อมูลเป็น numpy array
            feature_data = features_df[feature_columns].values
            
            # Normalize ข้อมูล
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(feature_data)
            
            # สร้าง sequences
            X, y = [], []
            
            for i in range(lookback_days, len(normalized_data) - prediction_horizon + 1):
                # Input sequence
                X.append(normalized_data[i-lookback_days:i])
                
                # Target (ราคาปิดในอนาคต)
                close_idx = feature_columns.index('close')
                future_price = normalized_data[i + prediction_horizon - 1, close_idx]
                current_price = normalized_data[i - 1, close_idx]
                
                # คำนวณ return
                price_return = (future_price - current_price) / (abs(current_price) + 1e-8)
                
                # แปลงเป็น classification target
                if price_return > 0.02:  # ขึ้นมากกว่า 2%
                    target = 2  # Strong Buy
                elif price_return > 0.005:  # ขึ้นมากกว่า 0.5%
                    target = 1  # Buy
                elif price_return < -0.02:  # ลงมากกว่า 2%
                    target = 0  # Strong Sell
                elif price_return < -0.005:  # ลงมากกว่า 0.5%
                    target = 0  # Sell
                else:
                    target = 1  # Hold/Neutral
                
                y.append(target)
            
            if len(X) == 0:
                self.logger.warning("No sequences created - insufficient data")
                return None
            
            X = np.array(X)
            y = np.array(y)
            
            # แบ่งข้อมูล train/validation
            split_idx = int(len(X) * 0.8)
            
            return {
                'X_train': X[:split_idx],
                'y_train': y[:split_idx],
                'X_val': X[split_idx:],
                'y_val': y[split_idx:],
                'X': X,
                'y': y,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'lookback_days': lookback_days,
                'prediction_horizon': prediction_horizon
            }
            
        except Exception as e:
            self.logger.error(f"Error creating sequences: {e}")
            return None
    
    def get_latest_features(self, symbol: str, data_dir: str = "data") -> Optional[np.ndarray]:
        """ดึง features ล่าสุดสำหรับการทำนาย"""
        try:
            # โหลดข้อมูลล่าสุด
            historical_data = self.load_historical_data(symbol, data_dir)
            
            if historical_data is None or len(historical_data) < 50:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # สร้าง features
            features_data = self.create_features_from_data(historical_data)
            
            # เลือกคอลัมน์ที่เป็น features
            feature_columns = [col for col in features_data.columns 
                             if col not in ['timestamp', 'symbol']]
            
            # ดึงข้อมูลล่าสุด
            latest_features = features_data[feature_columns].iloc[-30:].values  # 30 วันล่าสุด
            
            return latest_features
            
        except Exception as e:
            self.logger.error(f"Error getting latest features for {symbol}: {e}")
            return None
    
    def get_data_summary(self, symbols: List[str], data_dir: str = "data") -> Dict[str, Any]:
        """สรุปข้อมูลที่มีอยู่"""
        try:
            summary = {
                'total_symbols': len(symbols),
                'available_data': {},
                'data_quality': {},
                'date_ranges': {},
                'missing_symbols': []
            }
            
            for symbol in symbols:
                try:
                    # ลองโหลดข้อมูล
                    data = self.load_historical_data(symbol, data_dir)
                    
                    if data is not None and len(data) > 0:
                        summary['available_data'][symbol] = {
                            'rows': len(data),
                            'start_date': str(data['timestamp'].min()) if 'timestamp' in data.columns else 'Unknown',
                            'end_date': str(data['timestamp'].max()) if 'timestamp' in data.columns else 'Unknown',
                            'columns': list(data.columns)
                        }
                        
                        # ตรวจสอบคุณภาพข้อมูล
                        quality_score = self._assess_data_quality(data)
                        summary['data_quality'][symbol] = quality_score
                        
                        # ช่วงวันที่
                        if 'timestamp' in data.columns:
                            date_range = (data['timestamp'].max() - data['timestamp'].min()).days
                            summary['date_ranges'][symbol] = date_range
                    else:
                        summary['missing_symbols'].append(symbol)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing {symbol}: {e}")
                    summary['missing_symbols'].append(symbol)
            
            # สถิติรวม
            summary['statistics'] = {
                'symbols_with_data': len(summary['available_data']),
                'symbols_missing': len(summary['missing_symbols']),
                'average_rows': np.mean([info['rows'] for info in summary['available_data'].values()]) if summary['available_data'] else 0,
                'average_quality': np.mean(list(summary['data_quality'].values())) if summary['data_quality'] else 0
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating data summary: {e}")
            return {}
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """ประเมินคุณภาพข้อมูล (0-100)"""
        try:
            quality_score = 100.0
            
            # ตรวจสอบ missing values
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            if missing_cells > 0:
                quality_score -= (missing_cells / total_cells) * 30
            
            # ตรวจสอบข้อมูลราคา
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    # ตรวจสอบค่าลบ
                    negative_values = (df[col] <= 0).sum()
                    if negative_values > 0:
                        quality_score -= (negative_values / len(df)) * 20
                    
                    # ตรวจสอบค่าผิดปกติ (outliers)
                    q75, q25 = np.percentile(df[col], [75, 25])
                    iqr = q75 - q25
                    outliers = ((df[col] < (q25 - 1.5 * iqr)) | (df[col] > (q75 + 1.5 * iqr))).sum()
                    if outliers > len(df) * 0.05:  # มากกว่า 5%
                        quality_score -= 10
            
            # ตรวจสอบความสมเหตุสมผลของ OHLC
            if all(col in df.columns for col in price_columns):
                invalid_ohlc = ((df['high'] < df['low']) | 
                               (df['high'] < df['open']) | 
                               (df['high'] < df['close']) |
                               (df['low'] > df['open']) |
                               (df['low'] > df['close'])).sum()
                if invalid_ohlc > 0:
                    quality_score -= (invalid_ohlc / len(df)) * 25
            
            # ตรวจสอบ timestamp
            if 'timestamp' in df.columns:
                # ตรวจสอบลำดับเวลา
                if not df['timestamp'].is_monotonic_increasing:
                    quality_score -= 15
                
                # ตรวจสอบช่องว่างในเวลา
                time_gaps = df['timestamp'].diff().dt.total_seconds()
                if time_gaps.std() > time_gaps.mean():  # ความผันแปรสูง
                    quality_score -= 10
            
            return max(0, min(100, quality_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return 50.0  # คะแนนกลางถ้าไม่สามารถประเมินได้
    
    def cleanup_cache(self):
        """ล้าง cache ข้อมูล"""
        try:
            self.data_cache.clear()
            self.logger.info("Data cache cleared")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def export_processed_data(self, symbol: str, data_dir: str = "data", 
                             output_dir: str = "processed_data") -> bool:
        """ส่งออกข้อมูลที่ประมวลผลแล้ว"""
        try:
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs(output_dir, exist_ok=True)
            
            # โหลดและประมวลผลข้อมูล
            data = self.load_historical_data(symbol, data_dir)
            if data is None:
                return False
            
            features_data = self.create_features_from_data(data)
            
            # ส่งออกไฟล์
            output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
            features_data.to_csv(output_file, index=False)
            
            self.logger.success(f"Exported processed data for {symbol} to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data for {symbol}: {e}")
            return False