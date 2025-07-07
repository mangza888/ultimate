#!/usr/bin/env python3
# data/data_manager.py - Fixed Data Manager with Dynamic Random Seeds
# แก้ไขให้แต่ละครั้งสร้างข้อมูลที่แตกต่างกัน

import os
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from utils.config_manager import get_config
from utils.logger import get_logger

class DataManager:
    """จัดการข้อมูลสำหรับการเทรด - with Dynamic Randomization"""
    
    def __init__(self):
        """เริ่มต้น DataManager"""
        self.config = get_config()
        self.logger = get_logger()
        self.data_cache = {}
        self.generation_count = 0
        
    def _get_dynamic_seed(self, symbol: str) -> int:
        """สร้าง dynamic seed ที่แตกต่างกันแต่ละครั้ง"""
        self.generation_count += 1
        
        # รวม: symbol hash + current time + generation count + random
        symbol_hash = hash(symbol) % 10000
        time_factor = int(time.time() * 1000) % 100000
        random_factor = random.randint(1, 10000)
        
        dynamic_seed = (symbol_hash + time_factor + self.generation_count * 1000 + random_factor) % 2**31
        
        self.logger.debug(f"Dynamic seed for {symbol}: {dynamic_seed} (generation: {self.generation_count})")
        return dynamic_seed
        
    def generate_synthetic_data(self, symbol: str, num_samples: int = 1000) -> pd.DataFrame:
        """สร้างข้อมูลจำลองที่แตกต่างกันแต่ละครั้ง"""
        try:
            # ใช้ dynamic seed แต่ละครั้ง
            dynamic_seed = self._get_dynamic_seed(symbol)
            np.random.seed(dynamic_seed)
            random.seed(dynamic_seed)
            
            # ราคาเริ่มต้นที่มีการสุ่ม
            base_prices = {'BTC': 50000, 'ETH': 3000, 'BNB': 300, 'LTC': 100}
            start_price = base_prices.get(symbol, 100)
            
            # เพิ่มการสุ่มราคาเริ่มต้น ±10%
            price_variation = np.random.uniform(0.9, 1.1)
            start_price = start_price * price_variation
            
            # สร้างข้อมูลราคาที่เหมือนจริง
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(days=num_samples),
                periods=num_samples,
                freq='1H'
            )
            
            # เพิ่มการสุ่มเวลาเริ่มต้น
            time_offset = timedelta(hours=np.random.randint(-24, 25))
            timestamps = timestamps + time_offset
            
            # สร้างราคาด้วยพารามิเตอร์ที่สุ่ม
            prices = self._generate_realistic_prices(start_price, num_samples, symbol, dynamic_seed)
            
            # สร้าง OHLCV ด้วยการสุ่ม
            data = self._generate_randomized_ohlcv(prices, timestamps, symbol, dynamic_seed)
            
            df = pd.DataFrame(data)
            
            # ปรับแต่งให้เหมือนจริงมากขึ้น
            df = self._post_process_synthetic_data(df, dynamic_seed)
            
            self.logger.debug(f"Generated synthetic data for {symbol}: {len(df)} rows (seed: {dynamic_seed})")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data for {symbol}: {e}")
            raise
    
    def _generate_realistic_prices(self, start_price: float, num_samples: int, 
                                 symbol: str, seed: int) -> np.ndarray:
        """สร้างราคาที่เหมือนจริงด้วยพารามิเตอร์ที่สุ่ม"""
        
        np.random.seed(seed)
        prices = [start_price]
        
        # พารามิเตอร์ที่สุ่มตาม symbol และ seed
        volatility_params = {
            'BTC': {'base_vol': np.random.uniform(0.025, 0.045), 'trend_strength': np.random.uniform(0.002, 0.004)},
            'ETH': {'base_vol': np.random.uniform(0.035, 0.055), 'trend_strength': np.random.uniform(0.003, 0.005)},
            'BNB': {'base_vol': np.random.uniform(0.030, 0.050), 'trend_strength': np.random.uniform(0.001, 0.003)},
            'LTC': {'base_vol': np.random.uniform(0.040, 0.060), 'trend_strength': np.random.uniform(0.001, 0.003)}
        }
        
        default_params = {
            'base_vol': np.random.uniform(0.020, 0.040), 
            'trend_strength': np.random.uniform(0.001, 0.003)
        }
        
        params = volatility_params.get(symbol, default_params)
        
        # สุ่มพารามิเตอร์เพิ่มเติม
        trend_frequency = np.random.uniform(0.003, 0.008)
        cycle_frequency = np.random.uniform(0.015, 0.025)
        mean_reversion_strength = np.random.uniform(0.03, 0.07)
        jump_probability = np.random.uniform(0.003, 0.007)
        
        for i in range(1, num_samples):
            # Multiple time series components with randomization
            
            # 1. Long-term trend (randomized)
            trend = np.sin(i * trend_frequency) * params['trend_strength']
            
            # 2. Medium-term cycles (randomized)
            cycle = (np.sin(i * cycle_frequency) * np.random.uniform(0.0008, 0.0012) + 
                    np.cos(i * cycle_frequency * 1.5) * np.random.uniform(0.0003, 0.0007))
            
            # 3. Mean reversion (randomized strength)
            deviation = (prices[-1] - start_price) / start_price
            mean_reversion = -mean_reversion_strength * deviation if abs(deviation) > 0.1 else 0
            
            # 4. Volatility clustering (randomized)
            if i > 1:
                prev_return = (prices[-1] - prices[-2]) / prices[-2]
                vol_clustering = np.random.uniform(0.2, 0.4) * abs(prev_return)
            else:
                vol_clustering = 0
            
            # 5. Random component (randomized volatility)
            base_volatility = params['base_vol'] * (1 + vol_clustering)
            random_component = np.random.normal(0, base_volatility)
            
            # 6. Jump component (randomized)
            if np.random.random() < jump_probability:
                jump_size = np.random.uniform(0.05, 0.15)
                jump = jump_size * (1 if np.random.random() > 0.5 else -1)
            else:
                jump = 0
            
            # Combine all components
            total_change = trend + cycle + mean_reversion + random_component + jump
            
            # Apply change
            new_price = prices[-1] * (1 + total_change)
            
            # Ensure price stays reasonable (randomized bounds)
            min_bound = start_price * np.random.uniform(0.05, 0.15)
            max_bound = start_price * np.random.uniform(5, 15)
            new_price = max(new_price, min_bound)
            new_price = min(new_price, max_bound)
            
            prices.append(new_price)
        
        return np.array(prices)
    
    def _generate_randomized_ohlcv(self, prices: np.ndarray, timestamps: pd.DatetimeIndex, 
                                 symbol: str, seed: int) -> Dict:
        """สร้าง OHLCV ที่สมจริงด้วยการสุ่ม"""
        
        np.random.seed(seed + 1000)  # Offset seed for OHLCV generation
        num_samples = len(prices)
        
        # สร้าง OHLC
        opens = np.zeros(num_samples)
        highs = np.zeros(num_samples)
        lows = np.zeros(num_samples)
        closes = prices.copy()
        
        # Volume parameters ตาม symbol (randomized)
        volume_params = {
            'BTC': {'base': np.random.randint(15000000, 25000000), 'multiplier': np.random.uniform(0.4, 0.6)},
            'ETH': {'base': np.random.randint(12000000, 18000000), 'multiplier': np.random.uniform(0.5, 0.7)},
            'BNB': {'base': np.random.randint(3000000, 7000000), 'multiplier': np.random.uniform(0.6, 0.8)},
            'LTC': {'base': np.random.randint(2000000, 4000000), 'multiplier': np.random.uniform(0.7, 0.9)}
        }
        
        vol_param = volume_params.get(symbol, {
            'base': np.random.randint(500000, 1500000), 
            'multiplier': np.random.uniform(0.8, 1.2)
        })
        
        for i in range(num_samples):
            if i == 0:
                opens[i] = closes[i]
            else:
                # Open มักจะใกล้เคียงกับ close ของวันก่อน (randomized gap)
                gap_range = np.random.uniform(0.001, 0.003)
                gap = np.random.normal(0, gap_range)
                opens[i] = closes[i-1] * (1 + gap)
            
            # สร้าง intraday range (randomized)
            volatility = abs(closes[i] - opens[i]) / opens[i] + np.random.uniform(0.003, 0.008)
            
            # High และ Low (randomized factors)
            high_factor = 1 + abs(np.random.normal(0, volatility * np.random.uniform(0.8, 1.2)))
            low_factor = 1 - abs(np.random.normal(0, volatility * np.random.uniform(0.8, 1.2)))
            
            highs[i] = max(opens[i], closes[i]) * high_factor
            lows[i] = min(opens[i], closes[i]) * low_factor
            
            # ตรวจสอบความสมเหตุสมผล
            highs[i] = max(highs[i], max(opens[i], closes[i]))
            lows[i] = min(lows[i], min(opens[i], closes[i]))
        
        # สร้าง Volume ที่สมเหตุสมผล (randomized)
        volumes = []
        for i in range(num_samples):
            # Volume สูงเมื่อมีการเคลื่อนไหวมาก
            price_change = abs(closes[i] - opens[i]) / opens[i] if opens[i] > 0 else 0
            volume_multiplier = 1 + (price_change * np.random.uniform(2, 4))
            
            # Add randomness
            random_factor = np.random.lognormal(0, vol_param['multiplier'])
            
            # Time-based volume patterns (randomized)
            hour = i % 24
            if hour in [9, 10, 15, 16]:  # High volume hours
                time_multiplier = np.random.uniform(1.2, 1.8)
            elif hour in [0, 1, 2, 3, 4, 5]:  # Low volume hours
                time_multiplier = np.random.uniform(0.3, 0.7)
            else:
                time_multiplier = np.random.uniform(0.8, 1.2)
            
            volume = vol_param['base'] * volume_multiplier * random_factor * time_multiplier
            volumes.append(max(volume, vol_param['base'] * 0.1))
        
        return {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'symbol': symbol
        }
    
    def _post_process_synthetic_data(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """ปรับแต่งข้อมูลสังเคราะห์หลังการสร้าง (randomized)"""
        
        try:
            np.random.seed(seed + 2000)  # Another offset for post-processing
            
            # ปัดราคาให้สมจริง (randomized precision)
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if df[col].iloc[0] > 1000:  # ราคาสูง
                    precision = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
                    df[col] = np.round(df[col], precision)
                elif df[col].iloc[0] > 100:  # ราคาปานกลาง
                    precision = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
                    df[col] = np.round(df[col], precision)
                else:  # ราคาต่ำ
                    precision = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
                    df[col] = np.round(df[col], precision)
            
            # ปัด Volume (randomized)
            volume_precision = np.random.choice([0, 1], p=[0.8, 0.2])
            df['volume'] = np.round(df['volume'], volume_precision).astype(int)
            
            # เพิ่ม noise เล็กน้อยให้สมจริง (randomized noise)
            for col in price_columns:
                noise_strength = np.random.uniform(0.00005, 0.00015)
                noise = np.random.normal(0, df[col] * noise_strength, len(df))
                df[col] += noise
                df[col] = np.maximum(df[col], 0.01)  # ป้องกันราคาติดลบ
            
            # เพิ่ม market microstructure noise (randomized)
            if np.random.random() > 0.7:  # 30% chance
                microstructure_noise = np.random.uniform(0.0001, 0.0005)
                for col in price_columns:
                    micro_noise = np.random.normal(0, microstructure_noise, len(df))
                    df[col] += df[col] * micro_noise
            
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
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            features_df['obv'] = obv
            
            # 8. Price action indicators
            features_df['price_range'] = (high - low) / (close + 1e-8)
            features_df['body_size'] = abs(close - features_df['open']) / (close + 1e-8)
            features_df['upper_shadow'] = (high - np.maximum(close, features_df['open'])) / (close + 1e-8)
            features_df['lower_shadow'] = (np.minimum(close, features_df['open']) - low) / (close + 1e-8)
            
            # 9. Support and Resistance levels
            features_df['resistance_20'] = high.rolling(window=20, min_periods=1).max()
            features_df['support_20'] = low.rolling(window=20, min_periods=1).min()
            features_df['price_position'] = (close - features_df['support_20']) / (features_df['resistance_20'] - features_df['support_20'] + 1e-8)
            
            # 10. Trend indicators
            features_df['trend_5'] = (features_df['sma_5'] > features_df['sma_5'].shift(1)).astype(int)
            features_df['trend_20'] = (features_df['sma_20'] > features_df['sma_20'].shift(1)).astype(int)
            features_df['ma_cross'] = (features_df['sma_5'] > features_df['sma_20']).astype(int)
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0)
            
            self.logger.debug(f"Created {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return df
    
    def load_historical_data(self, symbol: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
        """โหลดข้อมูลย้อนหลัง (existing method - no changes needed)"""
        # Keep existing implementation
        pass
    
    def get_combined_data(self, symbols: List[str], data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """ดึงข้อมูลรวมสำหรับหลาย symbols (existing method - no changes needed)"""
        # Keep existing implementation  
        pass