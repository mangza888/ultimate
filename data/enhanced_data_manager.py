#!/usr/bin/env python3
# data/enhanced_data_manager.py - Enhanced Multi-Timeframe Data Manager
# Multi-timeframe, Multi-symbol data with External Features

import os
import pandas as pd
import numpy as np
import time
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yfinance as yf
import ccxt
import ta
import pandas_ta as pta
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from utils.config_manager import get_config
from utils.logger import get_logger

@dataclass
class MarketData:
    """Market data container"""
    symbol: str
    timeframe: str
    ohlcv: pd.DataFrame
    features: pd.DataFrame
    metadata: Dict[str, Any]

@dataclass
class ExternalData:
    """External data container"""
    source: str
    data_type: str
    data: pd.DataFrame
    last_updated: datetime

class OnChainDataProvider:
    """On-chain metrics provider"""
    
    def __init__(self):
        self.logger = get_logger()
        self.base_urls = {
            'glassnode': 'https://api.glassnode.com/v1/metrics',
            'messari': 'https://data.messari.io/api/v1',
            'coinmetrics': 'https://api.coinmetrics.io/v4'
        }
    
    async def get_onchain_metrics(self, symbol: str = 'BTC') -> Optional[pd.DataFrame]:
        """Get on-chain metrics for symbol"""
        try:
            # Mock implementation - replace with real API calls
            metrics = {
                'active_addresses': np.random.randint(800000, 1200000),
                'transaction_count': np.random.randint(250000, 350000),
                'nvt_ratio': np.random.uniform(30, 80),
                'mvrv_ratio': np.random.uniform(0.8, 2.5),
                'hodl_waves_1y+': np.random.uniform(60, 80),
                'exchange_inflow': np.random.uniform(0.5, 2.0),
                'exchange_outflow': np.random.uniform(0.5, 2.0),
                'funding_rate': np.random.uniform(-0.1, 0.1),
                'long_short_ratio': np.random.uniform(0.8, 1.5)
            }
            
            df = pd.DataFrame([metrics], index=[datetime.now()])
            self.logger.debug(f"Retrieved on-chain metrics for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting on-chain metrics: {e}")
            return None

class SentimentDataProvider:
    """News and social sentiment provider"""
    
    def __init__(self):
        self.logger = get_logger()
        self.apis = {
            'cryptopanic': 'https://cryptopanic.com/api/v1',
            'newsapi': 'https://newsapi.org/v2',
            'twitter': 'https://api.twitter.com/2'
        }
    
    async def get_sentiment_data(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get sentiment data for symbols"""
        try:
            # Mock implementation - replace with real sentiment analysis
            sentiment_data = {
                'news_sentiment': np.random.uniform(-1, 1),  # -1 bearish, +1 bullish
                'social_sentiment': np.random.uniform(-1, 1),
                'fear_greed_index': np.random.uniform(0, 100),
                'crypto_news_count': np.random.randint(50, 200),
                'positive_news_ratio': np.random.uniform(0.3, 0.7),
                'social_volume': np.random.uniform(1000, 5000),
                'influencer_sentiment': np.random.uniform(-1, 1)
            }
            
            df = pd.DataFrame([sentiment_data], index=[datetime.now()])
            self.logger.debug("Retrieved sentiment data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment data: {e}")
            return None

class MacroDataProvider:
    """Macro economic data provider"""
    
    def __init__(self):
        self.logger = get_logger()
    
    async def get_macro_data(self) -> Optional[pd.DataFrame]:
        """Get macro economic indicators"""
        try:
            # Get real macro data using yfinance
            tickers = ['^VIX', '^GSPC', 'GC=F', 'DX-Y.NYB', '^TNX']
            
            macro_data = {}
            for ticker in tickers:
                try:
                    data = yf.download(ticker, period='1d', interval='1h', progress=False)
                    if not data.empty:
                        latest_price = data['Close'].iloc[-1]
                        daily_change = (latest_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                        
                        if ticker == '^VIX':
                            macro_data['vix'] = latest_price
                            macro_data['vix_change'] = daily_change
                        elif ticker == '^GSPC':
                            macro_data['sp500'] = latest_price
                            macro_data['sp500_change'] = daily_change
                        elif ticker == 'GC=F':
                            macro_data['gold'] = latest_price
                            macro_data['gold_change'] = daily_change
                        elif ticker == 'DX-Y.NYB':
                            macro_data['dxy'] = latest_price
                            macro_data['dxy_change'] = daily_change
                        elif ticker == '^TNX':
                            macro_data['us10y'] = latest_price
                            macro_data['us10y_change'] = daily_change
                except:
                    continue
            
            # Add additional macro indicators
            macro_data.update({
                'risk_on_score': np.random.uniform(-2, 2),  # Composite risk sentiment
                'crypto_dominance': np.random.uniform(40, 60),
                'btc_dominance': np.random.uniform(45, 55),
                'total_market_cap': np.random.uniform(1.5e12, 2.5e12)
            })
            
            df = pd.DataFrame([macro_data], index=[datetime.now()])
            self.logger.debug("Retrieved macro economic data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting macro data: {e}")
            return None

class OrderBookAnalyzer:
    """Order book depth analysis"""
    
    def __init__(self):
        self.logger = get_logger()
        self.exchanges = ['binance', 'coinbase', 'kraken']
    
    async def get_orderbook_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Analyze order book for features"""
        try:
            # Mock order book analysis - replace with real exchange APIs
            features = {
                'bid_ask_spread': np.random.uniform(0.01, 0.1),
                'depth_imbalance': np.random.uniform(-0.5, 0.5),  # negative = more sells
                'top_level_ratio': np.random.uniform(0.1, 0.9),
                'weighted_mid_price': np.random.uniform(0.99, 1.01),
                'orderbook_pressure': np.random.uniform(-1, 1),
                'large_order_ratio': np.random.uniform(0.1, 0.4),
                'price_clustering': np.random.uniform(0.2, 0.8)
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error analyzing order book: {e}")
            return None

class EnhancedDataManager:
    """Enhanced Data Manager with Multi-timeframe and External Data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.logger = get_logger()
        
        # Data providers
        self.onchain_provider = OnChainDataProvider()
        self.sentiment_provider = SentimentDataProvider()
        self.macro_provider = MacroDataProvider()
        self.orderbook_analyzer = OrderBookAnalyzer()
        
        # Scalers for different data types
        self.scalers = {
            'price': RobustScaler(),
            'volume': RobustScaler(),
            'technical': StandardScaler(),
            'onchain': StandardScaler(),
            'sentiment': StandardScaler(),
            'macro': StandardScaler()
        }
        
        # Data cache
        self.data_cache = {}
        self.external_cache = {}
        
        # Timeframes to use
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.primary_timeframe = '1m'
        
        # Supported symbols
        self.crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'LTC/USDT', 'ADA/USDT', 'DOT/USDT']
        
    async def get_comprehensive_dataset(self, 
                                      symbols: List[str] = None,
                                      timeframes: List[str] = None,
                                      periods: int = 2000) -> Dict[str, MarketData]:
        """Get comprehensive multi-timeframe dataset with external features"""
        
        if symbols is None:
            symbols = self.crypto_symbols
        if timeframes is None:
            timeframes = self.timeframes
            
        self.logger.info(f"Building comprehensive dataset: {len(symbols)} symbols x {len(timeframes)} timeframes")
        
        comprehensive_data = {}
        
        # Get market data for each symbol and timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Get OHLCV data
                    ohlcv_data = await self._get_market_data(symbol, timeframe, periods)
                    
                    if ohlcv_data is not None and len(ohlcv_data) > 100:
                        # Create technical features
                        technical_features = self._create_technical_features(ohlcv_data)
                        
                        # Add cross-timeframe features
                        cross_tf_features = await self._add_cross_timeframe_features(
                            symbol, timeframe, timeframes, periods
                        )
                        
                        # Add external features
                        external_features = await self._add_external_features(symbol)
                        
                        # Combine all features
                        combined_features = self._combine_features(
                            technical_features, cross_tf_features, external_features
                        )
                        
                        # Create MarketData object
                        key = f"{symbol}_{timeframe}"
                        comprehensive_data[key] = MarketData(
                            symbol=symbol,
                            timeframe=timeframe,
                            ohlcv=ohlcv_data,
                            features=combined_features,
                            metadata={
                                'last_updated': datetime.now(),
                                'data_quality': self._assess_data_quality(combined_features),
                                'feature_count': len(combined_features.columns),
                                'sample_count': len(combined_features)
                            }
                        )
                        
                        self.logger.debug(f"Processed {key}: {len(combined_features)} samples, {len(combined_features.columns)} features")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    continue
        
        self.logger.success(f"Comprehensive dataset ready: {len(comprehensive_data)} datasets")
        return comprehensive_data
    
    async def _get_market_data(self, symbol: str, timeframe: str, periods: int) -> Optional[pd.DataFrame]:
        """Get market data from exchange"""
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{periods}"
            if cache_key in self.data_cache:
                cache_data, cache_time = self.data_cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=5):  # 5-minute cache
                    return cache_data
            
            # Generate realistic synthetic data (replace with real exchange API)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=periods * self._timeframe_to_minutes(timeframe))
            
            # Create datetime index
            freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D'}
            freq = freq_map.get(timeframe, '1T')
            
            dates = pd.date_range(start=start_time, end=end_time, freq=freq)[:periods]
            
            # Generate realistic price data
            base_price = self._get_base_price(symbol)
            prices = self._generate_realistic_prices(base_price, len(dates), timeframe)
            
            # Create OHLCV
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                if i == 0:
                    open_price = price
                else:
                    open_price = data[i-1]['close']
                
                # Generate OHLC with realistic intraday movement
                volatility = 0.002 * self._timeframe_to_minutes(timeframe) / 60  # Higher vol for longer TF
                high = price * (1 + abs(np.random.normal(0, volatility)))
                low = price * (1 - abs(np.random.normal(0, volatility)))
                close = price
                
                # Ensure OHLC consistency
                high = max(high, max(open_price, close))
                low = min(low, min(open_price, close))
                
                # Generate volume
                base_volume = self._get_base_volume(symbol, timeframe)
                volume_multiplier = 1 + abs(np.random.normal(0, 0.3))
                volume = base_volume * volume_multiplier
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self.data_cache[cache_key] = (df, datetime.now())
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol} {timeframe}: {e}")
            return None
    
    def _create_technical_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators using pandas-ta"""
        
        try:
            df = ohlcv.copy()
            
            # Add pandas-ta indicators
            df.ta.strategy("all")  # This adds all available indicators
            
            # Custom indicators
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['price_acceleration'] = df['price_change'].diff()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Volume-based features
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_volume'] = df['close'] * df['volume']
            df['vwap'] = (df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum())
            
            # Volatility features
            for window in [5, 10, 20, 50]:
                df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
                df[f'high_low_volatility_{window}'] = (df['high'] - df['low']).rolling(window).std()
            
            # Support/Resistance levels
            for window in [20, 50]:
                df[f'resistance_{window}'] = df['high'].rolling(window).max()
                df[f'support_{window}'] = df['low'].rolling(window).min()
                df[f'distance_to_resistance_{window}'] = (df[f'resistance_{window}'] - df['close']) / df['close']
                df[f'distance_to_support_{window}'] = (df['close'] - df[f'support_{window}']) / df['close']
            
            # Momentum features
            for period in [1, 3, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'high_momentum_{period}'] = df['high'] / df['high'].shift(period) - 1
                df[f'low_momentum_{period}'] = df['low'] / df['low'].shift(period) - 1
            
            # Statistical features
            for window in [10, 20, 50]:
                returns = df['price_change'].rolling(window)
                df[f'skewness_{window}'] = returns.skew()
                df[f'kurtosis_{window}'] = returns.kurt()
                df[f'sharpe_{window}'] = returns.mean() / returns.std()
            
            # Fibonacci retracement levels
            df['fib_23.6'] = df['low'] + 0.236 * (df['high'] - df['low'])
            df['fib_38.2'] = df['low'] + 0.382 * (df['high'] - df['low'])
            df['fib_50.0'] = df['low'] + 0.500 * (df['high'] - df['low'])
            df['fib_61.8'] = df['low'] + 0.618 * (df['high'] - df['low'])
            
            # Market microstructure
            df['bid_ask_spread_proxy'] = (df['high'] - df['low']) / df['close']
            df['tick_direction'] = np.where(df['close'] > df['close'].shift(1), 1, 
                                   np.where(df['close'] < df['close'].shift(1), -1, 0))
            df['tick_momentum'] = df['tick_direction'].rolling(10).sum()
            
            # Remove non-numeric columns and handle NaN
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            self.logger.debug(f"Created {len(df.columns)} technical features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating technical features: {e}")
            return ohlcv
    
    async def _add_cross_timeframe_features(self, symbol: str, current_tf: str, 
                                          all_timeframes: List[str], periods: int) -> pd.DataFrame:
        """Add features from different timeframes"""
        
        try:
            cross_tf_features = pd.DataFrame()
            
            for tf in all_timeframes:
                if tf == current_tf:
                    continue
                
                # Get data from different timeframe
                tf_data = await self._get_market_data(symbol, tf, periods // self._timeframe_ratio(tf, current_tf))
                
                if tf_data is not None and len(tf_data) > 20:
                    # Resample to current timeframe
                    resampled = self._resample_to_timeframe(tf_data, current_tf)
                    
                    # Create features with timeframe prefix
                    tf_features = pd.DataFrame(index=resampled.index)
                    
                    # Price trend from higher timeframe
                    tf_features[f'{tf}_trend'] = (resampled['close'] > resampled['close'].shift(1)).astype(int)
                    tf_features[f'{tf}_sma_20'] = resampled['close'].rolling(20).mean()
                    tf_features[f'{tf}_rsi'] = ta.momentum.RSIIndicator(resampled['close']).rsi()
                    tf_features[f'{tf}_volatility'] = resampled['close'].pct_change().rolling(20).std()
                    
                    # Alignment features
                    if len(cross_tf_features) == 0:
                        cross_tf_features = tf_features
                    else:
                        cross_tf_features = cross_tf_features.join(tf_features, how='outer')
            
            if len(cross_tf_features) > 0:
                cross_tf_features = cross_tf_features.fillna(method='ffill').fillna(0)
                self.logger.debug(f"Added {len(cross_tf_features.columns)} cross-timeframe features")
            
            return cross_tf_features
            
        except Exception as e:
            self.logger.error(f"Error adding cross-timeframe features: {e}")
            return pd.DataFrame()
    
    async def _add_external_features(self, symbol: str) -> pd.DataFrame:
        """Add external features (on-chain, sentiment, macro)"""
        
        try:
            external_features = pd.DataFrame()
            
            # Get on-chain data
            onchain_data = await self.onchain_provider.get_onchain_metrics(symbol.split('/')[0])
            if onchain_data is not None:
                external_features = external_features.join(onchain_data, how='outer')
            
            # Get sentiment data
            sentiment_data = await self.sentiment_provider.get_sentiment_data([symbol])
            if sentiment_data is not None:
                external_features = external_features.join(sentiment_data, how='outer')
            
            # Get macro data
            macro_data = await self.macro_provider.get_macro_data()
            if macro_data is not None:
                external_features = external_features.join(macro_data, how='outer')
            
            # Forward fill external data (updates less frequently)
            if len(external_features) > 0:
                external_features = external_features.fillna(method='ffill').fillna(0)
                self.logger.debug(f"Added {len(external_features.columns)} external features")
            
            return external_features
            
        except Exception as e:
            self.logger.error(f"Error adding external features: {e}")
            return pd.DataFrame()
    
    def _combine_features(self, technical: pd.DataFrame, cross_tf: pd.DataFrame, 
                         external: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature types"""
        
        try:
            # Start with technical features
            combined = technical.copy()
            
            # Add cross-timeframe features
            if len(cross_tf) > 0:
                combined = combined.join(cross_tf, how='left')
            
            # Add external features (broadcast to all timestamps)
            if len(external) > 0:
                for col in external.columns:
                    combined[col] = external[col].iloc[-1] if len(external) > 0 else 0
            
            # Fill any remaining NaN values
            combined = combined.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove infinite values
            combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            self.logger.debug(f"Combined features: {len(combined.columns)} total features")
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining features: {e}")
            return technical
    
    def _assess_data_quality(self, features: pd.DataFrame) -> Dict[str, float]:
        """Assess data quality metrics"""
        
        try:
            quality_metrics = {
                'completeness': 1.0 - (features.isnull().sum().sum() / (len(features) * len(features.columns))),
                'consistency': 1.0 - (features.isin([np.inf, -np.inf]).sum().sum() / (len(features) * len(features.columns))),
                'uniqueness': features.nunique().mean() / len(features),
                'validity': 1.0 - (features.abs() > 1e6).sum().sum() / (len(features) * len(features.columns))
            }
            
            # Overall quality score
            quality_metrics['overall_score'] = np.mean(list(quality_metrics.values()))
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return {'overall_score': 0.5}
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        return mapping.get(timeframe, 1)
    
    def _timeframe_ratio(self, tf1: str, tf2: str) -> int:
        """Get ratio between two timeframes"""
        return self._timeframe_to_minutes(tf1) // self._timeframe_to_minutes(tf2)
    
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol"""
        price_map = {
            'BTC/USDT': 45000, 'ETH/USDT': 2800, 'BNB/USDT': 320,
            'LTC/USDT': 95, 'ADA/USDT': 0.45, 'DOT/USDT': 7.5
        }
        return price_map.get(symbol, 100)
    
    def _get_base_volume(self, symbol: str, timeframe: str) -> float:
        """Get base volume for symbol and timeframe"""
        base_volumes = {
            'BTC/USDT': 1000000, 'ETH/USDT': 800000, 'BNB/USDT': 500000,
            'LTC/USDT': 300000, 'ADA/USDT': 2000000, 'DOT/USDT': 400000
        }
        base = base_volumes.get(symbol, 100000)
        multiplier = self._timeframe_to_minutes(timeframe) / 60  # Scale by timeframe
        return base * multiplier
    
    def _generate_realistic_prices(self, base_price: float, length: int, timeframe: str) -> np.ndarray:
        """Generate realistic price movements"""
        
        # Adjust volatility based on timeframe
        base_vol = 0.02  # 2% daily volatility
        tf_minutes = self._timeframe_to_minutes(timeframe)
        volatility = base_vol * np.sqrt(tf_minutes / 1440)  # Scale to timeframe
        
        # Generate price path with drift and mean reversion
        prices = [base_price]
        drift = 0.0001  # Slight upward drift
        mean_reversion = 0.05  # Mean reversion strength
        
        for i in range(length - 1):
            # Mean reversion component
            price_deviation = (prices[-1] - base_price) / base_price
            reversion = -mean_reversion * price_deviation
            
            # Random component
            random_shock = np.random.normal(0, volatility)
            
            # Price change
            price_change = drift + reversion + random_shock
            new_price = prices[-1] * (1 + price_change)
            
            # Keep price within reasonable bounds
            new_price = max(new_price, base_price * 0.5)
            new_price = min(new_price, base_price * 2.0)
            
            prices.append(new_price)
        
        return np.array(prices)
    
    def _resample_to_timeframe(self, data: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample data to target timeframe"""
        
        try:
            freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D'}
            target_freq = freq_map.get(target_tf, '1T')
            
            resampled = data.resample(target_freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return resampled.dropna()
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return data

    def create_labels(self, data: pd.DataFrame, method: str = 'triple_barrier') -> pd.Series:
        """Create labels using various methods"""
        
        try:
            if method == 'triple_barrier':
                return self._triple_barrier_labels(data)
            elif method == 'fixed_horizon':
                return self._fixed_horizon_labels(data)
            elif method == 'volatility_adjusted':
                return self._volatility_adjusted_labels(data)
            else:
                return self._simple_labels(data)
                
        except Exception as e:
            self.logger.error(f"Error creating labels: {e}")
            return pd.Series(1, index=data.index)  # Default to hold
    
    def _triple_barrier_labels(self, data: pd.DataFrame, 
                              profit_threshold: float = 0.02,
                              loss_threshold: float = 0.01,
                              time_horizon: int = 60) -> pd.Series:
        """Triple barrier labeling method"""
        
        labels = []
        prices = data['close'].values
        
        for i in range(len(prices) - time_horizon):
            current_price = prices[i]
            future_prices = prices[i+1:i+time_horizon+1]
            
            # Calculate returns
            returns = (future_prices - current_price) / current_price
            
            # Check barriers
            profit_hit = np.any(returns >= profit_threshold)
            loss_hit = np.any(returns <= -loss_threshold)
            
            if profit_hit and not loss_hit:
                labels.append(2)  # Buy
            elif loss_hit and not profit_hit:
                labels.append(0)  # Sell
            elif profit_hit and loss_hit:
                # Both hit - check which comes first
                profit_idx = np.argmax(returns >= profit_threshold)
                loss_idx = np.argmax(returns <= -loss_threshold)
                labels.append(2 if profit_idx < loss_idx else 0)
            else:
                labels.append(1)  # Hold
        
        # Pad with holds for remaining periods
        labels.extend([1] * time_horizon)
        
        return pd.Series(labels, index=data.index)
    
    def _fixed_horizon_labels(self, data: pd.DataFrame, horizon: int = 20) -> pd.Series:
        """Fixed horizon return-based labels"""
        
        returns = data['close'].pct_change(periods=horizon).shift(-horizon)
        
        # Dynamic thresholds based on volatility
        vol = data['close'].pct_change().rolling(60).std()
        upper_threshold = vol * 1.5
        lower_threshold = -vol * 1.5
        
        labels = pd.Series(1, index=data.index)  # Default hold
        labels[returns > upper_threshold] = 2  # Buy
        labels[returns < lower_threshold] = 0  # Sell
        
        return labels.fillna(1)
    
    def _volatility_adjusted_labels(self, data: pd.DataFrame) -> pd.Series:
        """Volatility-adjusted labels using ATR"""
        
        # Calculate ATR
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Forward returns
        future_returns = close.shift(-10) / close - 1
        
        # Dynamic thresholds based on ATR
        atr_multiplier = 2.0
        upper_threshold = atr_multiplier * atr / close
        lower_threshold = -atr_multiplier * atr / close
        
        labels = pd.Series(1, index=data.index)
        labels[future_returns > upper_threshold] = 2
        labels[future_returns < lower_threshold] = 0
        
        return labels.fillna(1)
    
    def _simple_labels(self, data: pd.DataFrame) -> pd.Series:
        """Simple return-based labels"""
        
        returns = data['close'].pct_change(5).shift(-5)
        
        labels = pd.Series(1, index=data.index)
        labels[returns > 0.015] = 2  # Buy if >1.5% return
        labels[returns < -0.015] = 0  # Sell if <-1.5% return
        
        return labels.fillna(1)

# Usage example and testing
async def main():
    """Test the enhanced data manager"""
    
    logger = get_logger()
    logger.info("Testing Enhanced Data Manager")
    
    # Initialize data manager
    data_manager = EnhancedDataManager()
    
    # Get comprehensive dataset
    symbols = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT']
    timeframes = ['1m', '5m', '1h']
    
    dataset = await data_manager.get_comprehensive_dataset(
        symbols=symbols,
        timeframes=timeframes,
        periods=1000
    )
    
    # Display results
    for key, market_data in dataset.items():
        logger.info(f"Dataset: {key}")
        logger.info(f"  Features: {len(market_data.features.columns)}")
        logger.info(f"  Samples: {len(market_data.features)}")
        logger.info(f"  Data Quality: {market_data.metadata['data_quality']['overall_score']:.3f}")
        
        # Create labels
        labels = data_manager.create_labels(market_data.ohlcv, method='triple_barrier')
        logger.info(f"  Label distribution: {labels.value_counts().to_dict()}")

if __name__ == "__main__":
    asyncio.run(main())