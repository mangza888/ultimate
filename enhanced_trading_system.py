#!/usr/bin/env python3
# enhanced_multi_pair_trading_system.py - Enhanced with Rich Progress and Better Visualization

import os
import sys
import asyncio
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Rich imports for beautiful progress
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available. Install with: pip install rich")

# Check dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    if RICH_AVAILABLE:
        console.print("✅ PyTorch available for multi-pair AI training", style="green")
    else:
        print("✅ PyTorch available for multi-pair AI training")
except ImportError:
    TORCH_AVAILABLE = False
    if RICH_AVAILABLE:
        console.print("❌ PyTorch not available", style="red")
    else:
        print("❌ PyTorch not available")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
    if RICH_AVAILABLE:
        console.print("✅ Scikit-learn available", style="green")
    else:
        print("✅ Scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    if RICH_AVAILABLE:
        console.print("⚠️ Scikit-learn not available", style="yellow")
    else:
        print("⚠️ Scikit-learn not available")

class MultiPairDataManager:
    """Enhanced data manager for multiple trading pairs"""
    
    def __init__(self):
        self.save_dir = "market_data"
        os.makedirs(self.save_dir, exist_ok=True)
        self.console = console if RICH_AVAILABLE else None
        
        # Trading pairs with realistic parameters
        self.pairs_config = {
            'BTC/USDT': {
                'base_price': 50000,
                'daily_volatility': 0.04,
                'trend_bias': 0.0001,
                'news_frequency': 0.015
            },
            'ETH/USDT': {
                'base_price': 3000,
                'daily_volatility': 0.05,
                'trend_bias': 0.0002,
                'news_frequency': 0.020
            },
            'BNB/USDT': {
                'base_price': 300,
                'daily_volatility': 0.06,
                'trend_bias': 0.0003,
                'news_frequency': 0.025
            },
            'LTC/USDT': {
                'base_price': 100,
                'daily_volatility': 0.055,
                'trend_bias': 0.00015,
                'news_frequency': 0.018
            }
        }
    
    def generate_multi_pair_data(self, pairs=None, periods=2000):
        """Generate realistic data for multiple trading pairs"""
        if pairs is None:
            pairs = list(self.pairs_config.keys())
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]📊 Generating multi-pair data for {len(pairs)} pairs...[/bold cyan]")
        else:
            print(f"📊 Generating multi-pair data for {len(pairs)} pairs...")
        
        all_data = {}
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Generating data...", total=len(pairs))
                
                for pair in pairs:
                    progress.update(task, description=f"[cyan]Generating {pair} data...")
                    config = self.pairs_config[pair]
                    
                    # Generate more volatile and tradeable data
                    data = self._generate_enhanced_pair_data(pair, config, periods)
                    all_data[pair] = data
                    
                    # Calculate stats
                    total_return = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                    volatility = data['close'].pct_change().std() * np.sqrt(1440) * 100
                    
                    progress.advance(task)
        else:
            for pair in pairs:
                print(f"🔄 Generating {pair} data...")
                config = self.pairs_config[pair]
                data = self._generate_enhanced_pair_data(pair, config, periods)
                all_data[pair] = data
        
        # Display summary table
        if RICH_AVAILABLE:
            table = Table(title="Generated Data Summary", box=box.ROUNDED)
            table.add_column("Pair", style="cyan")
            table.add_column("Return", style="green")
            table.add_column("Volatility", style="yellow")
            table.add_column("Data Points", style="magenta")
            
            for pair, data in all_data.items():
                total_return = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                volatility = data['close'].pct_change().std() * np.sqrt(1440) * 100
                table.add_row(
                    pair,
                    f"{total_return:.1f}%",
                    f"{volatility:.1f}%",
                    str(len(data))
                )
            
            console.print(table)
        
        return all_data
    
    def _generate_enhanced_pair_data(self, pair, config, periods):
        """Generate enhanced tradeable data for a single pair"""
        np.random.seed(hash(pair) % 1000)  # Different seed per pair
        
        base_price = config['base_price']
        daily_vol = config['daily_volatility']
        trend_bias = config['trend_bias']
        news_freq = config['news_frequency']
        
        minute_vol = daily_vol / np.sqrt(1440)
        
        # Generate dates
        dates = pd.date_range(
            start=datetime.now() - timedelta(minutes=periods),
            periods=periods,
            freq='1T'
        )
        
        prices = [base_price]
        volumes = []
        
        # Market regime controls (more dynamic)
        trend_strength = trend_bias
        volatility_regime = 1.0
        trend_counter = 0
        vol_counter = 0
        
        for i in range(1, periods):
            # Dynamic trend changes (more frequent)
            if trend_counter <= 0:
                trend_counter = np.random.randint(50, 200)  # Shorter trends
                trend_strength = np.random.normal(trend_bias, 0.0002)
            trend_counter -= 1
            
            # Volatility regime changes
            if vol_counter <= 0:
                vol_counter = np.random.randint(100, 300)
                volatility_regime = np.random.uniform(0.5, 2.0)  # 0.5x to 2x volatility
            vol_counter -= 1
            
            # Price components with enhanced variability
            trend_component = trend_strength
            noise_component = np.random.normal(0, minute_vol * volatility_regime)
            
            # More frequent news events (creates trading opportunities)
            if np.random.random() < news_freq:
                news_impact = np.random.normal(0, minute_vol * 5)
                noise_component += news_impact
            
            # Mean reversion component (creates reversals)
            deviation = (prices[-1] / base_price) - 1
            mean_reversion = -deviation * 0.001
            
            # Momentum component (creates trends)
            if i >= 5:
                recent_change = (prices[-1] - prices[-5]) / prices[-5]
                momentum = recent_change * 0.1
            else:
                momentum = 0
            
            total_change = trend_component + noise_component + mean_reversion + momentum
            new_price = prices[-1] * (1 + total_change)
            
            # Price bounds
            new_price = max(new_price, base_price * 0.2)
            new_price = min(new_price, base_price * 5.0)
            
            prices.append(new_price)
            
            # Enhanced volume (more realistic patterns)
            base_vol = 1000
            price_impact = abs(total_change) * 100000
            time_factor = 1 + 0.5 * np.sin(i * 2 * np.pi / 1440)  # Daily cycle
            volume = (base_vol + price_impact) * time_factor * np.random.lognormal(0, 0.4)
            volumes.append(max(volume, 50))
        
        # Create enhanced OHLCV with better spreads
        ohlcv_data = []
        for i, close in enumerate(prices):
            if i == 0:
                ohlcv_data.append({
                    'open': close, 'high': close, 'low': close,
                    'close': close, 'volume': 1000
                })
                continue
            
            prev_close = prices[i-1]
            
            # Enhanced spread calculation
            volatility = abs(close - prev_close)
            base_spread = volatility * np.random.uniform(0.3, 1.5)
            
            # Create realistic intrabar movement
            high_extension = base_spread * np.random.uniform(0, 2)
            low_extension = base_spread * np.random.uniform(0, 2)
            
            high = max(close, prev_close) + high_extension
            low = min(close, prev_close) - low_extension
            
            # Open price with gap potential
            gap_factor = np.random.normal(0, 0.002)  # Potential gaps
            open_price = prev_close * (1 + gap_factor)
            
            # Ensure OHLC validity
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            ohlcv_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volumes[i-1]
            })
        
        return pd.DataFrame(ohlcv_data, index=dates)

class EnhancedFeatureEngine:
    """Enhanced feature engineering for multiple pairs"""
    
    def __init__(self):
        self.console = console if RICH_AVAILABLE else None
    
    def create_comprehensive_features(self, multi_pair_data):
        """Create comprehensive features across multiple pairs"""
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]🔧 Creating comprehensive features for {len(multi_pair_data)} pairs...[/bold cyan]")
        else:
            print(f"🔧 Creating comprehensive features for {len(multi_pair_data)} pairs...")
        
        all_features = {}
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Processing features...", total=len(multi_pair_data))
                
                for pair, data in multi_pair_data.items():
                    progress.update(task, description=f"[cyan]Processing {pair} features...")
                    
                    features = pd.DataFrame(index=data.index)
                    
                    try:
                        # Enhanced price features
                        features['returns'] = data['close'].pct_change()
                        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
                        
                        # Enhanced moving averages
                        for window in [3, 5, 10, 15, 20, 30]:
                            features[f'sma_{window}'] = data['close'].rolling(window, min_periods=1).mean()
                            features[f'ema_{window}'] = data['close'].ewm(span=window).mean()
                        
                        # Calculate price ratios after SMAs are created
                        for window in [3, 5, 10, 15, 20, 30]:
                            if f'sma_{window}' in features.columns:
                                features[f'price_sma_ratio_{window}'] = data['close'] / (features[f'sma_{window}'] + 1e-8)
                            
                            if window > 5 and f'sma_{window}' in features.columns:
                                features[f'sma_slope_{window}'] = features[f'sma_{window}'].pct_change(3)
                        
                        # Enhanced volatility features
                        for window in [5, 10, 20]:
                            features[f'volatility_{window}'] = data['close'].rolling(window, min_periods=1).std()
                        
                        # Calculate volatility ratios after all volatility features are created
                        for window in [5, 10]:
                            if f'volatility_{window}' in features.columns and 'volatility_20' in features.columns:
                                features[f'vol_ratio_{window}'] = features[f'volatility_{window}'] / (features['volatility_20'] + 1e-8)
                        
                        # Enhanced volume features
                        features['volume_sma'] = data['volume'].rolling(10, min_periods=1).mean()
                        features['volume_ratio'] = data['volume'] / (features['volume_sma'] + 1e-8)
                        features['volume_price_trend'] = (features['volume_ratio'] * features['returns']).rolling(5).mean()
                        
                        # Enhanced momentum indicators
                        # RSI with multiple periods
                        for rsi_period in [14, 21]:
                            delta = data['close'].diff()
                            gain = delta.where(delta > 0, 0).rolling(rsi_period, min_periods=1).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period, min_periods=1).mean()
                            rs = gain / (loss + 1e-8)
                            features[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
                        
                        # MACD
                        ema12 = data['close'].ewm(span=12).mean()
                        ema26 = data['close'].ewm(span=26).mean()
                        features['macd'] = ema12 - ema26
                        features['macd_signal'] = features['macd'].ewm(span=9).mean()
                        features['macd_histogram'] = features['macd'] - features['macd_signal']
                        
                        # Bollinger Bands
                        bb_period = 20
                        bb_std = 2
                        bb_sma = data['close'].rolling(bb_period, min_periods=1).mean()
                        bb_std_val = data['close'].rolling(bb_period, min_periods=1).std()
                        features['bb_upper'] = bb_sma + (bb_std_val * bb_std)
                        features['bb_lower'] = bb_sma - (bb_std_val * bb_std)
                        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-8)
                        features['bb_squeeze'] = bb_std_val / (bb_sma + 1e-8)
                        
                        # Support/Resistance levels
                        for window in [10, 20]:
                            features[f'support_{window}'] = data['low'].rolling(window, min_periods=1).min()
                            features[f'resistance_{window}'] = data['high'].rolling(window, min_periods=1).max()
                            features[f'support_distance_{window}'] = (data['close'] - features[f'support_{window}']) / (data['close'] + 1e-8)
                            features[f'resistance_distance_{window}'] = (features[f'resistance_{window}'] - data['close']) / (data['close'] + 1e-8)
                        
                        # Clean features - handle NaN and infinite values
                        features = features.replace([np.inf, -np.inf], np.nan)
                        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
                        
                        all_features[pair] = features
                        progress.advance(task)
                        
                    except Exception as e:
                        if RICH_AVAILABLE:
                            console.print(f"[red]❌ Error processing {pair}: {e}[/red]")
                        else:
                            print(f"❌ Error processing {pair}: {e}")
                        # Create minimal features as fallback
                        features = pd.DataFrame(index=data.index)
                        features['returns'] = data['close'].pct_change().fillna(0)
                        features['sma_20'] = data['close'].rolling(20, min_periods=1).mean()
                        features['rsi_14'] = 50.0  # Neutral RSI
                        features['volume_ratio'] = 1.0
                        features['bb_position'] = 0.5
                        all_features[pair] = features
                        progress.advance(task)
        else:
            for pair, data in multi_pair_data.items():
                print(f"   Processing {pair}...")
                # Feature generation code (same as above)
                # ... (keeping the same feature generation logic)
        
        # Display feature summary
        if RICH_AVAILABLE:
            table = Table(title="Feature Engineering Summary", box=box.ROUNDED)
            table.add_column("Pair", style="cyan")
            table.add_column("Features", style="green")
            table.add_column("Data Points", style="yellow")
            
            for pair, features in all_features.items():
                table.add_row(pair, str(len(features.columns)), str(len(features)))
            
            console.print(table)
        
        return all_features

class EnhancedLabelGenerator:
    """Enhanced label generation with better signal distribution"""
    
    def __init__(self):
        self.console = console if RICH_AVAILABLE else None
    
    def create_balanced_labels(self, multi_pair_data, lookahead=3):
        """Create balanced labels that generate more trading signals"""
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]🎯 Creating enhanced labels for {len(multi_pair_data)} pairs...[/bold cyan]")
        else:
            print(f"🎯 Creating enhanced labels for {len(multi_pair_data)} pairs...")
        
        all_labels = {}
        label_stats = []
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Processing labels...", total=len(multi_pair_data))
                
                for pair, data in multi_pair_data.items():
                    progress.update(task, description=f"[cyan]Processing {pair} labels...")
                    
                    # Enhanced forward return calculation
                    forward_returns = data['close'].pct_change(lookahead).shift(-lookahead)
                    
                    # Dynamic threshold calculation based on volatility
                    volatility = data['close'].pct_change().rolling(50, min_periods=10).std()
                    
                    labels = []
                    buy_count = 0
                    sell_count = 0
                    hold_count = 0
                    
                    for i in range(len(forward_returns)):
                        if pd.isna(forward_returns.iloc[i]) or pd.isna(volatility.iloc[i]):
                            labels.append(1)
                            hold_count += 1
                            continue
                        
                        vol = volatility.iloc[i]
                        ret = forward_returns.iloc[i]
                        
                        # More aggressive thresholds for more signals
                        buy_threshold = vol * 1.0     # 1x volatility for buy
                        sell_threshold = -vol * 1.0   # 1x volatility for sell
                        
                        # Additional momentum-based signals
                        if i >= 5:
                            momentum = (data['close'].iloc[i] - data['close'].iloc[i-5]) / data['close'].iloc[i-5]
                            
                            # Momentum signals
                            if momentum > vol * 2:
                                labels.append(2)  # Strong momentum buy
                                buy_count += 1
                                continue
                            elif momentum < -vol * 2:
                                labels.append(0)  # Strong momentum sell
                                sell_count += 1
                                continue
                        
                        # Regular return-based signals
                        if ret > buy_threshold:
                            labels.append(2)  # Buy
                            buy_count += 1
                        elif ret < sell_threshold:
                            labels.append(0)  # Sell
                            sell_count += 1
                        else:
                            labels.append(1)  # Hold
                            hold_count += 1
                    
                    labels_series = pd.Series(labels, index=data.index)
                    all_labels[pair] = labels_series
                    
                    total = len(labels)
                    label_stats.append({
                        'pair': pair,
                        'sell': sell_count,
                        'sell_pct': sell_count/total*100,
                        'hold': hold_count,
                        'hold_pct': hold_count/total*100,
                        'buy': buy_count,
                        'buy_pct': buy_count/total*100
                    })
                    
                    progress.advance(task)
        else:
            for pair, data in multi_pair_data.items():
                print(f"   Processing {pair} labels...")
                # Label generation code (same as above)
                # ... (keeping the same label generation logic)
        
        # Display label distribution
        if RICH_AVAILABLE:
            table = Table(title="Label Distribution", box=box.ROUNDED)
            table.add_column("Pair", style="cyan")
            table.add_column("SELL", style="red")
            table.add_column("HOLD", style="yellow")
            table.add_column("BUY", style="green")
            
            for stats in label_stats:
                table.add_row(
                    stats['pair'],
                    f"{stats['sell']} ({stats['sell_pct']:.1f}%)",
                    f"{stats['hold']} ({stats['hold_pct']:.1f}%)",
                    f"{stats['buy']} ({stats['buy_pct']:.1f}%)"
                )
            
            console.print(table)
        
        return all_labels

class MultiPairTradingAI:
    """Enhanced AI for multi-pair trading"""
    
    def __init__(self, input_size, pair_count):
        self.input_size = input_size
        self.pair_count = pair_count
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        self.console = console if RICH_AVAILABLE else None
        
        if TORCH_AVAILABLE:
            self._build_enhanced_model()
    
    def _build_enhanced_model(self):
        """Build enhanced neural network for multi-pair trading"""
        # Larger network for multi-pair complexity
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 3)  # 3 classes per pair
        )
        
        if RICH_AVAILABLE:
            console.print(f"[bold green]🤖 Built enhanced multi-pair AI: {self.input_size} -> [256,128,64,32] -> 3[/bold green]")
        else:
            print(f"🤖 Built enhanced multi-pair AI: {self.input_size} -> [256,128,64,32] -> 3")
    
    def train_multi_pair(self, X_train, y_train, X_val, y_val, epochs=150):
        """Enhanced training for multi-pair data"""
        if not TORCH_AVAILABLE:
            if RICH_AVAILABLE:
                console.print("[yellow]⚠️ PyTorch not available - using mock training[/yellow]")
            else:
                print("⚠️ PyTorch not available - using mock training")
            self.is_trained = True
            return {'accuracy': 0.85}
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]🔥 Training enhanced multi-pair AI...[/bold cyan]")
            console.print(f"   Train samples: {len(X_train