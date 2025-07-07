#!/usr/bin/env python3
# features/advanced_feature_engineering.py - Advanced Feature Engineering System
# Statistical, ML-derived, and Alternative Data Features

import pandas as pd
import numpy as np
import ta
import pandas_ta as pta
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

class StatisticalFeatureEngine:
    """Advanced statistical feature engineering"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def create_statistical_features(self, data: pd.DataFrame, 
                                  windows: List[int] = [10, 20, 50, 100]) -> pd.DataFrame:
        """Create advanced statistical features"""
        
        features = pd.DataFrame(index=data.index)
        price = data['close']
        returns = price.pct_change()
        
        for window in windows:
            # Rolling statistics
            rolling_returns = returns.rolling(window)
            rolling_price = price.rolling(window)
            
            # Moments
            features[f'mean_{window}'] = rolling_returns.mean()
            features[f'std_{window}'] = rolling_returns.std()
            features[f'skew_{window}'] = rolling_returns.skew()
            features[f'kurt_{window}'] = rolling_returns.kurt()
            
            # Risk metrics
            features[f'sharpe_{window}'] = features[f'mean_{window}'] / features[f'std_{window}']
            features[f'sortino_{window}'] = features[f'mean_{window}'] / rolling_returns[rolling_returns < 0].std()
            features[f'var_95_{window}'] = rolling_returns.quantile(0.05)
            features[f'cvar_95_{window}'] = rolling_returns[rolling_returns <= features[f'var_95_{window}']].mean()
            
            # Price-based statistics
            features[f'price_zscore_{window}'] = (price - rolling_price.mean()) / rolling_price.std()
            features[f'price_percentile_{window}'] = rolling_price.rank(pct=True)
            
            # Autocorrelation
            features[f'autocorr_1_{window}'] = rolling_returns.apply(lambda x: x.autocorr(lag=1))
            features[f'autocorr_5_{window}'] = rolling_returns.apply(lambda x: x.autocorr(lag=5))
            
            # Hurst exponent (simplified)
            features[f'hurst_{window}'] = rolling_returns.apply(self._calculate_hurst, raw=False)
            
            # Maximum drawdown
            rolling_max = rolling_price.max()
            drawdown = (price - rolling_max) / rolling_max
            features[f'max_drawdown_{window}'] = rolling_returns.apply(lambda x: x.min())
            
        return features.fillna(0)
    
    def _calculate_hurst(self, ts: pd.Series) -> float:
        """Calculate Hurst exponent"""
        try:
            if len(ts) < 10:
                return 0.5
            
            lags = range(2, min(20, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            if len(tau) < 2:
                return 0.5
                
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5

class MarketMicrostructureEngine:
    """Market microstructure feature engineering"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def create_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        
        features = pd.DataFrame(index=data.index)
        
        high, low, close, volume = data['high'], data['low'], data['close'], data['volume']
        
        # Price impact measures
        features['price_impact'] = abs(close.pct_change()) / (volume.pct_change() + 1e-8)
        features['amihud_illiquidity'] = abs(close.pct_change()) / (volume * close + 1e-8)
        
        # Bid-ask spread proxies
        features['high_low_spread'] = (high - low) / close
        features['close_mid_deviation'] = abs(close - (high + low) / 2) / close
        
        # Volume patterns
        features['volume_acceleration'] = volume.pct_change().diff()
        features['volume_momentum'] = volume.pct_change().rolling(5).sum()
        
        # Tick direction and pressure
        features['tick_direction'] = np.sign(close.diff())
        features['tick_momentum'] = features['tick_direction'].rolling(10).sum()
        features['buying_pressure'] = (close - low) / (high - low + 1e-8)
        features['selling_pressure'] = (high - close) / (high - low + 1e-8)
        
        # Order flow imbalance (proxied)
        features['order_flow_imbalance'] = features['buying_pressure'] - features['selling_pressure']
        
        # Price clustering (round number effects)
        features['price_clustering'] = self._detect_price_clustering(close)
        
        # Microstructure noise
        features['noise_ratio'] = self._calculate_noise_ratio(close)
        
        return features.fillna(0)
    
    def _detect_price_clustering(self, prices: pd.Series) -> pd.Series:
        """Detect price clustering around round numbers"""
        
        # Check if price ends in round numbers
        price_str = prices.astype(str)
        round_endings = ['0', '5', '00', '50']
        
        clustering_score = pd.Series(0.0, index=prices.index)
        
        for ending in round_endings:
            mask = price_str.str.endswith(ending)
            clustering_score[mask] += len(ending)
        
        return clustering_score
    
    def _calculate_noise_ratio(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate microstructure noise ratio"""
        
        # Realized variance
        returns = prices.pct_change()
        realized_var = returns.rolling(window).var()
        
        # Bid-ask bounce variance (simplified)
        bounce_var = returns.diff().rolling(window).var() / 2
        
        noise_ratio = bounce_var / (realized_var + 1e-8)
        return noise_ratio.fillna(0)

class AlternativeDataEngine:
    """Alternative data feature engineering"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def create_alternative_features(self, data: pd.DataFrame, 
                                  external_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Create features from alternative data sources"""
        
        features = pd.DataFrame(index=data.index)
        
        if external_data:
            # Sentiment features
            if 'sentiment' in external_data:
                features = self._add_sentiment_features(features, external_data['sentiment'])
            
            # On-chain features
            if 'onchain' in external_data:
                features = self._add_onchain_features(features, external_data['onchain'])
            
            # Macro features
            if 'macro' in external_data:
                features = self._add_macro_features(features, external_data['macro'])
        
        # Seasonal features
        features = self._add_seasonal_features(features, data.index)
        
        # Time-based features
        features = self._add_time_features(features, data.index)
        
        return features.fillna(0)
    
    def _add_sentiment_features(self, features: pd.DataFrame, 
                               sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features"""
        
        # Raw sentiment scores
        for col in sentiment_data.columns:
            features[f'sentiment_{col}'] = sentiment_data[col]
            
            # Sentiment momentum
            features[f'sentiment_{col}_momentum'] = sentiment_data[col].diff()
            features[f'sentiment_{col}_acceleration'] = sentiment_data[col].diff().diff()
            
            # Sentiment extremes
            features[f'sentiment_{col}_extreme'] = np.where(
                abs(sentiment_data[col]) > sentiment_data[col].rolling(30).quantile(0.9), 1, 0
            )
        
        return features
    
    def _add_onchain_features(self, features: pd.DataFrame, 
                            onchain_data: pd.DataFrame) -> pd.DataFrame:
        """Add on-chain metrics features"""
        
        for col in onchain_data.columns:
            # Raw values
            features[f'onchain_{col}'] = onchain_data[col]
            
            # Normalized values
            rolling_mean = onchain_data[col].rolling(30).mean()
            rolling_std = onchain_data[col].rolling(30).std()
            features[f'onchain_{col}_zscore'] = (onchain_data[col] - rolling_mean) / (rolling_std + 1e-8)
            
            # Momentum
            features[f'onchain_{col}_momentum'] = onchain_data[col].pct_change()
            
            # Percentile rank
            features[f'onchain_{col}_percentile'] = onchain_data[col].rolling(90).rank(pct=True)
        
        return features
    
    def _add_macro_features(self, features: pd.DataFrame, 
                          macro_data: pd.DataFrame) -> pd.DataFrame:
        """Add macro economic features"""
        
        for col in macro_data.columns:
            # Raw values
            features[f'macro_{col}'] = macro_data[col]
            
            # Changes
            features[f'macro_{col}_change'] = macro_data[col].pct_change()
            
            # Regime detection (simple)
            features[f'macro_{col}_regime'] = np.where(
                macro_data[col] > macro_data[col].rolling(60).mean(), 1, 0
            )
        
        return features
    
    def _add_seasonal_features(self, features: pd.DataFrame, 
                             index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add seasonal and calendar features"""
        
        # Day of week effects
        features['dow_monday'] = (index.dayofweek == 0).astype(int)
        features['dow_friday'] = (index.dayofweek == 4).astype(int)
        features['dow_weekend'] = (index.dayofweek >= 5).astype(int)
        
        # Hour of day effects (for intraday data)
        features['hour_open'] = (index.hour == 9).astype(int)
        features['hour_close'] = (index.hour == 16).astype(int)
        features['hour_lunch'] = ((index.hour >= 12) & (index.hour <= 14)).astype(int)
        
        # Month effects
        features['month'] = index.month
        features['quarter'] = index.quarter
        
        # Special periods
        features['month_end'] = (index.day >= 28).astype(int)
        features['quarter_end'] = ((index.month % 3 == 0) & (index.day >= 28)).astype(int)
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame, 
                         index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add time-based cyclical features"""
        
        # Cyclical encoding for time features
        features['hour_sin'] = np.sin(2 * np.pi * index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * index.hour / 24)
        
        features['dow_sin'] = np.sin(2 * np.pi * index.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * index.dayofweek / 7)
        
        features['month_sin'] = np.sin(2 * np.pi * index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * index.month / 12)
        
        return features

class TechnicalPatternEngine:
    """Technical pattern recognition engine"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def create_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical pattern features"""
        
        features = pd.DataFrame(index=data.index)
        high, low, close, volume = data['high'], data['low'], data['close'], data['volume']
        
        # Candlestick patterns
        features.update(self._candlestick_patterns(data))
        
        # Chart patterns
        features.update(self._chart_patterns(data))
        
        # Volume patterns
        features.update(self._volume_patterns(data))
        
        # Support/Resistance levels
        features.update(self._support_resistance_features(data))
        
        return features.fillna(0)
    
    def _candlestick_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect candlestick patterns"""
        
        patterns = pd.DataFrame(index=data.index)
        
        open_price, high, low, close = data['open'], data['high'], data['low'], data['close']
        
        # Basic patterns
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        
        # Doji
        patterns['doji'] = (body < (high - low) * 0.1).astype(int)
        
        # Hammer
        patterns['hammer'] = ((lower_shadow > body * 2) & 
                            (upper_shadow < body * 0.5) & 
                            (close > open_price)).astype(int)
        
        # Shooting star
        patterns['shooting_star'] = ((upper_shadow > body * 2) & 
                                   (lower_shadow < body * 0.5) & 
                                   (close < open_price)).astype(int)
        
        # Engulfing patterns
        prev_body = body.shift(1)
        patterns['bullish_engulfing'] = ((close > open_price) & 
                                       (close.shift(1) < open_price.shift(1)) & 
                                       (body > prev_body * 1.5)).astype(int)
        
        patterns['bearish_engulfing'] = ((close < open_price) & 
                                       (close.shift(1) > open_price.shift(1)) & 
                                       (body > prev_body * 1.5)).astype(int)
        
        return patterns
    
    def _chart_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect chart patterns"""
        
        patterns = pd.DataFrame(index=data.index)
        close = data['close']
        
        # Trend patterns
        patterns['higher_highs'] = self._detect_higher_highs(data['high'])
        patterns['lower_lows'] = self._detect_lower_lows(data['low'])
        
        # Reversal patterns
        patterns['double_top'] = self._detect_double_top(data)
        patterns['double_bottom'] = self._detect_double_bottom(data)
        
        # Breakout patterns
        patterns['resistance_break'] = self._detect_resistance_break(data)
        patterns['support_break'] = self._detect_support_break(data)
        
        return patterns
    
    def _volume_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect volume patterns"""
        
        patterns = pd.DataFrame(index=data.index)
        volume = data['volume']
        close = data['close']
        
        # Volume spikes
        vol_ma = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        patterns['volume_spike'] = (volume > vol_ma + 2 * vol_std).astype(int)
        
        # Price-volume divergence
        price_trend = close.rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        volume_trend = volume.rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        patterns['pv_divergence'] = (price_trend != volume_trend).astype(int)
        
        # On-balance volume patterns
        obv = self._calculate_obv(data)
        patterns['obv_bullish'] = (obv > obv.rolling(20).mean()).astype(int)
        
        return patterns
    
    def _support_resistance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Support and resistance level features"""
        
        features = pd.DataFrame(index=data.index)
        high, low, close = data['high'], data['low'], data['close']
        
        # Detect support/resistance levels
        support_levels = self._find_support_levels(low)
        resistance_levels = self._find_resistance_levels(high)
        
        # Distance to nearest levels
        features['distance_to_support'] = self._distance_to_nearest_level(close, support_levels)
        features['distance_to_resistance'] = self._distance_to_nearest_level(close, resistance_levels)
        
        # Level strength
        features['support_strength'] = self._calculate_level_strength(close, support_levels)
        features['resistance_strength'] = self._calculate_level_strength(close, resistance_levels)
        
        return features
    
    def _detect_higher_highs(self, high: pd.Series, window: int = 20) -> pd.Series:
        """Detect higher highs pattern"""
        peaks, _ = find_peaks(high, distance=window//2)
        hh_pattern = pd.Series(0, index=high.index)
        
        for i in range(1, len(peaks)):
            if high.iloc[peaks[i]] > high.iloc[peaks[i-1]]:
                hh_pattern.iloc[peaks[i]] = 1
        
        return hh_pattern
    
    def _detect_lower_lows(self, low: pd.Series, window: int = 20) -> pd.Series:
        """Detect lower lows pattern"""
        troughs, _ = find_peaks(-low, distance=window//2)
        ll_pattern = pd.Series(0, index=low.index)
        
        for i in range(1, len(troughs)):
            if low.iloc[troughs[i]] < low.iloc[troughs[i-1]]:
                ll_pattern.iloc[troughs[i]] = 1
        
        return ll_pattern
    
    def _detect_double_top(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        """Detect double top pattern"""
        high = data['high']
        peaks, _ = find_peaks(high, distance=window//4)
        dt_pattern = pd.Series(0, index=data.index)
        
        for i in range(1, len(peaks)):
            peak1_val = high.iloc[peaks[i-1]]
            peak2_val = high.iloc[peaks[i]]
            
            # Check if peaks are similar height
            if abs(peak1_val - peak2_val) / peak1_val < 0.02:
                dt_pattern.iloc[peaks[i]] = 1
        
        return dt_pattern
    
    def _detect_double_bottom(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        """Detect double bottom pattern"""
        low = data['low']
        troughs, _ = find_peaks(-low, distance=window//4)
        db_pattern = pd.Series(0, index=data.index)
        
        for i in range(1, len(troughs)):
            trough1_val = low.iloc[troughs[i-1]]
            trough2_val = low.iloc[troughs[i]]
            
            # Check if troughs are similar depth
            if abs(trough1_val - trough2_val) / trough1_val < 0.02:
                db_pattern.iloc[troughs[i]] = 1
        
        return db_pattern
    
    def _detect_resistance_break(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        """Detect resistance breakout"""
        high = data['high']
        close = data['close']
        
        resistance = high.rolling(window).max()
        breakout = (close > resistance.shift(1) * 1.005).astype(int)  # 0.5% breakout
        
        return breakout
    
    def _detect_support_break(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        """Detect support breakdown"""
        low = data['low']
        close = data['close']
        
        support = low.rolling(window).min()
        breakdown = (close < support.shift(1) * 0.995).astype(int)  # 0.5% breakdown
        
        return breakdown
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        close = data['close']
        volume = data['volume']
        
        obv = pd.Series(0.0, index=data.index)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _find_support_levels(self, low: pd.Series, window: int = 50) -> List[float]:
        """Find support levels"""
        troughs, _ = find_peaks(-low, distance=window//4)
        return low.iloc[troughs].tolist()
    
    def _find_resistance_levels(self, high: pd.Series, window: int = 50) -> List[float]:
        """Find resistance levels"""
        peaks, _ = find_peaks(high, distance=window//4)
        return high.iloc[peaks].tolist()
    
    def _distance_to_nearest_level(self, price: pd.Series, levels: List[float]) -> pd.Series:
        """Calculate distance to nearest support/resistance level"""
        if not levels:
            return pd.Series(0, index=price.index)
        
        distances = []
        for p in price:
            min_distance = min(abs(p - level) / p for level in levels)
            distances.append(min_distance)
        
        return pd.Series(distances, index=price.index)
    
    def _calculate_level_strength(self, price: pd.Series, levels: List[float], 
                                tolerance: float = 0.01) -> pd.Series:
        """Calculate strength of support/resistance levels"""
        strength = pd.Series(0.0, index=price.index)
        
        for level in levels:
            touches = abs(price - level) / price < tolerance
            level_strength = touches.rolling(100).sum()
            strength = np.maximum(strength, level_strength)
        
        return strength

class FeatureSelector:
    """Advanced feature selection and engineering"""
    
    def __init__(self):
        self.logger = get_logger()
        self.selected_features = None
        self.feature_importance = None
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', 
                       k: int = 100) -> pd.DataFrame:
        """Select best features using various methods"""
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            # Use correlation-based selection
            return self._correlation_based_selection(X, y, k)
        
        X_selected = selector.fit_transform(X, y)
        selected_feature_names = X.columns[selector.get_support()]
        
        self.selected_features = selected_feature_names
        self.feature_importance = dict(zip(selected_feature_names, selector.scores_[selector.get_support()]))
        
        self.logger.info(f"Selected {len(selected_feature_names)} features using {method}")
        return pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
    
    def _correlation_based_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> pd.DataFrame:
        """Select features based on correlation with target"""
        
        correlations = abs(X.corrwith(y)).sort_values(ascending=False)
        selected_features = correlations.head(k).index
        
        self.selected_features = selected_features
        self.feature_importance = correlations.head(k).to_dict()
        
        return X[selected_features]
    
    def create_interaction_features(self, X: pd.DataFrame, 
                                  max_interactions: int = 50) -> pd.DataFrame:
        """Create interaction features between top features"""
        
        if self.selected_features is None or len(self.selected_features) < 2:
            return X
        
        # Select top features for interactions
        top_features = list(self.selected_features[:10])
        interaction_df = X.copy()
        
        interactions_created = 0
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                if interactions_created >= max_interactions:
                    break
                
                feat1, feat2 = top_features[i], top_features[j]
                
                # Multiplicative interaction
                interaction_df[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                
                # Ratio interaction (if no zero values)
                if (X[feat2] != 0).all():
                    interaction_df[f'{feat1}_div_{feat2}'] = X[feat1] / X[feat2]
                
                interactions_created += 2
        
        self.logger.info(f"Created {interactions_created} interaction features")
        return interaction_df
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for top features"""
        
        if self.selected_features is None:
            return X
        
        poly_df = X.copy()
        top_features = list(self.selected_features[:20])  # Top 20 features only
        
        for feat in top_features:
            for d in range(2, degree + 1):
                poly_df[f'{feat}_pow_{d}'] = X[feat] ** d
        
        return poly_df

class DimensionalityReducer:
    """Dimensionality reduction and manifold learning"""
    
    def __init__(self):
        self.logger = get_logger()
        self.reducers = {}
    
    def apply_pca(self, X: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """Apply PCA dimensionality reduction"""
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X.fillna(0))
        
        # Create feature names
        feature_names = [f'pca_{i}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=feature_names, index=X.index)
        
        self.reducers['pca'] = pca
        self.logger.info(f"PCA: {X.shape[1]} -> {n_components} features, "
                        f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return pca_df
    
    def apply_ica(self, X: pd.DataFrame, n_components: int = 30) -> pd.DataFrame:
        """Apply Independent Component Analysis"""
        
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        X_ica = ica.fit_transform(X.fillna(0))
        
        feature_names = [f'ica_{i}' for i in range(n_components)]
        ica_df = pd.DataFrame(X_ica, columns=feature_names, index=X.index)
        
        self.reducers['ica'] = ica
        self.logger.info(f"ICA: {X.shape[1]} -> {n_components} independent components")
        
        return ica_df
    
    def apply_clustering_features(self, X: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
        """Create clustering-based features"""
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X.fillna(0))
        
        cluster_features = pd.DataFrame(index=X.index)
        cluster_features['cluster'] = clusters
        
        # Distance to each cluster center
        distances = kmeans.transform(X.fillna(0))
        for i in range(n_clusters):
            cluster_features[f'dist_cluster_{i}'] = distances[:, i]
        
        # Distance to nearest cluster center
        cluster_features['dist_nearest_cluster'] = distances.min(axis=1)
        
        self.reducers['kmeans'] = kmeans
        self.logger.info(f"Created {len(cluster_features.columns)} clustering features")
        
        return cluster_features

class ComprehensiveFeatureEngine:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self):
        self.logger = get_logger()
        self.statistical_engine = StatisticalFeatureEngine()
        self.microstructure_engine = MarketMicrostructureEngine()
        self.alternative_engine = AlternativeDataEngine()
        self.pattern_engine = TechnicalPatternEngine()
        self.feature_selector = FeatureSelector()
        self.dimensionality_reducer = DimensionalityReducer()
        
        self.feature_pipeline = []
        self.final_features = None
    
    def create_comprehensive_features(self, data: pd.DataFrame,
                                    external_data: Dict[str, pd.DataFrame] = None,
                                    target: pd.Series = None) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        self.logger.info("Creating comprehensive feature set...")
        
        # Start with basic technical indicators
        features = data.copy()
        
        # Add statistical features
        stat_features = self.statistical_engine.create_statistical_features(data)
        features = pd.concat([features, stat_features], axis=1)
        self.logger.info(f"Added {len(stat_features.columns)} statistical features")
        
        # Add microstructure features
        micro_features = self.microstructure_engine.create_microstructure_features(data)
        features = pd.concat([features, micro_features], axis=1)
        self.logger.info(f"Added {len(micro_features.columns)} microstructure features")
        
        # Add alternative data features
        alt_features = self.alternative_engine.create_alternative_features(data, external_data)
        features = pd.concat([features, alt_features], axis=1)
        self.logger.info(f"Added {len(alt_features.columns)} alternative features")
        
        # Add pattern features
        pattern_features = self.pattern_engine.create_pattern_features(data)
        features = pd.concat([features, pattern_features], axis=1)
        self.logger.info(f"Added {len(pattern_features.columns)} pattern features")
        
        # Clean features
        features = self._clean_features(features)
        
        # Feature selection if target provided
        if target is not None:
            # Align target with features
            aligned_target = target.reindex(features.index).fillna(1)  # Default to hold
            
            # Select best features
            selected_features = self.feature_selector.select_features(features, aligned_target, k=100)
            
            # Create interaction features
            interaction_features = self.feature_selector.create_interaction_features(selected_features)
            
            # Apply dimensionality reduction
            pca_features = self.dimensionality_reducer.apply_pca(interaction_features, n_components=50)
            cluster_features = self.dimensionality_reducer.apply_clustering_features(selected_features, n_clusters=8)
            
            # Combine final features
            final_features = pd.concat([
                selected_features,
                pca_features,
                cluster_features
            ], axis=1)
            
        else:
            final_features = features
        
        # Final cleaning
        final_features = self._clean_features(final_features)
        
        self.final_features = final_features
        self.logger.success(f"Comprehensive feature engineering complete: {len(final_features.columns)} final features")
        
        return final_features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        
        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant columns
        constant_columns = features.columns[features.nunique() <= 1]
        if len(constant_columns) > 0:
            features = features.drop(columns=constant_columns)
            self.logger.debug(f"Removed {len(constant_columns)} constant columns")
        
        # Remove highly correlated features
        features = self._remove_highly_correlated(features)
        
        return features
    
    def _remove_highly_correlated(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""
        
        if len(features.columns) < 2:
            return features
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            features = features.drop(columns=to_drop)
            self.logger.debug(f"Removed {len(to_drop)} highly correlated features")
        
        return features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        
        return self.feature_selector.feature_importance or {}
    
    def transform_new_data(self, data: pd.DataFrame,
                          external_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        
        if self.final_features is None:
            raise ValueError("Feature pipeline not fitted. Call create_comprehensive_features first.")
        
        # Apply same transformations
        features = self.create_comprehensive_features(data, external_data)
        
        # Ensure same columns as training
        missing_cols = set(self.final_features.columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0
        
        # Reorder columns to match training
        features = features[self.final_features.columns]
        
        return features

# Usage example
def main():
    """Test comprehensive feature engineering"""
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + abs(np.random.randn(1000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - abs(np.random.randn(1000) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'volume': 1000000 + np.random.randint(-100000, 100000, 1000)
    }, index=dates)
    
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Create target
    target = pd.Series(
        np.random.choice([0, 1, 2], 1000, p=[0.3, 0.4, 0.3]),
        index=dates
    )
    
    # Create feature engine
    feature_engine = ComprehensiveFeatureEngine()
    
    # Generate features
    features = feature_engine.create_comprehensive_features(data, target=target)
    
    print(f"Original data shape: {data.shape}")
    print(f"Final features shape: {features.shape}")
    print(f"Feature importance (top 10):")
    
    importance = feature_engine.get_feature_importance()
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {score:.4f}")

if __name__ == "__main__":
    main()