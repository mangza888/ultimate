#!/usr/bin/env python3
# ai/traditional_ml.py - Traditional Machine Learning Models
# XGBoost, LightGBM, Random Forest สำหรับการเทรด

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from utils.config_manager import get_config
from utils.logger import get_logger

# ML Libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class TraditionalMLTrainer:
    """Traditional ML Models Trainer"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.logger = get_logger()
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    async def train_all_models(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all traditional ML models"""
        try:
            self.logger.info("Training traditional ML models...")
            
            # Prepare combined dataset
            combined_data = self._prepare_combined_dataset(training_data)
            if combined_data is None:
                return {}
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_train_test_split(combined_data)
            
            results = {}
            
            # Train XGBoost
            if XGB_AVAILABLE:
                xgb_result = await self._train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
                if xgb_result:
                    results['xgboost'] = xgb_result
            
            # Train LightGBM
            if LGB_AVAILABLE:
                lgb_result = await self._train_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test)
                if lgb_result:
                    results['lightgbm'] = lgb_result
            
            # Train Random Forest
            rf_result = await self._train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
            if rf_result:
                results['random_forest'] = rf_result
            
            # Train MLP
            mlp_result = await self._train_mlp(X_train, X_val, X_test, y_train, y_val, y_test)
            if mlp_result:
                results['mlp'] = mlp_result
            
            # Create ensemble
            if len(results) > 1:
                ensemble_result = await self._create_ensemble(results, X_test, y_test)
                if ensemble_result:
                    results['ensemble'] = ensemble_result
            
            self.logger.success(f"Traditional ML training completed: {len(results)} models")
            return results
            
        except Exception as e:
            self.logger.error(f"Traditional ML training failed: {e}")
            return {}
    
    def _prepare_combined_dataset(self, training_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare combined dataset from multiple symbols"""
        try:
            combined_dfs = []
            
            for symbol, df in training_data.items():
                # Create features
                features_df = self._create_features(df)
                
                # Create target
                target_df = self._create_target(features_df)
                
                combined_dfs.append(target_df)
            
            if not combined_dfs:
                self.logger.error("No data available for training")
                return None
            
            # Combine all dataframes
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            
            # Remove NaN values
            combined_df = combined_df.dropna()
            
            if len(combined_df) < 100:
                self.logger.error("Insufficient data after cleaning")
                return None
            
            self.logger.info(f"Combined dataset: {len(combined_df)} samples")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error preparing combined dataset: {e}")
            return None
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features"""
        try:
            features_df = df.copy()
            
            # Price features
            close = features_df['close']
            high = features_df['high']
            low = features_df['low']
            volume = features_df['volume']
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                sma = close.rolling(window=period, min_periods=1).mean()
                features_df[f'sma_{period}'] = sma
                features_df[f'price_to_sma_{period}'] = close / sma
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_histogram'] = macd - macd_signal
            
            # Bollinger Bands
            bb_sma = close.rolling(window=20, min_periods=1).mean()
            bb_std = close.rolling(window=20, min_periods=1).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            features_df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
            
            # Volatility
            returns = close.pct_change().fillna(0)
            for window in [5, 10, 20]:
                features_df[f'volatility_{window}'] = returns.rolling(window=window, min_periods=1).std()
            
            # Momentum
            for period in [1, 5, 10, 20]:
                features_df[f'momentum_{period}'] = close.pct_change(periods=period).fillna(0)
            
            # Volume indicators
            volume_sma = volume.rolling(window=20, min_periods=1).mean()
            features_df['volume_ratio'] = volume / (volume_sma + 1e-8)
            
            # Price range
            features_df['price_range'] = (high - low) / (close + 1e-8)
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return df
    
    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trading target"""
        try:
            target_df = df.copy()
            
            # Create trading signals based on multiple strategies
            signals = np.zeros(len(target_df))
            
            # Strategy 1: RSI
            if 'rsi' in target_df.columns:
                rsi = target_df['rsi']
                signals += (rsi < 30).astype(int)  # Oversold -> Buy
                signals -= (rsi > 70).astype(int)  # Overbought -> Sell
            
            # Strategy 2: MACD
            if 'macd' in target_df.columns and 'macd_signal' in target_df.columns:
                macd = target_df['macd']
                macd_signal = target_df['macd_signal']
                signals += (macd > macd_signal).astype(int)  # MACD above signal -> Buy
                signals -= (macd < macd_signal).astype(int)  # MACD below signal -> Sell
            
            # Strategy 3: Momentum
            if 'momentum_5' in target_df.columns:
                momentum = target_df['momentum_5']
                signals += (momentum > 0.01).astype(int)  # Strong positive momentum -> Buy
                signals -= (momentum < -0.01).astype(int)  # Strong negative momentum -> Sell
            
            # Strategy 4: Bollinger Bands
            if 'bb_position' in target_df.columns:
                bb_pos = target_df['bb_position']
                signals += (bb_pos < 0.2).astype(int)  # Near lower band -> Buy
                signals -= (bb_pos > 0.8).astype(int)  # Near upper band -> Sell
            
            # Convert to classification labels
            # 0: Sell, 1: Hold, 2: Buy
            thresholds = np.percentile(signals, [33, 67])
            target_df['target'] = np.digitize(signals, thresholds)
            
            return target_df
            
        except Exception as e:
            self.logger.error(f"Error creating target: {e}")
            target_df['target'] = np.random.choice([0, 1, 2], len(target_df))
            return target_df
    
    def _prepare_train_test_split(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare train/validation/test split"""
        try:
            # Select feature columns
            feature_columns = [col for col in data.columns if col not in ['target', 'timestamp', 'symbol']]
            X = data[feature_columns].values
            y = data['target'].values
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['main'] = scaler
            
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing train/test split: {e}")
            raise
    
    async def _train_xgboost(self, X_train, X_val, X_test, y_train, y_val, y_test) -> Optional[Dict[str, Any]]:
        """Train XGBoost model"""
        try:
            self.logger.info("Training XGBoost...")
            
            # Get config
            xgb_config = self.config.get('ai_models.traditional_ml.xgboost', {})
            params = xgb_config.get('params', {})
            
            # Set default parameters
            default_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'tree_method': 'gpu_hist' if self.config.get('gpu.enabled') else 'hist',
                'gpu_id': 0 if self.config.get('gpu.enabled') else None
            }
            
            # Merge with config
            final_params = {**default_params, **params}
            
            # Remove None values
            final_params = {k: v for k, v in final_params.items() if v is not None}
            
            # Create model
            model = xgb.XGBClassifier(**final_params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            # Calculate win rate
            win_rate = self._calculate_win_rate(y_test, test_pred)
            
            # Store model
            self.models['xgboost'] = model
            
            result = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'win_rate': win_rate,
                'model_type': 'xgboost'
            }
            
            self.logger.success(f"XGBoost: Test Acc {test_acc:.3f}, Win Rate {win_rate:.2f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            return None
    
    async def _train_lightgbm(self, X_train, X_val, X_test, y_train, y_val, y_test) -> Optional[Dict[str, Any]]:
        """Train LightGBM model"""
        try:
            self.logger.info("Training LightGBM...")
            
            # Get config
            lgb_config = self.config.get('ai_models.traditional_ml.lightgbm', {})
            params = lgb_config.get('params', {})
            
            # Set default parameters
            default_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'device': 'gpu' if self.config.get('gpu.enabled') else 'cpu',
                'verbose': -1
            }
            
            # Merge with config
            final_params = {**default_params, **params}
            
            # Create model
            model = lgb.LGBMClassifier(**final_params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            # Calculate win rate
            win_rate = self._calculate_win_rate(y_test, test_pred)
            
            # Store model
            self.models['lightgbm'] = model
            
            result = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'win_rate': win_rate,
                'model_type': 'lightgbm'
            }
            
            self.logger.success(f"LightGBM: Test Acc {test_acc:.3f}, Win Rate {win_rate:.2f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            return None
    
    async def _train_random_forest(self, X_train, X_val, X_test, y_train, y_val, y_test) -> Optional[Dict[str, Any]]:
        """Train Random Forest model"""
        try:
            self.logger.info("Training Random Forest...")
            
            # Get config
            rf_config = self.config.get('ai_models.traditional_ml.random_forest', {})
            params = rf_config.get('params', {})
            
            # Set default parameters
            default_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Merge with config
            final_params = {**default_params, **params}
            
            # Create model
            model = RandomForestClassifier(**final_params)
            
#!/usr/bin/env python3
# ai/traditional_ml.py - Traditional Machine Learning Models (ส่วนที่ขาดหายไป)
# ต่อจากบรรทัดที่ขาดหายไป

        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Calculate win rate
        win_rate = self._calculate_win_rate(y_test, test_pred)
        
        # Store model
        self.models['random_forest'] = model
        
        result = {
            'model': model,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'win_rate': win_rate,
            'model_type': 'random_forest'
        }
        
        self.logger.success(f"Random Forest: Test Acc {test_acc:.3f}, Win Rate {win_rate:.2f}%")
        return result
        
    except Exception as e:
        self.logger.error(f"Random Forest training failed: {e}")
        return None

async def _train_mlp(self, X_train, X_val, X_test, y_train, y_val, y_test) -> Optional[Dict[str, Any]]:
    """Train MLP model"""
    try:
        self.logger.info("Training MLP...")
        
        # Create MLP model
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Calculate win rate
        win_rate = self._calculate_win_rate(y_test, test_pred)
        
        # Store model
        self.models['mlp'] = model
        
        result = {
            'model': model,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'win_rate': win_rate,
            'model_type': 'mlp'
        }
        
        self.logger.success(f"MLP: Test Acc {test_acc:.3f}, Win Rate {win_rate:.2f}%")
        return result
        
    except Exception as e:
        self.logger.error(f"MLP training failed: {e}")
        return None

async def _create_ensemble(self, results: Dict[str, Any], X_test, y_test) -> Optional[Dict[str, Any]]:
    """Create ensemble model"""
    try:
        self.logger.info("Creating ensemble model...")
        
        # Get individual models
        models = []
        names = []
        
        for name, result in results.items():
            if isinstance(result, dict) and 'model' in result:
                models.append((name, result['model']))
                names.append(name)
        
        if len(models) < 2:
            self.logger.warning("Not enough models for ensemble")
            return None
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=models,
            voting='soft'
        )
        
        # Train ensemble (this is just to set it up)
        voting_clf.fit(X_test[:100], y_test[:100])  # Small sample for setup
        
        # Evaluate ensemble
        ensemble_pred = voting_clf.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        # Calculate win rate
        win_rate = self._calculate_win_rate(y_test, ensemble_pred)
        
        # Store ensemble
        self.models['ensemble'] = voting_clf
        
        result = {
            'model': voting_clf,
            'test_accuracy': ensemble_acc,
            'win_rate': win_rate,
            'model_type': 'ensemble',
            'base_models': names
        }
        
        self.logger.success(f"Ensemble: Test Acc {ensemble_acc:.3f}, Win Rate {win_rate:.2f}%")
        return result
        
    except Exception as e:
        self.logger.error(f"Ensemble creation failed: {e}")
        return None

def _calculate_win_rate(self, y_true, y_pred) -> float:
    """Calculate win rate for trading signals"""
    try:
        # Convert predictions to trading signals
        # 0: Sell, 1: Hold, 2: Buy
        
        # Calculate win rate based on correct directional predictions
        buy_signals = (y_pred == 2)
        sell_signals = (y_pred == 0)
        
        # True buy signals (should be profitable)
        correct_buys = buy_signals & (y_true == 2)
        # True sell signals (should be profitable)
        correct_sells = sell_signals & (y_true == 0)
        
        total_signals = buy_signals.sum() + sell_signals.sum()
        correct_signals = correct_buys.sum() + correct_sells.sum()
        
        if total_signals == 0:
            return 0.0
        
        win_rate = (correct_signals / total_signals) * 100
        return win_rate
        
    except Exception as e:
        self.logger.error(f"Error calculating win rate: {e}")
        return 0.0

def save_models(self, save_dir: str = "models/traditional_ml"):
    """Save trained models"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {name} model to {model_path}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f"{name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            self.logger.info(f"Saved {name} scaler to {scaler_path}")
        
        self.logger.success(f"All models saved to {save_dir}")
        
    except Exception as e:
        self.logger.error(f"Error saving models: {e}")

def load_models(self, load_dir: str = "models/traditional_ml"):
    """Load trained models"""
    try:
        model_files = [f for f in os.listdir(load_dir) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '')
            model_path = os.path.join(load_dir, model_file)
            
            self.models[model_name] = joblib.load(model_path)
            self.logger.info(f"Loaded {model_name} model from {model_path}")
        
        # Load scalers
        scaler_files = [f for f in os.listdir(load_dir) if f.endswith('_scaler.pkl')]
        
        for scaler_file in scaler_files:
            scaler_name = scaler_file.replace('_scaler.pkl', '')
            scaler_path = os.path.join(load_dir, scaler_file)
            
            self.scalers[scaler_name] = joblib.load(scaler_path)
            self.logger.info(f"Loaded {scaler_name} scaler from {scaler_path}")
        
        self.logger.success(f"All models loaded from {load_dir}")
        
    except Exception as e:
        self.logger.error(f"Error loading models: {e}")

def predict(self, model_name: str, features: np.ndarray) -> Tuple[int, float]:
    """Make prediction using specific model"""
    try:
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return 1, 0.0  # Default to HOLD with 0 confidence
        
        model = self.models[model_name]
        scaler = self.scalers.get('main')
        
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities.max()
        else:
            confidence = 0.5  # Default confidence for models without probability
        
        return prediction, confidence
        
    except Exception as e:
        self.logger.error(f"Error making prediction: {e}")
        return 1, 0.0  # Default to HOLD with 0 confidence

def get_model_summary(self) -> Dict[str, Any]:
    """Get summary of all trained models"""
    summary = {
        'total_models': len(self.models),
        'available_models': list(self.models.keys()),
        'scalers': list(self.scalers.keys()),
        'results': self.results
    }
    
    return summary

# Available traditional ML models
AVAILABLE_MODELS = []
if XGB_AVAILABLE:
    AVAILABLE_MODELS.append('xgboost')
if LGB_AVAILABLE:
    AVAILABLE_MODELS.append('lightgbm')
AVAILABLE_MODELS.extend(['random_forest', 'mlp', 'ensemble'])

# Model priority (higher is better)
MODEL_PRIORITY = {
    'xgboost': 10,
    'lightgbm': 9,
    'ensemble': 8,
    'random_forest': 7,
    'mlp': 6
}