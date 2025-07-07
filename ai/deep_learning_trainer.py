#!/usr/bin/env python3
# ai/deep_learning_models.py - Fixed Deep Learning Models
# แก้ไขปัญหา GPU และ library compatibility

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

class MLPModel(nn.Module):
    """Simple MLP Model"""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class LSTMModel(nn.Module):
    """LSTM Model for sequence prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output

class TransformerModel(nn.Module):
    """Transformer Model for sequence prediction"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, 
                 num_layers: int, output_dim: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Take the last output
        x = x[:, -1, :]
        x = self.dropout(x)
        output = self.fc(x)
        return output

class DeepLearningTrainer:
    """Deep Learning Models Trainer - Fixed version"""
    
    def __init__(self):
        self.logger = get_logger()
        
        # Setup device (fixed GPU handling)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # Use first available GPU
            torch.cuda.set_device(0)
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            self.logger.warning("GPU not available, using CPU")
        
        self.models = {}
        self.scalers = {}
    
    async def train_all_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train all deep learning models"""
        
        try:
            self.logger.info("Training deep learning models...")
            
            results = {}
            
            # Prepare combined data
            combined_data = []
            for symbol, data in training_data.items():
                if isinstance(data, pd.DataFrame) and len(data) > 50:
                    combined_data.append(data)
            
            if not combined_data:
                self.logger.warning("No valid training data available")
                return {}
            
            # Combine all data
            full_data = pd.concat(combined_data, ignore_index=True)
            
            # Train each model type
            models_to_train = [
                ('mlp', self._train_mlp),
                ('lstm', self._train_lstm), 
                ('transformer', self._train_transformer)
            ]
            
            for model_name, train_func in models_to_train:
                try:
                    self.logger.info(f"Training {model_name}...")
                    model_result = await train_func(full_data)
                    if model_result:
                        results[model_name] = model_result
                        self.logger.success(f"{model_name} training completed")
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
                    continue
            
            self.logger.success(f"Deep learning training completed: {len(results)} models")
            return results
            
        except Exception as e:
            self.logger.error(f"Deep learning training failed: {e}")
            return {}
    
    async def _train_mlp(self, data: pd.DataFrame) -> Optional[Tuple[Any, Dict[str, float]]]:
        """Train MLP model"""
        
        try:
            # Prepare data
            X, y, scaler = self._prepare_data(data)
            if X is None:
                return None
            
            # Create model
            model = MLPModel(
                input_dim=X.shape[1],
                hidden_dims=[128, 64, 32],
                output_dim=3,
                dropout=0.2
            ).to(self.device)
            
            # Train model
            trained_model, metrics = self._train_model(model, X, y, epochs=50)
            
            if trained_model:
                self.models['mlp'] = trained_model
                self.scalers['mlp'] = scaler
                return trained_model, metrics
            
            return None
            
        except Exception as e:
            self.logger.error(f"MLP training failed: {e}")
            return None
    
    async def _train_lstm(self, data: pd.DataFrame) -> Optional[Tuple[Any, Dict[str, float]]]:
        """Train LSTM model"""
        
        try:
            # Prepare sequence data
            X, y, scaler = self._prepare_sequence_data(data, sequence_length=20)
            if X is None:
                return None
            
            # Create model
            model = LSTMModel(
                input_dim=X.shape[2],
                hidden_dim=64,
                num_layers=2,
                output_dim=3,
                dropout=0.2
            ).to(self.device)
            
            # Train model
            trained_model, metrics = self._train_model(model, X, y, epochs=50)
            
            if trained_model:
                self.models['lstm'] = trained_model
                self.scalers['lstm'] = scaler
                return trained_model, metrics
            
            return None
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            return None
    
    async def _train_transformer(self, data: pd.DataFrame) -> Optional[Tuple[Any, Dict[str, float]]]:
        """Train Transformer model"""
        
        try:
            # Prepare sequence data
            X, y, scaler = self._prepare_sequence_data(data, sequence_length=20)
            if X is None:
                return None
            
            # Create model
            model = TransformerModel(
                input_dim=X.shape[2],
                d_model=64,
                nhead=4,
                num_layers=2,
                output_dim=3,
                dropout=0.1
            ).to(self.device)
            
            # Train model
            trained_model, metrics = self._train_model(model, X, y, epochs=50)
            
            if trained_model:
                self.models['transformer'] = trained_model
                self.scalers['transformer'] = scaler
                return trained_model, metrics
            
            return None
            
        except Exception as e:
            self.logger.error(f"Transformer training failed: {e}")
            return None
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """Prepare data for training"""
        
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Create features (simplified)
            close = data['close']
            
            # Simple technical indicators
            features = []
            
            # Price features
            features.append(close.values)
            
            # Moving averages
            for window in [5, 10, 20]:
                ma = close.rolling(window=window, min_periods=1).mean()
                features.append(ma.values)
                features.append((close / ma).values)
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
            features.append(rsi.values)
            
            # Stack features
            X = np.column_stack(features)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Create labels (simplified)
            returns = close.pct_change().fillna(0)
            y = np.where(returns > 0.01, 2, np.where(returns < -0.01, 0, 1))
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Remove first 20 rows to ensure all features are valid
            X_scaled = X_scaled[20:]
            y = y[20:]
            
            self.logger.debug(f"Prepared data: X shape {X_scaled.shape}, y shape {y.shape}")
            
            return X_scaled, y, scaler
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None, None, None
    
    def _prepare_sequence_data(self, data: pd.DataFrame, sequence_length: int = 20) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """Prepare sequence data for LSTM/Transformer"""
        
        try:
            # Get regular features first
            X_flat, y_flat, scaler = self._prepare_data(data)
            if X_flat is None:
                return None, None, None
            
            # Create sequences
            sequences = []
            labels = []
            
            for i in range(sequence_length, len(X_flat)):
                sequences.append(X_flat[i-sequence_length:i])
                labels.append(y_flat[i])
            
            if not sequences:
                self.logger.warning("No sequences could be created")
                return None, None, None
            
            X_seq = np.array(sequences)
            y_seq = np.array(labels)
            
            self.logger.debug(f"Prepared sequence data: X shape {X_seq.shape}, y shape {y_seq.shape}")
            
            return X_seq, y_seq, scaler
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence data: {e}")
            return None, None, None
    
    def _train_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray, 
                    epochs: int = 50) -> Tuple[Optional[nn.Module], Dict[str, float]]:
        """Train a PyTorch model"""
        
        try:
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            model.train()
            best_val_acc = 0
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_predictions = torch.argmax(val_outputs, dim=1)
                        val_acc = (val_predictions == y_val).float().mean().item()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        self.logger.debug(f"Early stopping at epoch {epoch}")
                        break
                    
                    model.train()
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                train_predictions = torch.argmax(train_outputs, dim=1)
                train_acc = (train_predictions == y_train).float().mean().item()
                
                val_outputs = model(X_val)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_acc = (val_predictions == y_val).float().mean().item()
            
            # Calculate metrics
            metrics = {
                'accuracy': val_acc,
                'train_accuracy': train_acc,
                'win_rate': val_acc * 100,  # Use accuracy as win rate proxy
                'val_loss': criterion(val_outputs, y_val).item()
            }
            
            # Move model to CPU for compatibility
            model = model.cpu()
            
            self.logger.debug(f"Model training completed. Val accuracy: {val_acc:.4f}")
            
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None, {}

# Wrapper class for compatibility
class DeepLearningModelFactory:
    """Factory for creating deep learning models"""
    
    def __init__(self):
        self.trainer = DeepLearningTrainer()
    
    async def train_all_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train all models"""
        return await self.trainer.train_all_models(training_data)