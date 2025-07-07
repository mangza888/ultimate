#!/usr/bin/env python3
# ai/model_factory.py - Deep Learning Model Factory
# สำหรับสร้าง Deep Learning Models

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List  # เพิ่ม List
import numpy as np

class SimpleMLP(pl.LightningModule):
    """Simple Multi-Layer Perceptron"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 3, 
                 dropout: float = 0.2, learning_rate: float = 0.001):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SimpleLSTM(pl.LightningModule):
    """Simple LSTM for sequence prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 output_dim: int = 3, dropout: float = 0.2, learning_rate: float = 0.001):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # LSTM expects (batch, seq, features)
        # If input is 2D, treat each sample as a sequence of length 1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        output = self.classifier(last_output)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SimpleTransformer(pl.LightningModule):
    """Simple Transformer for sequence prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, output_dim: int = 3, dropout: float = 0.1, 
                 learning_rate: float = 0.001):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # If input is 2D, treat each sample as a sequence of length 1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class DeepLearningModelFactory:
    """Factory for creating deep learning models"""
    
    @staticmethod
    def create_model(model_name: str, config: Dict[str, Any]) -> pl.LightningModule:
        """Create a deep learning model based on name and config"""
        
        try:
            architecture = config.get('architecture', {})
            training = config.get('training', {})
            
            input_dim = architecture.get('input_dim', 50)
            output_dim = architecture.get('output_dim', 3)
            learning_rate = training.get('learning_rate', 0.001)
            dropout = architecture.get('dropout', 0.2)
            
            if model_name.lower() == 'mlp':
                hidden_dims = architecture.get('hidden_dims', [128, 64, 32])
                
                return SimpleMLP(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    dropout=dropout,
                    learning_rate=learning_rate
                )
                
            elif model_name.lower() == 'lstm':
                hidden_dim = architecture.get('hidden_dim', 64)
                num_layers = architecture.get('num_layers', 2)
                
                return SimpleLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    dropout=dropout,
                    learning_rate=learning_rate
                )
                
            elif model_name.lower() == 'transformer':
                d_model = architecture.get('d_model', 64)
                nhead = architecture.get('nhead', 4)
                num_layers = architecture.get('num_layers', 2)
                
                return SimpleTransformer(
                    input_dim=input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    dropout=dropout,
                    learning_rate=learning_rate
                )
                
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            # Return default MLP as fallback
            return SimpleMLP(
                input_dim=50,
                hidden_dims=[64, 32],
                output_dim=3,
                dropout=0.2,
                learning_rate=0.001
            )
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types"""
        return ['mlp', 'lstm', 'transformer']