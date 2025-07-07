#!/usr/bin/env python3
# ai/deep_learning_models.py - Deep Learning Models
# LSTM, Transformer, CNN-LSTM, Attention-LSTM สำหรับการเทรด

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
import numpy as np
from utils.config_manager import get_config
from utils.logger import get_logger

class LSTMTradingModel(pl.LightningModule):
    """LSTM Model สำหรับการเทรด"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model architecture
        self.input_dim = config['architecture']['input_dim']
        self.hidden_dim = config['architecture']['hidden_dim']
        self.num_layers = config['architecture']['num_layers']
        self.dropout = config['architecture']['dropout']
        self.bidirectional = config['architecture']['bidirectional']
        self.num_classes = config['architecture']['num_classes']
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = self.hidden_dim * (2 if self.bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size // 2, self.num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs']
        )
        
        return [optimizer], [scheduler]

class TransformerTradingModel(pl.LightningModule):
    """Transformer Model สำหรับการเทรด"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model architecture
        self.input_dim = config['architecture']['input_dim']
        self.d_model = config['architecture']['d_model']
        self.nhead = config['architecture']['nhead']
        self.num_layers = config['architecture']['num_layers']
        self.dropout = config['architecture']['dropout']
        self.num_classes = config['architecture']['num_classes']
        self.sequence_length = config['architecture']['sequence_length']
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def _create_positional_encoding(self):
        pe = torch.zeros(self.sequence_length, self.d_model)
        position = torch.arange(0, self.sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        pooled = x.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs']
        )
        
        return [optimizer], [scheduler]

class CNNLSTMTradingModel(pl.LightningModule):
    """CNN-LSTM Hybrid Model สำหรับการเทรด"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model architecture
        self.input_dim = config['architecture']['input_dim']
        self.cnn_filters = config['architecture']['cnn_filters']
        self.cnn_kernel_size = config['architecture']['cnn_kernel_size']
        self.lstm_hidden = config['architecture']['lstm_hidden']
        self.lstm_layers = config['architecture']['lstm_layers']
        self.dropout = config['architecture']['dropout']
        self.num_classes = config['architecture']['num_classes']
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in self.cnn_filters:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, self.cnn_kernel_size, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
            )
            in_channels = out_channels
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_filters[-1],
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden, self.lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.lstm_hidden // 2, self.num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape for CNN (batch_size, 1, sequence_length * input_dim)
        x = x.view(batch_size, 1, seq_len * input_dim)
        
        # CNN forward pass
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Reshape back for LSTM (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # (batch_size, features, sequence_length)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        output = self.classifier(lstm_out[:, -1, :])
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs']
        )
        
        return [optimizer], [scheduler]

class AttentionLSTMTradingModel(pl.LightningModule):
    """Attention-LSTM Model สำหรับการเทรด"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model architecture
        self.input_dim = config['architecture']['input_dim']
        self.lstm_hidden = config['architecture']['lstm_hidden']
        self.lstm_layers = config['architecture']['lstm_layers']
        self.attention_dim = config['architecture']['attention_dim']
        self.dropout = config['architecture']['dropout']
        self.num_classes = config['architecture']['num_classes']
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, self.attention_dim),
            nn.Tanh(),
            nn.Linear(self.attention_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, self.lstm_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.lstm_hidden, self.num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs']
        )
        
        return [optimizer], [scheduler]

class DeepLearningModelFactory:
    """Factory สำหรับสร้าง Deep Learning Models"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> pl.LightningModule:
        """สร้าง model ตาม type"""
        models = {
            'lstm': LSTMTradingModel,
            'transformer': TransformerTradingModel,
            'cnn_lstm': CNNLSTMTradingModel,
            'attention_lstm': AttentionLSTMTradingModel
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type](config)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """ดึงรายชื่อ models ที่มี"""
        return ['lstm', 'transformer', 'cnn_lstm', 'attention_lstm']

# Model registry
MODEL_REGISTRY = {
    'lstm': LSTMTradingModel,
    'transformer': TransformerTradingModel,
    'cnn_lstm': CNNLSTMTradingModel,
    'attention_lstm': AttentionLSTMTradingModel
}