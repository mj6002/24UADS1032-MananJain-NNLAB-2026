"""
Experiment 8: Mini Project 2 - Bike Driving Behavior Analysis (Models)
======================================================================
Dates: 
- Model Submissions: 21.03.2026
- Bonus Submissions: 30.03.2026

Author: B.E. (AI & DS) VI Semester Student
"""
import torch
import torch.nn as nn
import math

class RNNModel(nn.Module):
    """LSTM or GRU based model for driving behavior analysis."""
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2, model_type='LSTM'):
        super(RNNModel, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Score between 0 and 1
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.rnn(x)
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out.squeeze()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """Transformer based model for driving behavior analysis."""
    def __init__(self, input_size=6, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Global average pooling over time steps or take first token [CLS]
        # Here we take the mean across the sequence length
        out = x.mean(dim=1)
        out = self.fc(out)
        return out.squeeze()
