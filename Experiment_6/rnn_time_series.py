"""
Experiment 6: RNN using PyTorch for Time Series Prediction
==========================================================
Objective: Train and evaluate a Recurrent Neural Network using PyTorch
           to predict the next value in a sample time series dataset.

Author: B.E. (AI & DS) VI Semester Student
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

output_dir = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Generate Sample Time Series Dataset
# ============================================================
print("\n" + "=" * 60)
print("  EXPERIMENT 6: RNN FOR TIME SERIES PREDICTION")
print("=" * 60)

print(f"\n[1] Generating sample time series dataset...")

# Create a complex time series: combination of sinusoidal waves + trend + noise
t = np.linspace(0, 100, 2000)
# Primary wave + secondary wave + trend + noise
series = (np.sin(0.1 * t) * 10 + 
          np.sin(0.3 * t) * 5 + 
          np.cos(0.05 * t) * 3 + 
          0.02 * t +  # slight upward trend
          np.random.normal(0, 0.5, len(t)))  # noise

print(f"  Time series length: {len(series)}")
print(f"  Min value: {series.min():.2f}")
print(f"  Max value: {series.max():.2f}")
print(f"  Mean: {series.mean():.2f}")

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
series_normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# ============================================================
# Create Sequences for RNN
# ============================================================
SEQUENCE_LENGTH = 50

def create_sequences(data, seq_length):
    """Create input-output pairs for time series prediction."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(series_normalized, SEQUENCE_LENGTH)

# Train/Test split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (batch, seq, features)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_t = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset_t = TensorDataset(X_test_t, y_test_t)

print(f"  Sequence length: {SEQUENCE_LENGTH}")
print(f"  Training sequences: {len(X_train)}")
print(f"  Test sequences: {len(X_test)}")

# ============================================================
# RNN Models
# ============================================================
class SimpleRNN(nn.Module):
    """Simple RNN for time series prediction."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out.squeeze()


class LSTMModel(nn.Module):
    """LSTM for time series prediction."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()


class GRUModel(nn.Module):
    """GRU for time series prediction."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


def train_model(model, train_loader, test_X, test_y, epochs=30, lr=0.001):
    """Train an RNN model and return metrics."""
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(train_loader))
        
        # Test loss
        model.eval()
        with torch.no_grad():
            test_pred = model(test_X.to(DEVICE))
            test_loss = criterion(test_pred, test_y.to(DEVICE))
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs}  Train Loss: {train_losses[-1]:.6f}  "
                  f"Test Loss: {test_losses[-1]:.6f}")
    
    elapsed = time.time() - start_time
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        predictions = model(test_X.to(DEVICE)).cpu().numpy()
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'predictions': predictions,
        'time': elapsed,
        'model': model
    }


# ============================================================
# Training All Three Models
# ============================================================
EPOCHS = 2
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

models_config = {
    'Simple RNN': SimpleRNN(hidden_size=64, num_layers=2),
    'LSTM': LSTMModel(hidden_size=64, num_layers=2),
    'GRU': GRUModel(hidden_size=64, num_layers=2),
}

results = {}
for name, model in models_config.items():
    print(f"\n[Training {name}]")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    results[name] = train_model(model, train_loader, X_test_t, y_test_t, epochs=EPOCHS)

# ============================================================
# Evaluation Metrics
# ============================================================
print(f"\n{'='*70}")
print(f"  MODEL COMPARISON")
print(f"{'='*70}")
print(f"  {'Model':<15} {'MSE':<12} {'MAE':<12} {'R² Score':<12} {'Time':<10}")
print(f"  {'-'*61}")

for name, res in results.items():
    preds = res['predictions']
    actuals = y_test
    
    # Inverse transform for real-scale metrics
    preds_real = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals_real = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(actuals_real, preds_real)
    mae = mean_absolute_error(actuals_real, preds_real)
    r2 = r2_score(actuals_real, preds_real)
    
    res['mse'] = mse
    res['mae'] = mae
    res['r2'] = r2
    res['preds_real'] = preds_real
    res['actuals_real'] = actuals_real
    
    print(f"  {name:<15} {mse:<12.4f} {mae:<12.4f} {r2:<12.4f} {res['time']:<10.1f}s")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Experiment 6: RNN for Time Series Prediction', fontsize=16, fontweight='bold')

# 1. Original Time Series
axes[0, 0].plot(t[:500], series[:500], color='#3498db', linewidth=1)
axes[0, 0].set_title('Sample Time Series (First 500 points)', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# 2. Training Loss Comparison
colors = {'Simple RNN': '#e74c3c', 'LSTM': '#3498db', 'GRU': '#2ecc71'}
for name, res in results.items():
    axes[0, 1].plot(res['train_losses'], label=name, color=colors[name], linewidth=2)
axes[0, 1].set_title('Training Loss Comparison', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Test Loss Comparison
for name, res in results.items():
    axes[0, 2].plot(res['test_losses'], label=name, color=colors[name], linewidth=2)
axes[0, 2].set_title('Test Loss Comparison', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('MSE Loss')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4-6: Predictions vs Actual for each model
for idx, (name, res) in enumerate(results.items()):
    ax = axes[1, idx]
    plot_range = min(200, len(res['actuals_real']))
    ax.plot(res['actuals_real'][:plot_range], label='Actual', color='black', linewidth=1.5, alpha=0.7)
    ax.plot(res['preds_real'][:plot_range], label='Predicted', color=colors[name], 
            linewidth=1.5, linestyle='--')
    ax.set_title(f'{name} Predictions (R²={res["r2"]:.4f})', fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rnn_time_series_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[✓] Plots saved to 'rnn_time_series_results.png'")

# Save the best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
torch.save(results[best_model_name]['model'].state_dict(), 
           os.path.join(output_dir, f'rnn_best_model_{best_model_name.lower().replace(" ", "_")}.pth'))
print(f"[✓] Best model ({best_model_name}) saved.")

# Save dataset
np.savez(os.path.join(output_dir, 'time_series_dataset.npz'),
         train_X=X_train, train_y=y_train, test_X=X_test, test_y=y_test,
         full_series=series, time=t)
print(f"[✓] Dataset saved to 'time_series_dataset.npz'")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Best Model: {best_model_name} (R²={results[best_model_name]['r2']:.4f})")
print(f"{'='*60}")
