"""
Experiment 8: Mini Project 2 - Bike Driving Behavior Analysis (Trainer)
=======================================================================
Dates: 
- Model Submissions: 21.03.2026
- Bonus Submissions: 30.03.2026

Author: B.E. (AI & DS) VI Semester Student
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from models import RNNModel, TransformerModel
from dataset_generator import generate_bike_sensor_data

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 32
SEQ_LEN = 50
HIDDEN_SIZE = 64
LR = 0.001

def train_model(model, train_loader, val_loader, name):
    print(f"\nTraining {name}...")
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
            
    training_time = time.time() - start_time
    return train_losses, val_losses, training_time

def main():
    # 1. Prepare Data
    X, y = generate_bike_sensor_data(num_samples=1200, seq_len=SEQ_LEN)
    
    # Split
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)
    
    # 2. Initialize Models
    lstm_model = RNNModel(model_type='LSTM', hidden_size=HIDDEN_SIZE)
    transformer_model = TransformerModel(d_model=HIDDEN_SIZE)
    
    # 3. Train
    lstm_res = train_model(lstm_model, train_loader, val_loader, "LSTM")
    trans_res = train_model(transformer_model, train_loader, val_loader, "Transformer")
    
    # 4. Compare and Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_res[1], label='LSTM Val Loss')
    plt.plot(trans_res[1], label='Transformer Val Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    # Performance summary
    print("\n" + "="*40)
    print("      BONUS: MODEL PERFORMANCE COMPARISON")
    print("="*40)
    print(f"{'Model':<15} | {'Final Val Loss':<15} | {'Training Time':<15}")
    print("-" * 50)
    print(f"{'LSTM':<15} | {lstm_res[1][-1]:<15.4f} | {lstm_res[2]:<15.2f}s")
    print(f"{'Transformer':<15} | {trans_res[1][-1]:<15.4f} | {trans_res[2]:<15.2f}s")
    
    plt.subplot(1, 2, 2)
    models = ['LSTM', 'Transformer']
    times = [lstm_res[2], trans_res[2]]
    plt.bar(models, times, color=['blue', 'orange'])
    plt.title('Training Time Comparison')
    plt.ylabel('Seconds')
    
    output_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(output_dir, 'model_comparison_results.png'))
    print(f"\nResults saved to {os.path.join(output_dir, 'model_comparison_results.png')}")
    
    # Save the best model
    if lstm_res[1][-1] < trans_res[1][-1]:
        torch.save(lstm_model.state_dict(), os.path.join(output_dir, 'best_driving_model.pth'))
        print("Best model: LSTM")
    else:
        torch.save(transformer_model.state_dict(), os.path.join(output_dir, 'best_driving_model.pth'))
        print("Best model: Transformer")

if __name__ == "__main__":
    main()
