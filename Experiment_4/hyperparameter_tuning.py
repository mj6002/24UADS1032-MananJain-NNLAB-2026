"""
Experiment 4: Hyperparameter Tuning for BPNN (TensorFlow / Keras)
===================================================================
Objective: Evaluate performance of the three-layer neural network with
           variations in activation functions, hidden layer size, learning rate,
           batch size, and number of epochs.

Author: B.E. (AI & DS) VI Semester Student
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Data Loading
# ============================================================
# Mock dataset for instant generation of results
X_train_full = np.random.randint(0, 255, (100, 28, 28))
y_train_full = np.random.randint(0, 10, (100,))
X_test_full = np.random.randint(0, 255, (20, 28, 28))
y_test_full = np.random.randint(0, 10, (20,))

X_train = (X_train_full.reshape(-1, 784) / 255.0).astype(np.float32)
y_train = y_train_full.astype(np.int64)

X_test = (X_test_full.reshape(-1, 784) / 255.0).astype(np.float32)
y_test = y_test_full.astype(np.int64)

# ============================================================
# Configurable Three-Layer Network
# ============================================================
def build_model(hidden_size=128, activation='relu'):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(hidden_size, activation=activation),
        tf.keras.layers.Dense(hidden_size // 2, activation=activation),
        tf.keras.layers.Dense(10, activation='linear')
    ])
    return model

def train_and_evaluate(hidden_size, activation, lr, batch_size, epochs, verbose=True):
    """Train a model with given hyperparameters and return metrics."""
    
    model = build_model(hidden_size, activation)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    elapsed = time.time() - start_time
    
    train_losses = history.history['loss']
    test_accs = [acc * 100 for acc in history.history['val_accuracy']]
    final_acc = test_accs[-1]
    
    if verbose:
        print(f"  Hidden={hidden_size:<4} Act={activation:<12} LR={lr:<8} "
              f"Batch={batch_size:<4} Epochs={epochs:<3} -> "
              f"Acc={final_acc:.2f}%  Time={elapsed:.1f}s")
    
    return {
        'hidden_size': hidden_size,
        'activation': activation,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'train_losses': train_losses,
        'test_accs': test_accs,
        'final_acc': final_acc,
        'time': elapsed
    }

# ============================================================
# Hyperparameter Experiments
# ============================================================
print("\n" + "=" * 70)
print("  EXPERIMENT 4: HYPERPARAMETER TUNING FOR BPNN")
print("=" * 70)

all_results = []

# --- Experiment 4.1: Activation Functions ---
print(f"\n{'='*70}")
print(f"  4.1: Effect of Activation Functions")
print(f"  (Hidden=128, LR=0.001, Batch=64, Epochs=5)")
print(f"{'='*70}")
activation_results = []
for act in ['relu', 'sigmoid', 'tanh', 'elu']:
    result = train_and_evaluate(128, act, 0.001, 64, 5)
    activation_results.append(result)
    all_results.append(result)

# --- Experiment 4.2: Hidden Layer Size ---
print(f"\n{'='*70}")
print(f"  4.2: Effect of Hidden Layer Size")
print(f"  (Act=relu, LR=0.001, Batch=64, Epochs=5)")
print(f"{'='*70}")
hidden_results = []
for hs in [32, 64, 128, 256, 512]:
    result = train_and_evaluate(hs, 'relu', 0.001, 64, 5)
    hidden_results.append(result)
    all_results.append(result)

# --- Experiment 4.3: Learning Rate ---
print(f"\n{'='*70}")
print(f"  4.3: Effect of Learning Rate")
print(f"  (Hidden=128, Act=relu, Batch=64, Epochs=5)")
print(f"{'='*70}")
lr_results = []
for lr_val in [0.0001, 0.001, 0.01]:
    result = train_and_evaluate(128, 'relu', lr_val, 64, 5)
    lr_results.append(result)
    all_results.append(result)

# --- Experiment 4.4: Batch Size ---
print(f"\n{'='*70}")
print(f"  4.4: Effect of Batch Size")
print(f"  (Hidden=128, Act=relu, LR=0.001, Epochs=5)")
print(f"{'='*70}")
batch_results = []
for bs in [32, 64, 128, 256]:
    result = train_and_evaluate(128, 'relu', 0.001, bs, 5)
    batch_results.append(result)
    all_results.append(result)

# --- Experiment 4.5: Number of Epochs ---
print(f"\n{'='*70}")
print(f"  4.5: Effect of Number of Epochs")
print(f"  (Hidden=128, Act=relu, LR=0.001, Batch=64)")
print(f"{'='*70}")
epoch_results = []
for ep in [2, 5, 10, 15]:
    result = train_and_evaluate(128, 'relu', 0.001, 64, ep)
    epoch_results.append(result)
    all_results.append(result)

# ============================================================
# Summary Table
# ============================================================
print(f"\n{'='*90}")
print(f"  COMPLETE RESULTS SUMMARY")
print(f"{'='*90}")
best_result = max(all_results, key=lambda x: x['final_acc'])
for r in all_results:
    print(f"  {r['hidden_size']:<8} {r['activation']:<12} {r['lr']:<10} "
          f"{r['batch_size']:<8} {r['epochs']:<8} {r['final_acc']:<12.2f} {r['time']:<8.1f}")
print(f"\n  Best: Hidden={best_result['hidden_size']}, Act={best_result['activation']}, "
      f"LR={best_result['lr']}, Batch={best_result['batch_size']}, "
      f"Epochs={best_result['epochs']} -> {best_result['final_acc']:.2f}%")
print(f"{'='*90}")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Experiment 4: Hyperparameter Tuning for BPNN', fontsize=16, fontweight='bold')

# 4.1: Activation Functions
for r in activation_results:
    axes[0, 0].plot(r['test_accs'], label=r['activation'], linewidth=2, marker='o')
axes[0, 0].set_title('Effect of Activation Functions', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Test Accuracy (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 4.2: Hidden Layer Size
for r in hidden_results:
    axes[0, 1].plot(r['test_accs'], label=f"H={r['hidden_size']}", linewidth=2, marker='o')
axes[0, 1].set_title('Effect of Hidden Layer Size', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 4.3: Learning Rate
for r in lr_results:
    axes[0, 2].plot(r['test_accs'], label=f"LR={r['lr']}", linewidth=2, marker='o')
axes[0, 2].set_title('Effect of Learning Rate', fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4.4: Batch Size
for r in batch_results:
    axes[1, 0].plot(r['test_accs'], label=f"BS={r['batch_size']}", linewidth=2, marker='o')
axes[1, 0].set_title('Effect of Batch Size', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4.5: Number of Epochs
accs_by_epoch = [r['final_acc'] for r in epoch_results]
epochs_list = [r['epochs'] for r in epoch_results]
axes[1, 1].bar(range(len(epochs_list)), accs_by_epoch, edgecolor='black')
axes[1, 1].set_xticks(range(len(epochs_list)))
axes[1, 1].set_xticklabels(epochs_list)
axes[1, 1].set_title('Effect of Number of Epochs', fontweight='bold')
for i, v in enumerate(accs_by_epoch):
    axes[1, 1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hyperparameter_tuning_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[✓] Plots saved to 'hyperparameter_tuning_results.png'")
