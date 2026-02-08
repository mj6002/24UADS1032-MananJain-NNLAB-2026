"""
Experiment 1: Simulating Perceptron Learning Algorithm
=====================================================
Objective: Visualize the Perceptron Learning Algorithm using numpy and matplotlib.
           Evaluate performance of a single perceptron for NAND and XOR truth tables.

Author: B.E. (AI & DS) VI Semester Student
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Perceptron Class Implementation
# ============================================================
class Perceptron:
    """Single Layer Perceptron with step activation function."""
    
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1) * 0.5  # +1 for bias
        self.lr = learning_rate
        self.epochs = epochs
        self.weight_history = []
        self.error_history = []
    
    def activation(self, x):
        """Step activation function."""
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        """Predict output for a single input."""
        x_with_bias = np.insert(x, 0, 1)  # Insert bias term
        weighted_sum = np.dot(self.weights, x_with_bias)
        return self.activation(weighted_sum)
    
    def train(self, X, y):
        """Train the perceptron using the Perceptron Learning Rule."""
        for epoch in range(self.epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                total_error += abs(error)
                
                # Weight update rule: w = w + lr * error * x
                x_with_bias = np.insert(xi, 0, 1)
                self.weights += self.lr * error * x_with_bias
            
            self.weight_history.append(self.weights.copy())
            self.error_history.append(total_error)
            
            # Early stopping if converged
            if total_error == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break
        
        return self.error_history


def evaluate_perceptron(X, y, gate_name, perceptron):
    """Evaluate and print results for a given gate."""
    print(f"\n{'='*50}")
    print(f"  {gate_name} Gate Evaluation")
    print(f"{'='*50}")
    print(f"  Final Weights (bias, w1, w2): {perceptron.weights}")
    print(f"  Training Epochs: {len(perceptron.error_history)}")
    print(f"\n  {'Input1':<8} {'Input2':<8} {'Expected':<10} {'Predicted':<10} {'Correct':<8}")
    print(f"  {'-'*44}")
    
    correct = 0
    total = len(X)
    for xi, yi in zip(X, y):
        pred = perceptron.predict(xi)
        is_correct = "Y" if pred == yi else "N"
        if pred == yi:
            correct += 1
        print(f"  {xi[0]:<8} {xi[1]:<8} {yi:<10} {pred:<10} {is_correct:<8}")
    
    accuracy = (correct / total) * 100
    print(f"\n  Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


def plot_decision_boundary(X, y, perceptron, gate_name, ax):
    """Plot the decision boundary of the perceptron."""
    colors = ['red' if label == 0 else 'green' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', 
               linewidths=2, zorder=5)
    
    # Add labels to points
    for i, (xi, yi) in enumerate(zip(X, y)):
        ax.annotate(f'({int(xi[0])},{int(xi[1])})→{yi}', 
                    (xi[0], xi[1]), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, fontweight='bold')
    
    # Plot decision boundary if weights permit
    w = perceptron.weights
    if abs(w[2]) > 1e-6:  # Avoid division by zero
        x_range = np.linspace(-0.5, 1.5, 100)
        y_line = -(w[0] + w[1] * x_range) / w[2]
        ax.plot(x_range, y_line, 'b--', linewidth=2, label='Decision Boundary')
    elif abs(w[1]) > 1e-6:
        x_val = -w[0] / w[1]
        ax.axvline(x=x_val, color='b', linestyle='--', linewidth=2, label='Decision Boundary')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Input 1', fontsize=12)
    ax.set_ylabel('Input 2', fontsize=12)
    ax.set_title(f'{gate_name} Gate - Decision Boundary', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_error_curve(error_history, gate_name, ax):
    """Plot the training error curve."""
    ax.plot(range(1, len(error_history) + 1), error_history, 'b-o', 
            markersize=4, linewidth=2, color='crimson')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Error', fontsize=12)
    ax.set_title(f'{gate_name} Gate - Training Error Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)


# ============================================================
# Dataset: Truth Tables
# ============================================================
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# NAND Gate Truth Table
y_nand = np.array([1, 1, 1, 0])

# XOR Gate Truth Table
y_xor = np.array([0, 1, 1, 0])

# ============================================================
# Train and Evaluate NAND Gate
# ============================================================
print("\n" + "=" * 60)
print("  EXPERIMENT 1: PERCEPTRON LEARNING ALGORITHM")
print("=" * 60)

print("\n[1] Training Perceptron for NAND Gate...")
nand_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
nand_errors = nand_perceptron.train(X, y_nand)
nand_accuracy = evaluate_perceptron(X, y_nand, "NAND", nand_perceptron)

# ============================================================
# Train and Evaluate XOR Gate
# ============================================================
print("\n[2] Training Perceptron for XOR Gate...")
xor_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
xor_errors = xor_perceptron.train(X, y_xor)
xor_accuracy = evaluate_perceptron(X, y_xor, "XOR", xor_perceptron)

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Experiment 1: Perceptron Learning Algorithm\nNAND vs XOR Gate', 
             fontsize=16, fontweight='bold', y=0.98)

# NAND Gate plots
plot_decision_boundary(X, y_nand, nand_perceptron, "NAND", axes[0, 0])
plot_error_curve(nand_errors, "NAND", axes[0, 1])

# XOR Gate plots
plot_decision_boundary(X, y_xor, xor_perceptron, "XOR", axes[1, 0])
plot_error_curve(xor_errors, "XOR", axes[1, 1])

plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'perceptron_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[✓] Plots saved to 'perceptron_results.png'")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  NAND Gate Accuracy: {nand_accuracy:.1f}%")
print(f"  XOR Gate Accuracy:  {xor_accuracy:.1f}%")
print(f"\n  Key Observation:")
print(f"  - NAND is linearly separable → Perceptron CAN learn it.")
print(f"  - XOR is NOT linearly separable → Perceptron CANNOT learn it.")
print(f"  - This demonstrates the fundamental limitation of single-layer perceptrons,")
print(f"    as proven by Minsky & Papert (1969).")
print("=" * 60)
