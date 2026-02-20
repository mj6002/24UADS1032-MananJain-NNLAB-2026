"""
Experiment 2: Multi-Layer Perceptron (MLP) for XOR
===================================================
Objective: Implement MLP with one hidden layer using numpy in Python.
           Demonstrate that it can learn the XOR Boolean function.

Author: B.E. (AI & DS) VI Semester Student
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Activation Functions
# ============================================================
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    return x * (1 - x)

# ============================================================
# Multi-Layer Perceptron Class
# ============================================================
class MLP:
    """Multi-Layer Perceptron with one hidden layer."""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        np.random.seed(42)
        # Xavier initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.bias_output = np.zeros((1, output_size))
        self.lr = learning_rate
        self.loss_history = []
    
    def forward(self, X):
        """Forward pass through the network."""
        # Input to Hidden Layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Hidden to Output Layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = sigmoid(self.output_input)
        
        return self.predicted_output
    
    def backward(self, X, y, output):
        """Backpropagation to update weights."""
        # Output layer error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        # Hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.lr
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.lr
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.lr
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr
    
    def train(self, X, y, epochs=10000):
        """Train the MLP network."""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss (MSE)
            loss = np.mean((y - output) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            if (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch + 1:>5}/{epochs}, Loss: {loss:.6f}")
        
        return self.loss_history
    
    def predict(self, X):
        """Predict output (thresholded)."""
        raw_output = self.forward(X)
        return (raw_output > 0.5).astype(int)


# ============================================================
# Dataset: XOR Truth Table
# ============================================================
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# ============================================================
# Training
# ============================================================
print("\n" + "=" * 60)
print("  EXPERIMENT 2: MULTI-LAYER PERCEPTRON FOR XOR")
print("=" * 60)

print("\n[1] Network Architecture:")
print("  Input Layer:  2 neurons")
print("  Hidden Layer: 4 neurons (Sigmoid activation)")
print("  Output Layer: 1 neuron  (Sigmoid activation)")
print("  Learning Rate: 0.5")
print("  Epochs: 10000")

print("\n[2] Training MLP...")
mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
losses = mlp.train(X, y, epochs=10000)

# ============================================================
# Evaluation
# ============================================================
print(f"\n[3] Evaluation Results:")
print(f"  {'='*50}")
predictions_raw = mlp.forward(X)
predictions = mlp.predict(X)

print(f"  {'Input1':<8} {'Input2':<8} {'Expected':<10} {'Raw Output':<12} {'Predicted':<10} {'Correct':<8}")
print(f"  {'-'*56}")

correct = 0
for i in range(len(X)):
    pred = predictions[i][0]
    expected = y[i][0]
    is_correct = "✓" if pred == expected else "✗"
    if pred == expected:
        correct += 1
    print(f"  {X[i][0]:<8} {X[i][1]:<8} {expected:<10} {predictions_raw[i][0]:<12.6f} {pred:<10} {is_correct:<8}")

accuracy = (correct / len(X)) * 100
print(f"\n  Accuracy: {accuracy:.1f}% ({correct}/{len(X)})")

# ============================================================
# Network Parameters
# ============================================================
print(f"\n[4] Learned Network Parameters:")
print(f"  Input→Hidden Weights:\n  {mlp.weights_input_hidden}")
print(f"  Hidden Biases: {mlp.bias_hidden}")
print(f"  Hidden→Output Weights:\n  {mlp.weights_hidden_output}")
print(f"  Output Bias: {mlp.bias_output}")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Experiment 2: Multi-Layer Perceptron for XOR', 
             fontsize=16, fontweight='bold')

# 1. Loss Curve
axes[0].plot(losses, color='crimson', linewidth=1.5)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Mean Squared Error', fontsize=12)
axes[0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 2. Decision Boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = mlp.forward(grid_points).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.8)
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
colors = ['red' if label == 0 else 'green' for label in y.flatten()]
axes[1].scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2, zorder=5)
for i, (xi, yi) in enumerate(zip(X, y)):
    axes[1].annotate(f'({xi[0]},{xi[1]})→{yi[0]}', (xi[0], xi[1]),
                     textcoords="offset points", xytext=(12, 12), fontsize=9, fontweight='bold')
axes[1].set_xlabel('Input 1', fontsize=12)
axes[1].set_ylabel('Input 2', fontsize=12)
axes[1].set_title('Decision Boundary', fontsize=14, fontweight='bold')
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylim(-0.5, 1.5)

# 3. Network Architecture Diagram
axes[2].set_xlim(0, 10)
axes[2].set_ylim(0, 10)
axes[2].set_aspect('equal')
axes[2].axis('off')
axes[2].set_title('Network Architecture', fontsize=14, fontweight='bold')

# Draw neurons
input_y = [6.5, 3.5]
hidden_y = [8, 6, 4, 2]
output_y = [5]

for iy in input_y:
    axes[2].add_patch(plt.Circle((2, iy), 0.4, color='#3498db', ec='black', lw=2, zorder=5))
for hy in hidden_y:
    axes[2].add_patch(plt.Circle((5, hy), 0.4, color='#e74c3c', ec='black', lw=2, zorder=5))
for oy in output_y:
    axes[2].add_patch(plt.Circle((8, oy), 0.4, color='#2ecc71', ec='black', lw=2, zorder=5))

# Draw connections
for iy in input_y:
    for hy in hidden_y:
        axes[2].plot([2.4, 4.6], [iy, hy], 'gray', alpha=0.4, lw=1)
for hy in hidden_y:
    for oy in output_y:
        axes[2].plot([5.4, 7.6], [hy, oy], 'gray', alpha=0.4, lw=1)

axes[2].text(2, 1, 'Input\n(2 neurons)', ha='center', fontsize=10, fontweight='bold')
axes[2].text(5, 1, 'Hidden\n(4 neurons)', ha='center', fontsize=10, fontweight='bold')
axes[2].text(8, 1, 'Output\n(1 neuron)', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'mlp_xor_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[✓] Plots saved to 'mlp_xor_results.png'")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  XOR Function learned successfully: {'YES' if accuracy == 100 else 'NO'}")
print(f"  Final Loss: {losses[-1]:.6f}")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"\n  Key Observation:")
print(f"  - A single perceptron CANNOT learn XOR (not linearly separable)")
print(f"  - An MLP with one hidden layer CAN learn XOR")
print(f"  - The hidden layer creates a new feature space where XOR becomes")
print(f"    linearly separable")
print(f"{'='*60}")
