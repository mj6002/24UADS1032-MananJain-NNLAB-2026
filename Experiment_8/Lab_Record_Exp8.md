# Experiment 8: Mini Project 2 - Bike Driving Behavior Analysis

## Submission Timeline
| Milestone | Submission Date |
| :--- | :--- |
| **Dataset Submission** | 26.02.2026 |
| **Model Submission** | 21.03.2026 |
| **Bonus Submission** | 30.03.2026 |
| **Poster Submission** | 04.04.2026 |

## 1. Objective
The objective of this project is to analyze motion sensor data (Accelerometer and Gyroscope) captured during bike riding to evaluate driving behavior. The project involves:
- Processing raw sensor data from multiple riders and vehicles.
- Building a model to generate a **Driving Score/Risk Score**.
- Comparing the performance of **RNNs (LSTM/GRU)** vs. **Transformers** for this sequence analysis task (Bonus).

## 2. Methodology

### 2.1 Dataset Characteristics
The dataset simulates bike riding motion characteristics with variations in:
- **Riders & Vehicles**: Modeled as different noise profiles and event magnitudes.
- **Speeds**: High-speed riding includes higher vibration noise.
- **Phone Placements**: Modeled as rotations in the sensor coordinate system.
- **Sensor Inputs**: 3-axis Accelerometer ($m/s^2$) and 3-axis Gyroscope ($rad/s$).

### 2.2 Model Architectures
- **RNN (LSTM)**: A 2-layer LSTM architecture designed to capture temporal dependencies in motion sequences. It uses the hidden state of the final time step to predict the risk score.
- **Transformer**: A self-attention-based encoder that processes the entire sequence in parallel. It uses global average pooling over time steps to produce a fixed-size representation for scoring.

### 2.3 Risk Scoring Logic
The model outputs a risk value between 0 and 1. This is converted to a **Driving Score** using:
$$\text{Driving Score} = 100 \times (1.0 - \text{Risk Score})$$
- **High Score (80-100)**: Safe driving behavior.
- **Low Score (<40)**: Risky/Erratic behavior (sudden brakes, sharp turns).

## 3. Implementation Details
The project is implemented using PyTorch and consists of the following modules:
1. `models.py`: Architecture definitions for RNN and Transformer.
2. `dataset_generator.py`: Synthetic data generation logic for bike rides.
3. `train_and_compare.py`: Training pipeline and performance comparison logic.
4. `evaluate_risk.py`: Interpretation layer for generating human-readable driving scores.

## 4. Performance Comparison (Bonus)
The following metrics were evaluated to compare LSTM and Transformer performance on the synthetic dataset:

| Model | Final Val Loss (MSE) | Training Time (s) | Evaluation |
| :--- | :--- | :--- | :--- |
| **LSTM** | 0.0085 | 8.54 | Faster training and slightly better accuracy for this sequence length. |
| **Transformer** | 0.0095 | 26.18 | Slower on CPU; performance likely to improve with larger datasets and seq lengths. |

The **LSTM** model was selected as the best performing model for this specific task configuration.

## 5. Driving Score Evaluation
Testing the model on sample trips yielded the following scores:
- **Safe Trip Score**: 85.53/100 -> *Excellent - Safe driving*
- **Risky Trip Score**: 21.90/100 -> *Dangerous - Highly erratic driving behavior*

The model effectively identifies erratic driving patterns like sudden braking and sharp turns, mapping them correctly to a risk-adjusted score.
