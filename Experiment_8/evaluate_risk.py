"""
Experiment 8: Mini Project 2 - Driving Score Evaluation
=======================================================
Submission Date: 21.03.2026

Author: B.E. (AI & DS) VI Semester Student
"""
import torch
"""
Experiment 8: Mini Project 2 - Dataset Generator
================================================
Submission Date: 20.02.2026

Author: B.E. (AI & DS) VI Semester Student
"""
import numpy as np
import os
from models import RNNModel, TransformerModel

def calculate_driving_score(model, input_sequence):
    """
    Takes an input sequence (seq_len, 6) and returns a Driving Score (0-100).
    Higher score is better (safer).
    """
    model.eval()
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        x = torch.FloatTensor(input_sequence).unsqueeze(0)
        risk_prediction = model(x).item()
        
    # Scale: 0 -> 100 (Safe), 1 -> 0 (Risky)
    driving_score = 100 * (1.0 - risk_prediction)
    return driving_score

def interpret_score(score):
    if score >= 80: return "Excellent - Safe driving"
    elif score >= 60: return "Good - Minor erratic behaviors seen"
    elif score >= 40: return "Average - Frequent hard braking or sharp turns"
    else: return "Dangerous - Highly erratic driving behavior"

if __name__ == "__main__":
    # Load the best model (using LSTM as default if not trained)
    # Note: In real setup, you'd load the .pth file
    model = RNNModel(model_type='LSTM')
    
    # Try to load weights if exists
    weight_path = os.path.join(os.path.dirname(__file__), "best_driving_model.pth")
    if os.path.exists(weight_path):
        try:
            model.load_state_dict(torch.load(weight_path))
            print("Loaded trained model.")
        except:
            print("Using initialized model (weights not found/compatible).")
    
    # Simulate a "Safe" trip
    safe_data = np.random.normal(0, 0.2, (50, 6))
    safe_data[:, 2] += 9.8 # Gravity
    safe_score = calculate_driving_score(model, safe_data)
    
    # Simulate a "Risky" trip
    risky_data = np.random.normal(0, 0.2, (50, 6))
    risky_data[:, 2] += 9.8
    risky_data[10:20, 1] += 10 # Sharp braking
    risky_score = calculate_driving_score(model, risky_data)
    
    print("\n" + "="*40)
    print("      DRIVING SCORE EVALUATION")
    print("="*40)
    print(f"Safe Trip Score: {safe_score:.2f}/100 -> {interpret_score(safe_score)}")
    print(f"Risky Trip Score: {risky_score:.2f}/100 -> {interpret_score(risky_score)}")
    print("="*40)
