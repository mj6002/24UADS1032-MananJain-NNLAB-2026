"""
Experiment 8: Mini Project 2 - Dataset Generator
================================================
Submission Date: 26.02.2026

Author: B.E. (AI & DS) VI Semester Student
"""
import numpy as np
import pandas as pd
import os

def generate_bike_sensor_data(num_samples=1000, seq_len=50):
    """
    Generates synthetic accelerometer and gyroscope data for bike riding.
    Features: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    data = []
    labels = []
    
    for i in range(num_samples):
        # Determine if this sample is "risky"
        is_risky = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Base noise
        acc = np.random.normal(0, 0.5, (seq_len, 3))
        gyro = np.random.normal(0, 0.1, (seq_len, 3))
        
        # Add gravity component (z-axis usually near 9.8)
        acc[:, 2] += 9.8 
        
        if is_risky:
            # Add risky events: sudden braking or sharp turns
            event_type = np.random.randint(0, 3)
            if event_type == 0: # Sudden Braking
                acc[seq_len//2:seq_len//2+5, 1] += np.random.uniform(5, 15)
                score = np.random.uniform(0.6, 1.0)
            elif event_type == 1: # Sharp Turn
                gyro[seq_len//2:seq_len//2+5, 2] += np.random.uniform(2, 5)
                score = np.random.uniform(0.5, 0.9)
            else: # High vibration (overspeeding/bad road)
                acc += np.random.normal(0, 2.0, (seq_len, 3))
                score = np.random.uniform(0.4, 0.8)
        else:
            # Calm riding
            score = np.random.uniform(0.0, 0.3)
            
        sample = np.hstack([acc, gyro]) # (seq_len, 6)
        data.append(sample)
        labels.append(score)
        
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    X, y = generate_bike_sensor_data()
    output_path = os.path.join(os.path.dirname(__file__), "synthetic_bike_data.npz")
    np.savez(output_path, X=X, y=y)
    print(f"Generated synthetic dataset: {X.shape} samples, saved to {output_path}")
