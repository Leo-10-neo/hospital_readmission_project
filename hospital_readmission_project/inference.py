# -*- coding: utf-8 -*-
"""
inference.py
============
Simple script to demonstrate how to use the trained model to predict 
readmission risk for new patient data.
"""

import os, pickle, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

def predict_readmission(patient_data):
    """
    Predict readmission risk for a patient.
    Input: DataFrame with all required features.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Predict probability
    prob = model.predict_proba(patient_data)[:, 1]
    prediction = model.predict(patient_data)
    
    return prediction, prob

if __name__ == "__main__":
    # Example: Loading a few rows from test set as 'new data'
    TEST_DATA = "data/processed/X_test.csv"
    if os.path.exists(TEST_DATA):
        X_test = pd.read_csv(TEST_DATA).head(5)
        preds, probs = predict_readmission(X_test)
        
        print("\n" + "="*40)
        print("  HOSPITAL READMISSION RISK INFERENCE")
        print("="*40)
        for i, (p, prob) in enumerate(zip(preds, probs)):
            risk_level = "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.3 else "LOW"
            print(f"Patient {i+1}: Risk Prob: {prob:.4f} | Prediction: {p} | Level: {risk_level}")
        print("="*40 + "\n")
    else:
        print("Test data not found for demonstration.")
