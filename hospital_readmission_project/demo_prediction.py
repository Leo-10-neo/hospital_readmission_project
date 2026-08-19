# -*- coding: utf-8 -*-
"""
demo_prediction.py
==================
STEP 9 - Live Prediction Demo for Hospital Readmission Risk.

This script demonstrates how to predict readmission risk for individual 
patients using the trained Random Forest model.
"""

import os, pickle, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
TEST_DATA   = os.path.join(BASE_DIR, "data", "processed", "X_test.csv")

# Load Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    best_model = pickle.load(f)

# ---------------------------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------------------------
def predict_patient_risk(patient_data_dict):
    """
    Takes a dictionary of patient features and returns risk assessment.
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([patient_data_dict])
    
    # Get Probability
    prob = best_model.predict_proba(df)[0][1]
    risk_score = prob * 100
    
    # Determine Category & Recommendation
    if risk_score > 35:
        category = "CRITICAL / HIGH RISK"
        recommendation = "IMMEDIATE Care Coordination, Pharmacist Review, and Home Visit within 48h."
        color = "\033[91m" # Red
    elif risk_score > 25:
        category = "MODERATE RISK"
        recommendation = "Scheduled follow-up call within 72h and PCP appointment within 7 days."
        color = "\033[93m" # Yellow
    else:
        category = "LOW RISK"
        recommendation = "Standard discharge instructions and routine follow-up."
        color = "\033[92m" # Green
        
    reset = "\033[0m"
    
    return {
        "score": risk_score,
        "category": category,
        "recommendation": recommendation,
        "color": color,
        "reset": reset
    }

# ---------------------------------------------------------------------------
# LIVE DEMO EXAMPLES
# ---------------------------------------------------------------------------
def run_demo():
    print("\n" + "="*70)
    print("  HOSPITAL READMISSION RISK SCORER — LIVE DEMO")
    print("  Model: Random Forest Classifier")
    print("="*70)
    
    # Load 3 example patients from the test set with different profiles
    if os.path.exists(TEST_DATA):
        X_test = pd.read_csv(TEST_DATA)
        
        # We'll pick 3 specific rows that likely represent different risk levels
        # based on our knowledge of the features (Comorbidity and LoS)
        
        # 1. HIGH RISK (High comorbidity, long stay)
        high_risk_patient = X_test.sort_values(['comorbidity_index', 'length_of_stay_log'], ascending=False).iloc[0].to_dict()
        
        # 2. MEDIUM RISK (Mid range)
        med_risk_patient = X_test.iloc[len(X_test)//2].to_dict()
        
        # 3. LOW RISK (Low comorbidity, short stay)
        low_risk_patient = X_test.sort_values(['comorbidity_index', 'length_of_stay_log'], ascending=True).iloc[0].to_dict()
        
        examples = [
            ("Patient A (Complex Case)", high_risk_patient),
            ("Patient B (Standard Case)", med_risk_patient),
            ("Patient C (Low Complexity)", low_risk_patient)
        ]
        
        for name, data in examples:
            result = predict_patient_risk(data)
            
            print(f"\n>>> {name}")
            print(f"    Clinical Profile : Comorbidity Index={data['comorbidity_index']:.2f}, LoS={np.exp(data['length_of_stay_log']):.1f} days")
            print(f"    Risk Score       : {result['color']}{result['score']:.1f}%{result['reset']}")
            print(f"    Category         : {result['color']}{result['category']}{result['reset']}")
            print(f"    Recommendation   : {result['recommendation']}")
            print("-" * 40)
            
    else:
        print("Demo Error: Could not find test data to extract examples.")

    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_demo()
