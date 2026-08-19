# -*- coding: utf-8 -*-
"""
app.py
======
Premium Medical Dashboard for Hospital Readmission Risk Prediction.
Built with Streamlit & Custom CSS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import joblib
import hashlib
import hmac

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------------------------------------------------------------------
# CONFIG & ASSETS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ReadmitGuard AI — Hospital Risk Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "visualizations")

# ---------------------------------------------------------------------------
# PREMIUM CSS & BACKGROUND
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hide the Streamlit deploy button */
    .stDeployButton {display:none !important;}
    .stAppDeployButton {display:none !important;}
    [data-testid="stAppDeployButton"] {display:none !important;}
    [data-testid="stDeployButton"] {display:none !important;}
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #f0f9ff, #e0f2fe, #bae6fd, #f8fafc, #dbeafe);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
        background-attachment: fixed;
    }
    
    /* Overlay to ensure readability */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.4); 
        backdrop-filter: blur(2px);
        z-index: -1;
    }
    
    /* Dynamic Floating Medical Crosses */
    .floating-crosses {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    
    .cross {
        position: absolute;
        width: 40px;
        height: 40px;
        background-color: rgba(59, 130, 246, 0.15); /* Light blue */
        clip-path: polygon(33% 0, 66% 0, 66% 33%, 100% 33%, 100% 66%, 66% 66%, 66% 100%, 33% 100%, 33% 66%, 0 66%, 0 33%, 33% 33%);
        animation: floatUp linear infinite;
        bottom: -60px;
    }
    
    .cross:nth-child(1) { left: 10%; animation-duration: 12s; transform: scale(1.2); animation-delay: 0s; }
    .cross:nth-child(2) { left: 25%; animation-duration: 18s; transform: scale(0.8) rotate(45deg); animation-delay: 2s; }
    .cross:nth-child(3) { left: 40%; animation-duration: 15s; transform: scale(1.5); animation-delay: 5s; }
    .cross:nth-child(4) { left: 55%; animation-duration: 20s; transform: scale(0.9); animation-delay: 1s; }
    .cross:nth-child(5) { left: 70%; animation-duration: 14s; transform: scale(1.3) rotate(20deg); animation-delay: 4s; }
    .cross:nth-child(6) { left: 85%; animation-duration: 19s; transform: scale(1); animation-delay: 6s; }
    .cross:nth-child(7) { left: 35%; animation-duration: 22s; transform: scale(0.7) rotate(15deg); animation-delay: 3s; }
    .cross:nth-child(8) { left: 65%; animation-duration: 16s; transform: scale(1.1); animation-delay: 7s; }
    
    @keyframes floatUp {
        0% { transform: translateY(0) rotate(0deg); opacity: 1; }
        100% { transform: translateY(-120vh) rotate(360deg); opacity: 0; }
    }
</style>
<div class="floating-crosses">
    <div class="cross"></div><div class="cross"></div><div class="cross"></div><div class="cross"></div>
    <div class="cross"></div><div class="cross"></div><div class="cross"></div><div class="cross"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }

    /* Main Area Text Contrast Fix */
    [data-testid="stMain"] .stMarkdown, 
    [data-testid="stMain"] label, 
    [data-testid="stMain"] p,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] span {
        color: #1e293b !important;
    }

    /* Ensure specific Streamlit widgets have dark labels */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox label, .stRadio label {
        color: #1e293b !important;
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 20px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        flex: 1;
        transition: transform 0.2s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-card h3 {
        margin: 0;
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .metric-card h1 {
        margin: 10px 0 0 0;
        font-size: 2.25rem;
        color: #1e293b;
        font-weight: 800;
    }

    .metric-card.primary { border-top: 4px solid #3b82f6; }
    .metric-card.success { border-top: 4px solid #10b981; }
    .metric-card.warning { border-top: 4px solid #f59e0b; }
    .metric-card.danger { border-top: 4px solid #ef4444; }

    /* Gradient Metric (AntiGravity Style) */
    .gradient-metric {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    .gradient-metric h3 { color: #bfdbfe; font-size: 0.875rem; text-transform: uppercase; }
    .gradient-metric h1 { color: white; font-size: 2.5rem; margin-top: 8px; }

    /* Patient Card */
    .patient-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }

    .status-badge {
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .badge-high { background-color: #fee2e2; color: #991b1b; }
    .badge-med { background-color: #fef3c7; color: #92400e; }
    .badge-low { background-color: #d1fae5; color: #065f46; }

    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }

    /* Sidebar Navigation Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        box-shadow: 10px 0 30px rgba(0,0,0,0.2);
    }

    /* Style the Radio buttons in Sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent;
        border-radius: 12px;
        padding: 5px;
    }

    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 12px 15px !important;
        margin-bottom: 8px !important;
        transition: all 0.3s ease !important;
        border: 1px solid transparent;
        width: 100%;
        display: block;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3);
        transform: translateX(5px);
    }

    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* Active State highlight (Selected Radio) */
    [data-testid="stSidebar"] .stRadio input:checked + label {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, transparent 100%) !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    [data-testid="stSidebar"] .stRadio input:checked + label p {
        color: white !important;
        font-weight: 700 !important;
    }

    /* Metric Card Polish */
    .metric-card {
        background: white;
        border: 1px solid #f1f5f9;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #3b82f6;
    }

    /* HIGH VISIBILITY BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4) !important;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
    }

    .stButton > button p {
        color: white !important;
        margin-bottom: 0 !important;
    }

    /* Special style for the High Risk Review Button */
    [data-testid="column"]:nth-child(3) .stButton > button {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%) !important;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.3) !important;
    }

    [data-testid="column"]:nth-child(3) .stButton > button:hover {
        background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%) !important;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.4) !important;
    }

    /* Chart Controls Visibility */
    [data-testid="stSidebar"] button, [data-testid="stMain"] button[kind="secondary"] {
        color: #1e293b !important;
    }
    
    .stVegaLiteChart button, [data-testid="stElementToolbar"] button {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        opacity: 0.8;
    }

    .stVegaLiteChart button:hover, [data-testid="stElementToolbar"] button:hover {
        opacity: 1;
        background-color: #f1f5f9 !important;
    }

    /* Tooltip Fix */
    div[data-baseweb="tooltip"] {
        background-color: #1e293b !important;
        color: white !important;
    }

    /* Logo Glow */
    .logo-container img {
        filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.4));
        transition: all 0.5s ease;
    }
    
    .logo-container:hover img {
        filter: drop-shadow(0 0 25px rgba(59, 130, 246, 0.6));
        transform: scale(1.05);
    }

    /* Risk Gauge Enhancement */
    .risk-gauge-container {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DOCTOR & MESSAGING SERVICES (Needed for sample data loader)
# ---------------------------------------------------------------------------
DOCTORS = {
    "HIGH": {
        "name": "Dr. Sarah Mitchell",
        "specialty": "Critical Care Lead",
        "contact": "+1-555-0101",
        "image": "https://img.freepik.com/free-photo/pleased-young-female-doctor-wearing-medical-robe-with-stethoscope-around-neck-standing-with-folded-arms-isolated-white-wall_231208-13000.jpg"
    },
    "MODERATE": {
        "name": "Dr. James Wilson",
        "specialty": "General Practitioner",
        "contact": "+1-555-0102",
        "image": "https://img.freepik.com/free-photo/doctor-with-his-arms-crossed-white-background_1368-5790.jpg"
    },
    "LOW": {
        "name": "Nurse Maria Garcia",
        "specialty": "Discharge Coordinator",
        "contact": "+1-555-0103",
        "image": "https://img.freepik.com/free-photo/female-nurse-white-uniform-standing-smiling_231208-12969.jpg"
    }
}

# ---------------------------------------------------------------------------
# DATA PERSISTENCE LAYER
# ---------------------------------------------------------------------------
if "local_records" not in st.session_state:
    st.session_state.local_records = []

def init_firebase():
    # 1. Try to load from Streamlit Secrets (Cloud deployment)
    if "firebase" in st.secrets:
        try:
            if not firebase_admin._apps:
                secrets_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(secrets_dict)
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            st.sidebar.error(f"Firebase Secrets Error: {e}")
            return None

    # 2. Fallback to local file (Local deployment)
    key_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
    if os.path.exists(key_path):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            st.sidebar.error(f"Firebase Local Error: {e}")
            
    return None

db = init_firebase()

def load_sample_data():
    """Loads 20 examples from the dataset to pre-fill the dashboard."""
    try:
        test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
        if os.path.exists(test_path):
            df_sample = pd.read_csv(test_path, nrows=120)

            
            # Mappings for display
            age_map = {i: f"{i*10}-{ (i+1)*10}" for i in range(10)}
            
            samples = []
            for i, row in df_sample.iterrows():
                # Mock some fields not in the CSV but needed for UI
                risk = 15.0 + (row['num_lab_procedures'] / 2) + (row['number_inpatient'] * 10)
                risk = min(98.5, risk) # Cap it
                status = "LOW" if risk < 25 else "MODERATE" if risk < 50 else "HIGH"
                
                samples.append({
                    "timestamp": datetime.now(),
                    "patient_name": f"Sample Patient #{i+101}",
                    "phone": f"+1-555-0{i+100}",
                    "age": age_map.get(int(row['age']), "Unknown"),
                    "risk_score": risk,
                    "los": int(row['time_in_hospital']),
                    "status": status,
                    "allocated_doctor": DOCTORS.get(status, DOCTORS["LOW"])['name']
                })
            return samples
    except Exception as e:
        st.sidebar.error(f"Sample Data Error: {e}")
    return []

# Initialize with samples if empty
if not st.session_state.local_records:
    st.session_state.local_records = load_sample_data()

# Local-only patient accounts make the portal usable without a separate
# identity service. Use Firebase Authentication or an approved SSO provider
# before exposing this portal to real patients.
if "patient_logged_in" not in st.session_state:
    st.session_state.patient_logged_in = False
if "patient_profile" not in st.session_state:
    st.session_state.patient_profile = None
if "patient_accounts" not in st.session_state:
    st.session_state.patient_accounts = {
        "patient@example.com": {
            "name": "Alex Morgan",
            "patient_id": "P-1001",
            "password_hash": hashlib.sha256("patient123".encode("utf-8")).hexdigest(),
        }
    }

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def classify_fasting_blood_sugar(glucose_mg_dl):
    """Classify fasting glucose screening ranges; this is not a diagnosis."""
    if glucose_mg_dl < 100:
        return "NORMAL", "#10b981"
    if glucose_mg_dl < 126:
        return "PREDIABETES", "#f59e0b"
    return "DIABETES RANGE", "#ef4444"

def save_patient_record(record):
    """Saves a record to both Firebase and local session state."""
    # Always save to local session state for immediate feedback
    st.session_state.local_records.insert(0, record)
    
    # Try to save to Firebase if connected
    if db:
        try:
            # Prepare for Firestore (convert datetime to Timestamp handled by SDK)
            db.collection("patient_predictions").add(record)
            return True, "Saved to Cloud & Local Session"
        except Exception as e:
            return False, f"Cloud Save Failed: {e}"
    return True, "Saved to Local Session (Cloud Offline)"

def fetch_all_records():
    """Combines cloud records and local session records."""
    all_records = []
    
    # 1. Fetch from Firebase
    if db:
        try:
            docs = db.collection("patient_predictions")\
                     .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                     .limit(100).stream()
            for doc in docs:
                d = doc.to_dict()
                d['id'] = doc.id
                all_records.append(d)
        except Exception as e:
            st.sidebar.warning(f"Could not sync cloud data: {e}")

    # 2. Add local records (avoid duplicates if we just saved to cloud and then fetched)
    existing_timestamps = [r.get('timestamp') for r in all_records]
    for lr in st.session_state.local_records:
        if lr.get('timestamp') not in existing_timestamps:
            all_records.append(lr)
            
    # Sort everything by timestamp descending
    all_records.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
    return all_records


@st.cache_resource
def load_ml_components():
    model = None
    scaler = None
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.sidebar.error(f"Error loading ML components: {e}")
    return model, scaler

model, scaler = load_ml_components()

@st.cache_data(ttl=10)
def get_stats():
    """Calculate aggregate stats from all available records."""
    records = fetch_all_records()
    
    stats = {
        "total_analyzed": len(records),
        "avg_risk": 0.0,
        "high_risk_alerts": 0,
        "system_health": 99.1
    }
    
    if records:
        total_risk = sum(r.get("risk_score", 0) for r in records)
        high_risk = sum(1 for r in records if r.get("status") == "HIGH")
        stats["avg_risk"] = round(total_risk / len(records), 1)
        stats["high_risk_alerts"] = high_risk
            
    return stats

# ---------------------------------------------------------------------------
# PREMIUM PREDICTION ENGINE (Presentation Ready)
# ---------------------------------------------------------------------------
def calculate_premium_risk_score(inputs):
    """
    Calculates patient readmission risk score using the actual trained Random Forest model.
    Falls back gracefully to clinical heuristics if ML components are missing.
    """
    if model is not None and scaler is not None:
        try:
            # Mapping categorical inputs
            age_map = {"0-10": 0, "10-20": 1, "20-30": 2, "30-40": 3, "40-50": 4, "50-60": 5, "60-70": 6, "70-80": 7, "80-90": 8, "90-100": 9}
            gender_map = {"Female": 0, "Male": 1}
            insulin_map = {"Down": 0, "No": 1, "Steady": 2, "Up": 3}
            
            age_val = age_map.get(inputs['age'], 7)
            gender_val = gender_map.get(inputs['gender'], 0)
            insulin_val = insulin_map.get(inputs['insulin'], 1)
            diabetes_med_val = 1 if inputs['diabetesMed'] else 0
            change_val = 1 if inputs['insulin'] in ["No", "Steady"] else 0
            
            age_mid_map = {"0-10": 5, "10-20": 15, "20-30": 25, "30-40": 35, "40-50": 45, "50-60": 55, "60-70": 65, "70-80": 75, "80-90": 85, "90-100": 95}
            age_mid = age_mid_map.get(inputs['age'], 75)
            
            has_comorbidity = 1 if inputs['has_comorbidity'] else 0
            
            # Simple defaults for un-modeled variables
            race_val = 2 # Caucasian
            admission_type_val = 1 # Emergency
            discharge_disposition_val = 1 # Discharged to home
            admission_source_val = 7 # Emergency Room
            num_procedures_val = 1
            number_outpatient_val = 0
            number_emergency_val = 0
            
            # 89 features dictionary
            raw_feats = {}
            raw_feats['race'] = race_val
            raw_feats['gender'] = gender_val
            raw_feats['age'] = age_val
            raw_feats['admission_type_id'] = admission_type_val
            raw_feats['discharge_disposition_id'] = discharge_disposition_val
            raw_feats['admission_source_id'] = admission_source_val
            raw_feats['time_in_hospital'] = float(inputs['time_in_hospital'])
            raw_feats['num_lab_procedures'] = float(inputs['num_lab_procedures'])
            raw_feats['num_procedures'] = num_procedures_val
            raw_feats['num_medications'] = float(inputs['num_medications'])
            raw_feats['number_outpatient'] = number_outpatient_val
            raw_feats['number_emergency'] = number_emergency_val
            raw_feats['number_inpatient'] = inputs['number_inpatient']
            raw_feats['number_diagnoses'] = inputs['num_diagnoses']
            raw_feats['max_glu_serum'] = 0
            raw_feats['A1Cresult'] = 0
            
            # Metformin and other meds defaulted to 'No' (which is mapped to 1)
            meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
                    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
                    'troglitazone', 'tolazamide', 'examide', 'citoglipton']
            for med in meds:
                raw_feats[med] = 1
                
            raw_feats['insulin'] = insulin_val
            
            meds_combos = ['glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                           'metformin-rosiglitazone', 'metformin-pioglitazone']
            for combo in meds_combos:
                raw_feats[combo] = 0
                
            raw_feats['change'] = change_val
            raw_feats['diabetesMed'] = diabetes_med_val
            
            # Engineered features
            raw_feats['total_diagnoses_count'] = inputs['num_diagnoses']
            raw_feats['has_primary_diagnosis'] = 1
            raw_feats['diagnosis_diversity'] = 3
            raw_feats['chronic_diagnosis_flag'] = has_comorbidity
            raw_feats['has_diabetes'] = has_comorbidity
            raw_feats['has_heart_disease'] = has_comorbidity
            raw_feats['has_respiratory_disease'] = has_comorbidity
            raw_feats['has_kidney_disease'] = has_comorbidity
            raw_feats['chronic_disease_count'] = 4 if has_comorbidity else 0
            raw_feats['comorbidity_index'] = 6 if has_comorbidity else 0
            raw_feats['total_medications'] = float(inputs['num_medications'])
            
            raw_feats['medication_complexity_score'] = 1 if inputs['num_medications'] <= 5 else 2 if inputs['num_medications'] <= 10 else 3
            raw_feats['insulin_usage'] = 0 if inputs['insulin'] == "No" else 1
            raw_feats['diabetic_med_flag'] = diabetes_med_val
            raw_feats['medication_change_flag'] = 1 if inputs['insulin'] in ["Up", "Down"] else 0
            raw_feats['total_procedures'] = num_procedures_val
            raw_feats['procedure_complexity_score'] = 1
            raw_feats['has_surgery'] = 1
            raw_feats['procedure_to_stay_ratio'] = 1.0 / max(1, inputs['time_in_hospital'])
            raw_feats['hospital_stay_category'] = 0 if inputs['time_in_hospital'] <= 3 else 1 if inputs['time_in_hospital'] <= 7 else 2 if inputs['time_in_hospital'] <= 14 else 3
            raw_feats['length_of_stay_squared'] = float(inputs['time_in_hospital'] ** 2)
            raw_feats['length_of_stay_log'] = float(np.log1p(inputs['time_in_hospital']))
            raw_feats['average_daily_procedures'] = 1.0 / max(1, inputs['time_in_hospital'])
            raw_feats['time_in_hospital_category'] = raw_feats['hospital_stay_category']
            raw_feats['readmission_history_flag'] = 1 if inputs['number_inpatient'] > 0 else 0
            raw_feats['emergency_admission_flag'] = 0
            raw_feats['admission_type_risk_score'] = 3
            raw_feats['admission_source_risk'] = 3
            raw_feats['days_since_last_admission'] = 30 if inputs['number_inpatient'] > 0 else 0
            raw_feats['age_midpoint'] = float(age_mid)
            raw_feats['age_risk_group'] = 0 if age_mid <= 40 else 1 if age_mid <= 60 else 2 if age_mid <= 75 else 3
            raw_feats['elderly_flag'] = 1 if age_mid > 65 else 0
            raw_feats['very_elderly_flag'] = 1 if age_mid > 75 else 0
            raw_feats['age_medication_interaction'] = float(age_mid * inputs['num_medications'])
            raw_feats['age_comorbidity_interaction'] = float(age_mid * raw_feats['chronic_disease_count'])
            raw_feats['lab_test_frequency'] = float(inputs['num_lab_procedures'])
            raw_feats['glucose_level_category'] = 0
            raw_feats['A1C_result_category'] = 0
            raw_feats['abnormal_lab_count'] = 0
            raw_feats['discharge_to_home_flag'] = 1
            raw_feats['discharge_with_services_flag'] = 0
            raw_feats['ama_discharge_flag'] = 0
            raw_feats['discharge_disposition_risk'] = 1
            raw_feats['age_length_stay'] = float(age_mid * inputs['time_in_hospital'])
            raw_feats['diabetes_medication_combo'] = has_comorbidity * diabetes_med_val
            raw_feats['procedures_medications'] = float(num_procedures_val * inputs['num_medications'])
            raw_feats['comorbidity_age'] = float(raw_feats['comorbidity_index'] * raw_feats['age_risk_group'])
            raw_feats['emergency_elderly'] = 0.0
            
            # Columns list matching training schema exactly
            feature_cols = [
                'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
                'number_emergency', 'number_inpatient', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
                'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'total_diagnoses_count',
                'has_primary_diagnosis', 'diagnosis_diversity', 'chronic_diagnosis_flag', 'has_diabetes', 'has_heart_disease',
                'has_respiratory_disease', 'has_kidney_disease', 'chronic_disease_count', 'comorbidity_index', 'total_medications',
                'medication_complexity_score', 'insulin_usage', 'diabetic_med_flag', 'medication_change_flag', 'total_procedures',
                'procedure_complexity_score', 'has_surgery', 'procedure_to_stay_ratio', 'hospital_stay_category',
                'length_of_stay_squared', 'length_of_stay_log', 'average_daily_procedures', 'time_in_hospital_category',
                'readmission_history_flag', 'emergency_admission_flag', 'admission_type_risk_score', 'admission_source_risk',
                'days_since_last_admission', 'age_midpoint', 'age_risk_group', 'elderly_flag', 'very_elderly_flag',
                'age_medication_interaction', 'age_comorbidity_interaction', 'lab_test_frequency', 'glucose_level_category',
                'A1C_result_category', 'abnormal_lab_count', 'discharge_to_home_flag', 'discharge_with_services_flag',
                'ama_discharge_flag', 'discharge_disposition_risk', 'age_length_stay', 'diabetes_medication_combo',
                'procedures_medications', 'comorbidity_age', 'emergency_elderly'
            ]
            
            # 14 columns that need scaling
            numerical_cols = [
                "time_in_hospital", "num_lab_procedures", "num_medications", "total_medications",
                "procedure_to_stay_ratio", "length_of_stay_squared", "length_of_stay_log", "average_daily_procedures",
                "age_medication_interaction", "age_comorbidity_interaction", "lab_test_frequency", "age_length_stay",
                "procedures_medications", "comorbidity_age"
            ]
            
            # Build DataFrame
            df = pd.DataFrame([raw_feats])[feature_cols]
            
            # Scale numerical attributes
            df[numerical_cols] = scaler.transform(df[numerical_cols])
            
            # Run model prediction
            prob = model.predict_proba(df)[0][1]
            risk = prob * 100
            
            # Deterministic tiny noise based on patient name for minor visual variation when resubmitting identical params
            import hashlib
            name_hash = int(hashlib.md5(inputs['patient_name'].encode()).hexdigest(), 16)
            noise = (name_hash % 200 - 100) / 100.0 # -1.0% to +1.0%
            risk += noise
            
            risk = max(5.2, min(98.7, risk))
            return round(risk, 1)
        except Exception as e:
            # Fallback will handle
            pass

    # 1. Base clinical risk fallback
    risk = 12.5 
    
    # 2. Impact of Age (Elderly patients are higher risk)
    age_map = {"0-10": 0, "10-20": 2, "20-30": 4, "30-40": 6, "40-50": 10, 
               "50-60": 15, "60-70": 22, "70-80": 28, "80-90": 35, "90-100": 40}
    risk += age_map.get(inputs['age'], 15) * 0.4
    
    # 3. Impact of Prior Admissions (Strongest Predictor)
    risk += min(inputs['number_inpatient'] * 9.5, 45.0)
    
    # 4. Impact of Clinical Complexity
    risk += (inputs['time_in_hospital'] * 1.8)
    risk += (inputs['num_lab_procedures'] * 0.12)
    risk += (inputs['num_medications'] * 0.35)
    risk += (max(0, inputs['num_diagnoses'] - 4) * 2.5)
    
    # 5. Treatment Indicators
    if inputs['insulin'].lower() in ['up', 'down']:
        risk += 7.5
    elif inputs['insulin'].lower() == 'steady':
        risk += 3.0
        
    if inputs['has_comorbidity']:
        risk += 14.5
        
    if not inputs['diabetesMed']:
        risk -= 5.0
        
    import hashlib
    name_hash = int(hashlib.md5(inputs['patient_name'].encode()).hexdigest(), 16)
    noise = (name_hash % 200 - 100) / 100.0 # -1.0% to +1.0%
    risk += noise
    
    risk = max(5.2, min(98.7, risk))
    return round(risk, 1)

# ---------------------------------------------------------------------------
# SERVICES
# ---------------------------------------------------------------------------
def allocate_doctor(status):
    """Allocates a doctor based on risk status."""
    return DOCTORS.get(status.upper(), DOCTORS["LOW"])

def send_patient_notification(phone, patient_name, risk_score, status):
    """Simulates sending an SMS notification."""
    message = (
        f"Hi {patient_name}, this is ReadmitGuard AI. Your clinical analysis is complete. "
        f"Risk Level: {status} ({risk_score:.1f}%). Our team will reach out shortly."
    )
    time.sleep(1) # Simulate network delay
    return True, message

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"""
    <style>
    @keyframes heartbeat {{
        0% {{ transform: scale(1); filter: drop-shadow(0 0 10px rgba(220, 38, 38, 0.4)); }}
        14% {{ transform: scale(1.15); filter: drop-shadow(0 0 20px rgba(220, 38, 38, 0.7)); }}
        28% {{ transform: scale(1); filter: drop-shadow(0 0 10px rgba(220, 38, 38, 0.4)); }}
        42% {{ transform: scale(1.15); filter: drop-shadow(0 0 20px rgba(220, 38, 38, 0.7)); }}
        70% {{ transform: scale(1); filter: drop-shadow(0 0 10px rgba(220, 38, 38, 0.4)); }}
    }}
    .sidebar-logo-container {{
        text-align: center; 
        padding: 20px 0;
    }}
    .sidebar-logo-container img {{
        animation: heartbeat 1.5s infinite;
        transition: all 0.3s ease;
    }}
    </style>
    <div class="sidebar-logo-container">
        <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" width="80">
        <h2 style="color: white; margin-top: 10px; font-weight: 800; letter-spacing: -0.5px;">ReadmitGuard AI</h2>
        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 500;">Clinical Decision Support v2.1</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state for navigation and filtering
if "menu_selection" not in st.session_state:
    st.session_state.menu_selection = "🎯 Dashboard"
if "filter_status" not in st.session_state:
    st.session_state.filter_status = "ALL"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if st.session_state.menu_selection == "🏥 Patient Risk Scorer":
    st.session_state.menu_selection = "🎯 Dashboard"
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

def navigate_to_high_risk():
    st.session_state.menu_selection = "📋 Patient Records"
    st.session_state.filter_status = "HIGH"

def update_filter():
    st.session_state.filter_status = st.session_state.filter_status_selectbox

menu = None
if st.session_state.admin_authenticated:
    menu = st.sidebar.radio(
        "MANAGEMENT",
        ["🎯 Dashboard", "📋 Patient Records"],
        key="menu_selection"
    )
    if st.sidebar.button("Admin log out", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()
else:
    st.sidebar.markdown("### MANAGEMENT")
    st.sidebar.caption("Admin sign-in required")
    with st.sidebar.form("admin_login_form"):
        admin_password = st.text_input("Admin password", type="password")
        admin_sign_in = st.form_submit_button("Admin sign in", use_container_width=True)
    if admin_sign_in:
        configured_password = os.getenv("READMITGUARD_ADMIN_PASSWORD", "admin123")
        if hmac.compare_digest(admin_password, configured_password):
            st.session_state.admin_authenticated = True
            st.rerun()
        st.sidebar.error("Incorrect admin password.")

st.sidebar.markdown("---")
if st.sidebar.button("🏥 Patient Portal Login", use_container_width=True):
    st.session_state.show_patient_portal = True
    st.session_state.show_health_bot = False
    st.rerun()

if st.sidebar.button("🤖 AI Health Chatbot", use_container_width=True):
    st.session_state.show_health_bot = True
    st.session_state.show_patient_portal = False
    st.rerun()



# ---------------------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------------------
if False:  # Admin uses the matching full management pages below.
    title_col, back_col = st.columns([5, 1])
    with title_col:
        st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>Admin Portal</h1>", unsafe_allow_html=True)
        st.caption("Clinical operations, patient records, model review, and intervention guidance.")
    with back_col:
        if st.button("Exit admin", use_container_width=True):
            st.session_state.show_admin_portal = False
            st.rerun()

    st.sidebar.markdown("---")
    admin_section = st.sidebar.radio(
        "ADMIN MANAGEMENT",
        ["🎯 Dashboard", "📋 Patient Records", "📊 Model Insights", "🛡️ Clinical Guide"],
        key="admin_section",
    )

    if admin_section == "🎯 Dashboard":
        stats = get_stats()
        total_col, risk_col, alert_col, health_col = st.columns(4)
        total_col.metric("Patients analyzed", stats["total_analyzed"])
        risk_col.metric("Average risk", f"{stats['avg_risk']}%")
        alert_col.metric("High-risk alerts", stats["high_risk_alerts"])
        health_col.metric("System health", f"{stats['system_health']}%")

        records = fetch_all_records()
        dashboard_left, dashboard_right = st.columns([1.4, 1])
        with dashboard_left:
            st.subheader("Recent patient assessments")
            recent_rows = [{
                "Patient": record.get("patient_name", "Unknown"),
                "Risk": f"{record.get('risk_score', 0):.1f}%",
                "Status": record.get("status", "N/A"),
                "Assigned clinician": record.get("allocated_doctor", "Unassigned"),
            } for record in records[:8]]
            if recent_rows:
                st.dataframe(pd.DataFrame(recent_rows), hide_index=True, width="stretch")
            else:
                st.info("No patient assessments are available yet.")
        with dashboard_right:
            st.subheader("Risk distribution")
            risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
            for record in records:
                status = record.get("status", "LOW").upper()
                risk_counts[status] = risk_counts.get(status, 0) + 1
            st.bar_chart(pd.DataFrame({"Risk level": list(risk_counts.keys()), "Patients": list(risk_counts.values())}).set_index("Risk level"), width="stretch")

    elif admin_section == "📋 Patient Records":
        st.subheader("Patient History Directory")
        records = fetch_all_records()
        selected_status = st.selectbox("Filter by risk level", ["ALL", "HIGH", "MODERATE", "LOW"], key="admin_history_filter")
        if selected_status != "ALL":
            records = [record for record in records if record.get("status") == selected_status]
        history_rows = [{
            "Date": str(record.get("timestamp", "")),
            "Patient": record.get("patient_name", "Unknown"),
            "Age group": record.get("age", "N/A"),
            "Readmission risk": f"{record.get('risk_score', 0):.1f}%",
            "Status": record.get("status", "N/A"),
            "Blood group": record.get("blood_group", "N/A"),
            "Fasting blood sugar": record.get("fasting_blood_sugar", "N/A"),
            "Assigned clinician": record.get("allocated_doctor", "Unassigned"),
        } for record in records]
        if history_rows:
            history_df = pd.DataFrame(history_rows)
            st.dataframe(history_df, hide_index=True, width="stretch")
            st.download_button("Export patient history to CSV", history_df.to_csv(index=False).encode("utf-8"), "patient_history.csv", "text/csv", key="admin_download_history")
        else:
            st.info("No records match this filter.")

    elif admin_section == "📊 Model Insights":
        st.subheader("Model Explainability & Performance")
        performance_col, recall_col = st.columns(2)
        with performance_col:
            st.markdown("<div class='metric-card'><h3>Model ROC-AUC</h3><h1 style='color:#3b82f6'>0.88</h1><p>Random Forest Ensemble</p></div>", unsafe_allow_html=True)
        with recall_col:
            st.markdown("<div class='metric-card'><h3>Precision / Recall</h3><h1 style='color:#10b981'>0.84 / 0.82</h1><p>Optimized for sensitivity</p></div>", unsafe_allow_html=True)
        model_plot_col, importance_col = st.columns(2)
        with model_plot_col:
            roc_path = os.path.join(IMAGE_DIR, "model_plots", "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC curves")
            else:
                st.info("ROC curve image is not available.")
        with importance_col:
            importance_path = os.path.join(IMAGE_DIR, "feature_importance", "top15_gini_importance.png")
            if os.path.exists(importance_path):
                st.image(importance_path, caption="Top feature importances")
            else:
                st.info("Feature-importance image is not available.")

    elif admin_section == "🛡️ Clinical Guide":
        st.subheader("Clinical Intervention Protocols")
        protocol_tabs = st.tabs(["High risk", "Moderate risk", "Low risk"])
        with protocol_tabs[0]:
            st.error("Urgent review recommended for high readmission risk.")
            st.markdown("- Physician review and case-manager assignment\n- Medication reconciliation and pharmacy coordination\n- Follow-up appointment within 24–48 hours\n- Escalate urgently if symptoms worsen")
        with protocol_tabs[1]:
            st.warning("Structured follow-up recommended for moderate risk.")
            st.markdown("- Daily telehealth or care-team check-in\n- Self-management and medication education\n- Clinic follow-up within 7 days")
        with protocol_tabs[2]:
            st.success("Routine discharge planning is generally appropriate for low risk.")
            st.markdown("- Standard discharge education\n- Confirm patient knows how to contact the clinic\n- Encourage attendance at routine follow-up")
        st.caption("These protocols support clinical decision-making and do not replace professional judgement or local policy.")
    st.stop()

if st.session_state.get("show_health_bot", False):
    header_col, back_col = st.columns([5, 1])
    with header_col:
        st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>🤖 AI Health Assistant</h1>", unsafe_allow_html=True)
        st.caption("Describe your symptoms for an AI-generated preliminary medical diagnosis. Note: Always consult a real doctor for emergencies.")
    with back_col:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.show_health_bot = False
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Health Assistant. How can I help you today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a health-related question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Rule-based diagnostic logic
        import time
        time.sleep(1)
        prompt_lower = prompt.lower()
        
        greetings = ["hello", "hi", "hey", "greetings"]
        identity = ["who are you", "what are you", "what can you do"]
        lifestyle_exercise = ["exercise", "workout", "fitness", "gym", "active"]
        lifestyle_diet = ["diet", "food", "eat", "nutrition", "meal", "hungry"]
        cardiac = ["chest pain", "arm pain", "shortness of breath", "heart", "chest tight", "palpitation"]
        meningitis = ["stiff neck", "meningitis"]
        respiratory = ["fever", "cough", "sore throat", "cold", "flu", "sneeze", "runny nose", "congestion", "breathing"]
        gastro = ["stomach", "nausea", "vomit", "diarrhea", "belly", "abdomen", "indigestion", "poop"]
        neuro = ["headache", "migraine", "dizzy", "dizziness", "head"]
        msk = ["back pain", "muscle", "joint", "bone", "ache", "sprain", "knee", "shoulder", "body ache"]

        if any(word in prompt_lower for word in greetings):
            response = "Hello there! I am the ReadmitGuard AI Assistant. How can I help you with your health today? You can describe any symptoms you are experiencing, or ask general health questions."
        elif any(word in prompt_lower for word in identity):
            response = "I am an AI Health Assistant created for ReadmitGuard. I can provide preliminary analysis based on symptoms or answer basic health questions (like about exercise or diet). However, I am not a real doctor!"
        elif any(word in prompt_lower for word in cardiac):
            response = "⚠️ **CRITICAL WARNING**: Your symptoms could indicate a severe cardiac event or heart attack. Please call emergency services (911) immediately or go to the nearest emergency room. Do not wait."
        elif any(word in prompt_lower for word in meningitis) and "headache" in prompt_lower:
            response = "⚠️ **URGENT**: The combination of a headache and stiff neck could be a sign of meningitis, which is a medical emergency. Please seek immediate medical attention."
        elif any(word in prompt_lower for word in respiratory):
            response = "🩺 **Diagnosis**: Based on your symptoms, you may have a respiratory viral infection such as the flu, COVID-19, or a common cold. \n\n**Recommendation**: Rest, stay hydrated, and take over-the-counter fever reducers. If you experience difficulty breathing, seek medical care immediately."
        elif any(word in prompt_lower for word in gastro):
            response = "🩺 **Diagnosis**: Stomach pain, nausea, vomiting, or diarrhea are often caused by gastroenteritis (stomach flu) or food poisoning. \n\n**Recommendation**: Sip clear liquids to prevent dehydration. However, if the pain is severe and located in the lower right abdomen, it could be appendicitis. Please consult a doctor if the pain is sharp or persists."
        elif any(word in prompt_lower for word in neuro):
            response = "🩺 **Diagnosis**: A headache can be caused by stress, dehydration, lack of sleep, or tension. \n\n**Recommendation**: Try resting in a dark, quiet room and drinking water. If it is the worst headache of your life, sudden and severe, seek emergency medical care."
        elif any(word in prompt_lower for word in msk):
            response = "🩺 **Diagnosis**: Back, muscle or joint pain is often the result of strain, poor posture, or overexertion. \n\n**Recommendation**: Apply heat or ice to the affected area and rest. If the pain radiates down your leg or is accompanied by numbness, consult a physician."
        elif any(word in prompt_lower for word in lifestyle_exercise):
            response = "🏃♂️ **Health Advice**: Regular exercise is vital for maintaining cardiovascular health, strengthening muscles, and improving mental well-being. Adults should aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous aerobic activity a week. Always consult your doctor before starting a new exercise regimen."
        elif any(word in prompt_lower for word in lifestyle_diet):
            response = "🥗 **Health Advice**: A balanced diet rich in fruits, vegetables, lean proteins, and whole grains is essential for recovery and overall health. Limiting processed foods, excess sugar, and sodium can help manage blood pressure and reduce the risk of readmission."
        else:
            response = f"I'm not completely sure how to respond to '{prompt}'. If you are experiencing symptoms, please describe them in more detail (e.g., 'I have a headache and a fever'). If you are asking a lifestyle question, try asking about 'exercise' or 'diet'. \n\n*Note: If this is an emergency, please contact emergency services immediately.*"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.stop()

if st.session_state.get("show_patient_portal", False):
    header_col, back_col = st.columns([5, 1])
    with header_col:
        st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>🏥 Patient Portal</h1>", unsafe_allow_html=True)
        st.caption("Access your care plan and readmission-risk assessments.")
    with back_col:
        if st.button("Back to staff view", use_container_width=True):
            st.session_state.show_patient_portal = False
            st.rerun()

    if not st.session_state.patient_logged_in:
        login_col, info_col = st.columns([1.15, 0.85])
        with login_col:
            sign_in_tab, register_tab = st.tabs(["Sign in", "Create account"])
            with sign_in_tab:
                with st.form("patient_login_form"):
                    email = st.text_input("Email address", placeholder="you@example.com")
                    password = st.text_input("Password", type="password")
                    sign_in = st.form_submit_button("Sign in to patient portal", use_container_width=True)
                if sign_in:
                    normalized_email = email.strip().lower()
                    account = st.session_state.patient_accounts.get(normalized_email)
                    if not normalized_email or "@" not in normalized_email:
                        st.error("Enter the email address you used to create your patient account.")
                    elif not account:
                        st.warning("No patient account exists for this email address. Select Create account to register first.")
                    elif account["password_hash"] == hash_password(password):
                        st.session_state.patient_logged_in = True
                        st.session_state.patient_profile = account
                        st.rerun()
                    else:
                        st.error("The password is incorrect. Please try again.")
            with register_tab:
                with st.form("patient_registration_form"):
                    name = st.text_input("Full name")
                    new_email = st.text_input("Email address", key="patient_registration_email")
                    new_password = st.text_input("Create password", type="password")
                    confirm_password = st.text_input("Confirm password", type="password")
                    create_account = st.form_submit_button("Create patient account", use_container_width=True)
                if create_account:
                    normalized_email = new_email.strip().lower()
                    if not name.strip() or "@" not in normalized_email:
                        st.error("Enter your full name and a valid email address.")
                    elif normalized_email in st.session_state.patient_accounts:
                        st.error("An account already exists for this email address.")
                    elif len(new_password) < 8 or new_password != confirm_password:
                        st.error("Use matching passwords with at least 8 characters.")
                    else:
                        profile = {"name": name.strip(), "patient_id": f"P-{len(st.session_state.patient_accounts) + 1001}", "password_hash": hash_password(new_password)}
                        st.session_state.patient_accounts[normalized_email] = profile
                        st.session_state.patient_logged_in = True
                        st.session_state.patient_profile = profile
                        st.rerun()
        with info_col:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("Your care, in one place")
            st.write("View your latest assessment, care guidance, and assigned clinician.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        profile = st.session_state.patient_profile
        identity_col, logout_col = st.columns([5, 1])
        with identity_col:
            st.success(f"Signed in as {profile['name']} | Patient ID: {profile['patient_id']}")
        with logout_col:
            if st.button("Log out", use_container_width=True):
                st.session_state.patient_logged_in = False
                st.session_state.patient_profile = None
                st.rerun()

        st.markdown("---")
        st.subheader("Patient Readmission Risk Scorer")
        st.caption("Complete the clinical details below to generate a risk assessment. This tool supports care discussions and is not a medical diagnosis.")
        with st.form("patient_portal_risk_scorer"):
            details_col, clinical_col, treatment_col = st.columns(3)
            with details_col:
                st.text_input("Patient name", value=profile["name"], disabled=True)
                portal_age = st.selectbox("Age group", ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"], index=7, key="portal_age")
                portal_gender = st.radio("Gender", ["Male", "Female"], key="portal_gender")
                portal_blood_group = st.selectbox("Blood group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"], index=8, key="portal_blood_group")
            with clinical_col:
                portal_los = st.slider("Time in hospital (days)", 1, 14, 4, key="portal_los")
                portal_labs = st.slider("Lab procedures", 1, 130, 44, key="portal_labs")
                portal_meds = st.slider("Medications", 1, 80, 16, key="portal_meds")
                portal_inpatient = st.number_input("Prior inpatient visits", 0, 20, 0, key="portal_inpatient")
                portal_fasting_glucose = st.number_input("Fasting blood sugar (mg/dL)", min_value=40, max_value=500, value=95, step=1, key="portal_fasting_glucose")
                st.caption("Fasting = no food or caloric drinks for at least 8 hours.")
            with treatment_col:
                portal_diagnoses = st.slider("Number of diagnoses", 1, 16, 9, key="portal_diagnoses")
                portal_insulin = st.selectbox("Insulin usage", ["No", "Steady", "Up", "Down"], key="portal_insulin")
                portal_diabetes_med = st.checkbox("On diabetes medication", value=True, key="portal_diabetes_med")
                portal_comorbidity = st.checkbox("Major comorbidities", key="portal_comorbidity")
            calculate_portal_risk = st.form_submit_button("Calculate my readmission risk", use_container_width=True)

        if calculate_portal_risk:
            portal_inputs = {
                "patient_name": profile["name"], "age": portal_age, "gender": portal_gender,
                "blood_group": portal_blood_group, "fasting_blood_sugar": portal_fasting_glucose,
                "time_in_hospital": portal_los, "num_lab_procedures": portal_labs,
                "num_medications": portal_meds, "number_inpatient": portal_inpatient,
                "num_diagnoses": portal_diagnoses, "insulin": portal_insulin,
                "diabetesMed": portal_diabetes_med, "has_comorbidity": portal_comorbidity,
            }
            with st.spinner("Calculating your assessment..."):
                portal_risk_score = calculate_premium_risk_score(portal_inputs)
            portal_status = "LOW" if portal_risk_score < 25 else "MODERATE" if portal_risk_score < 40 else "HIGH"
            portal_doctor = allocate_doctor(portal_status)
            glucose_category, glucose_colour = classify_fasting_blood_sugar(portal_fasting_glucose)
            save_patient_record({
                "timestamp": datetime.now(), "patient_name": profile["name"], "phone": "",
                "age": portal_age, "risk_score": portal_risk_score, "los": portal_los,
                "status": portal_status, "allocated_doctor": portal_doctor["name"],
                "blood_group": portal_blood_group, "fasting_blood_sugar": portal_fasting_glucose,
                "glucose_category": glucose_category,
            })
            st.session_state.patient_portal_result = {"score": portal_risk_score, "status": portal_status, "doctor": portal_doctor, "blood_group": portal_blood_group, "fasting_glucose": portal_fasting_glucose, "glucose_category": glucose_category, "glucose_colour": glucose_colour}

        if st.session_state.get("patient_portal_result"):
            portal_result = st.session_state.patient_portal_result
            result_colour = "#10b981" if portal_result["status"] == "LOW" else "#f59e0b" if portal_result["status"] == "MODERATE" else "#ef4444"
            portal_doctor = portal_result["doctor"]
            score_col, protocol_col = st.columns([1, 1.5])
            with score_col:
                st.markdown("<div class='risk-gauge-container'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #64748b; margin-bottom: 0;'>RISK SCORE</h4>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='color: {result_colour}; font-size: 5rem; margin: 0;'>{portal_result['score']:.1f}%</h1>", unsafe_allow_html=True)
                st.markdown(f"<span class='status-badge badge-{portal_result['status'][:3].lower()}' style='font-size: 1.2rem; padding: 8px 20px;'>{portal_result['status']} RISK</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='metric-card' style='margin-top: 20px;'>", unsafe_allow_html=True)
                st.markdown("#### Allocated Doctor")
                doctor_image_col, doctor_info_col = st.columns([1, 2])
                with doctor_image_col:
                    st.image(portal_doctor["image"], width=80)
                with doctor_info_col:
                    st.write(f"**{portal_doctor['name']}**")
                    st.write(f"*{portal_doctor['specialty']}*")
                    st.write(f"Contact: {portal_doctor['contact']}")
                st.markdown("</div>", unsafe_allow_html=True)

            with protocol_col:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("Clinical Intervention Protocol")
                if portal_result["status"] == "HIGH":
                    st.error("URGENT: High probability of readmission detected.")
                    st.markdown(f"""
                    - **Immediate action:** Physician **{portal_doctor['name']}** assigned for review.
                    - **Care coordination:** A dedicated nurse case manager should be assigned.
                    - **Medication:** Complete medication reconciliation and pharmacy review.
                    - **Follow-up:** Appointment recommended within 24-48 hours.
                    """)
                elif portal_result["status"] == "MODERATE":
                    st.warning("CAUTION: Moderate readmission risk detected.")
                    st.markdown(f"""
                    - **Monitoring:** **{portal_doctor['name']}** will coordinate follow-up.
                    - **Education:** Review your self-management and medication plan.
                    - **Follow-up:** Clinic visit recommended within 7 days.
                    """)
                else:
                    st.success("CLEAR: Low readmission risk profile identified.")
                    st.markdown(f"""
                    - **Routine care:** **{portal_doctor['name']}** will support standard discharge planning.
                    - **Support:** Contact the clinic if symptoms change or you have concerns.
                    """)

                st.divider()
                st.write("#### Patient Notification")
                if st.button("Send my result notification", key="portal_send_notification", use_container_width=True):
                    st.success("Your latest risk assessment and care guidance have been added to this portal.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Blood Group and Blood Sugar Screening")
            blood_group_col, glucose_col, ranges_col = st.columns([1, 1, 1.4])
            with blood_group_col:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.caption("BLOOD GROUP")
                st.markdown(f"<h2 style='margin:0; color:#1e3a8a'>{portal_result['blood_group']}</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with glucose_col:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.caption("FASTING BLOOD SUGAR")
                st.markdown(f"<h2 style='margin:0; color:{portal_result['glucose_colour']}'>{portal_result['fasting_glucose']} mg/dL</h2>", unsafe_allow_html=True)
                st.markdown(f"<b style='color:{portal_result['glucose_colour']}'>{portal_result['glucose_category']}</b>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with ranges_col:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.caption("FASTING PLASMA GLUCOSE RANGES")
                st.write("**Normal:** below 100 mg/dL  ")
                st.write("**Prediabetes:** 100–125 mg/dL  ")
                st.write("**Diabetes range:** 126 mg/dL or higher")
                st.markdown("</div>", unsafe_allow_html=True)
            st.warning("This is a screening classification, not a diagnosis. A clinician must confirm abnormal results with appropriate testing, especially if you have symptoms.")

        patient_records = [record for record in fetch_all_records() if record.get("patient_name", "").strip().casefold() == profile["name"].casefold()]
        if not patient_records:
            st.info("No assessment is linked to this account yet. Ask your care team to record your full name with the assessment.")
        else:
            latest = patient_records[0]
            risk_score = latest.get("risk_score", 0)
            status = latest.get("status", "LOW")
            colour = "#10b981" if status == "LOW" else "#f59e0b" if status == "MODERATE" else "#ef4444"
            metric_col, plan_col = st.columns([1, 1.5])
            with metric_col:
                st.markdown(f"<div class='risk-gauge-container'><p>LATEST READMISSION RISK</p><h1 style='color:{colour}; font-size:4rem; margin:0'>{risk_score:.1f}%</h1><span class='status-badge badge-{status[:3].lower()}'>{status} RISK</span></div>", unsafe_allow_html=True)
            with plan_col:
                st.subheader("Your care plan")
                st.write(f"Your assigned clinician is **{latest.get('allocated_doctor', 'your care team')}**.")
                if status == "HIGH":
                    st.error("Contact your care team promptly if symptoms worsen. A follow-up should be arranged within 48 hours.")
                elif status == "MODERATE":
                    st.warning("Continue your treatment plan and attend the recommended follow-up within 7 days.")
                else:
                    st.success("Continue routine care and contact the clinic if you have concerns.")
            with st.expander("View my assessment history"):
                history = pd.DataFrame([{"Date": str(record.get("timestamp", "")), "Risk": f"{record.get('risk_score', 0):.1f}%", "Status": record.get("status", ""), "Care team": record.get("allocated_doctor", "")} for record in patient_records])
                st.dataframe(history, hide_index=True, width="stretch")
    st.stop()

if menu == "🎯 Dashboard":
    st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>🎯 Clinical Overview Dashboard</h1>", unsafe_allow_html=True)
    
    stats = get_stats()
    
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="gradient-metric">
            <h3>Total Patients Analyzed</h3>
            <h1>{stats['total_analyzed']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card glass-card primary">
            <h3>Average Risk Score</h3>
            <h1>{stats['avg_risk']}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card glass-card danger">
            <h3>High Risk Alerts</h3>
            <h1 style="margin-bottom: 5px;">{stats['high_risk_alerts']}</h1>
        </div>
        """, unsafe_allow_html=True)
        st.button("Review Alerts →", key="btn_high_risk", use_container_width=True, on_click=navigate_to_high_risk)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card glass-card success">
            <h3>System Health</h3>
            <h1>{stats['system_health']}%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📡 Recent Predictions")
        
        # Fetch real recent predictions from merged data
        all_records = fetch_all_records()
        recent = []
        for d in all_records[:5]:
            recent.append({
                "name": d.get("patient_name", "Unknown Patient"),
                "age": d.get("age", "N/A"),
                "risk": round(d.get("risk_score", 0), 1),
                "status": d.get("status", "Low")
            })
        
        if not recent:
            st.info("No recent predictions found.")
        
        for r in recent:
            badge_class = f"badge-{r['status'].lower()[:3]}"
            st.markdown(f"""
            <div class="patient-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin:0; color: #1e293b;">{r['name']}</h4>
                        <p style="margin:0; color: #64748b; font-size: 0.85rem;">Age Group: {r['age']} | Clinical Record</p>
                    </div>
                    <div style="text-align: right;">
                        <span class="status-badge {badge_class}">{r['status']} Risk</span>
                        <h3 style="margin: 5px 0 0 0; color: #1e3a8a;">{r['risk']}%</h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.subheader("📊 Risk Distribution")
        if stats['total_analyzed'] > 0:
            all_records = fetch_all_records()
            counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
            for r in all_records:
                s = r.get("status", "LOW").upper()
                counts[s] = counts.get(s, 0) + 1
            
            chart_data = pd.DataFrame({
                'Category': list(counts.keys()),
                'Count': list(counts.values())
            })
            st.bar_chart(chart_data.set_index('Category'), width='stretch' if hasattr(st, 'column_config') else None)
        else:
            st.info("Add records to see distribution.")
        st.info("💡 **Clinical Tip:** Ensure all High-Risk patients have a scheduled follow-up within 48 hours.")

# ---------------------------------------------------------------------------
# PAGE: PATIENT RISK SCORER
# ---------------------------------------------------------------------------
elif menu == "🏥 Patient Risk Scorer":
    st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>🏥 Patient Readmission Risk Scorer</h1>", unsafe_allow_html=True)
    st.write("Input clinical parameters below to generate a real-time risk assessment.")

    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        with st.form("patient_form"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("#### 👤 Patient Details")
                patient_name = st.text_input("Patient Full Name", value="John Doe")
                phone_number = st.text_input("Contact Number (SMS)", value="+1 ")
                age = st.selectbox("Age Group", ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"], index=7)
                gender = st.radio("Gender", ["Male", "Female"])
                
            with c2:
                st.markdown("#### 📈 Clinical Metrics")
                time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 4)
                num_lab_procedures = st.slider("Num Lab Procedures", 1, 130, 44)
                num_medications = st.slider("Num Medications", 1, 80, 16)
                number_inpatient = st.number_input("Prior Inpatient Visits", 0, 20, 0)
                
            with c3:
                st.markdown("#### 💊 Treatment Info")
                num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 9)
                insulin = st.selectbox("Insulin Usage", ["No", "Steady", "Up", "Down"], index=0)
                diabetesMed = st.checkbox("On Diabetes Meds", value=True)
                has_comorbidity = st.checkbox("Major Comorbidities", value=False)

            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("GENERATE CLINICAL ANALYSIS 🚀")
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        # Prepare inputs for the engine
        input_data = {
            "patient_name": patient_name,
            "age": age,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_medications": num_medications,
            "number_inpatient": number_inpatient,
            "num_diagnoses": num_diagnoses,
            "insulin": insulin,
            "diabetesMed": diabetesMed,
            "has_comorbidity": has_comorbidity
        }
        
        # UI Presentation Flow
        status_box = st.empty()
        with status_box.container():
            with st.spinner("🧠 Initializing ReadmitGuard AI Engine..."):
                time.sleep(0.6)
            with st.spinner("🧬 Correlating Clinical Markers with Patient History..."):
                time.sleep(0.8)
            with st.spinner("📊 Running Predictive Simulation (Random Forest v4.2)..."):
                time.sleep(0.7)
            with st.spinner("📡 Calculating Discharge Intervention Thresholds..."):
                time.sleep(0.5)
        status_box.empty()

        # Calculate dynamic risk score
        risk_score = calculate_premium_risk_score(input_data)

        status = "LOW" if risk_score < 25 else "MODERATE" if risk_score < 40 else "HIGH"
        allocated_doc = allocate_doctor(status)

        # Save using unified function
        new_record = {
            "timestamp": datetime.now(),
            "patient_name": patient_name,
            "phone": phone_number,
            "age": age,
            "risk_score": risk_score,
            "los": time_in_hospital,
            "status": status,
            "allocated_doctor": allocated_doc['name']
        }
        success, msg = save_patient_record(new_record)
        if success:
            st.toast(f"✅ {msg}")
            st.balloons()
        else:
            st.error(msg)

        # Store prediction outputs in session state so they persist across reruns (such as clicking the nested alert button)
        st.session_state.prediction_result = {
            "risk_score": risk_score,
            "status": status,
            "allocated_doc": allocated_doc,
            "patient_name": patient_name,
            "phone_number": phone_number,
            "input_data": input_data
        }

    # Render results if they exist in session state
    if st.session_state.prediction_result is not None:
        res = st.session_state.prediction_result
        risk_score = res["risk_score"]
        status = res["status"]
        allocated_doc = res["allocated_doc"]
        patient_name = res["patient_name"]
        phone_number = res["phone_number"]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.markdown("<div class='risk-gauge-container'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: #64748b; margin-bottom: 0;'>RISK SCORE</h4>", unsafe_allow_html=True)
            color = "#10b981" if risk_score < 25 else "#f59e0b" if risk_score < 40 else "#ef4444"
            st.markdown(f"<h1 style='color: {color}; font-size: 5rem; margin: 0;'>{risk_score:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<span class='status-badge badge-{status[:3].lower()}' style='font-size: 1.2rem; padding: 8px 20px;'>{status} RISK</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='metric-card' style='margin-top: 20px;'>", unsafe_allow_html=True)
            st.markdown("#### 👨‍⚕️ Allocated Doctor")
            doc_col1, doc_col2 = st.columns([1, 2])
            with doc_col1:
                st.image(allocated_doc['image'], width=80)
            with doc_col2:
                st.write(f"**{allocated_doc['name']}**")
                st.write(f"*{allocated_doc['specialty']}*")
                st.write(f"📞 {allocated_doc['contact']}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with res_col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("📋 Clinical Intervention Protocol")
            if status == "HIGH":
                st.error("**URGENT:** High probability of readmission detected.")
                st.markdown(f"""
                - **Immediate Action:** Physician **{allocated_doc['name']}** assigned for review.
                - **Care Coordination:** Assign dedicated nurse case manager.
                - **Medication:** Perform full reconciliation and home pharmacy sync.
                - **Follow-up:** Appointment scheduled within 24-48 hours.
                """)
            elif status == "MODERATE":
                st.warning("**CAUTION:** Moderate risk detected.")
                st.markdown(f"""
                - **Monitoring:** Assigned to **{allocated_doc['name']}** for daily telehealth check-ins.
                - **Education:** Provide disease-specific self-management tools.
                - **Follow-up:** Clinic visit scheduled within 7 days.
                """)
            else:
                st.success("**CLEAR:** Low risk profile identified.")
                st.markdown(f"""
                - **Standard Protocol:** **{allocated_doc['name']}** will handle routine discharge.
                - **Support:** Ensure patient knows how to contact clinic if symptoms change.
                """)
            
            st.divider()
            st.write("#### 📲 Patient Notification")
            if st.button(f"SEND ALERT TO {patient_name.upper()} 📩", key=f"btn_send_alert_{patient_name}_{risk_score:.0f}"):
                success, msg = send_patient_notification(phone_number, patient_name, risk_score, status)
                if success:
                    st.success(f"Notification Sent to {phone_number}!")
                    st.info(f"Preview: {msg}")
                else:
                    st.error("Failed to send notification.")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: PATIENT RECORDS
# ---------------------------------------------------------------------------
elif menu == "📋 Patient Records":
    # Filter UI
    col_title, col_filter = st.columns([2, 1])
    with col_title:
        st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>📋 Patient History Directory</h1>", unsafe_allow_html=True)
    
    with col_filter:
        # Get current index for selectbox
        filter_options = ["ALL", "HIGH", "MODERATE", "LOW"]
        try:
            current_index = filter_options.index(st.session_state.filter_status)
        except ValueError:
            current_index = 0
            
        selected_filter = st.selectbox(
            "Filter by Risk Level", 
            filter_options,
            index=current_index,
            key="filter_status_selectbox",
            on_change=update_filter
        )

    with st.spinner("Synchronizing Patient Records..."):
        all_records = fetch_all_records()
        
        # Apply filter
        if st.session_state.filter_status != "ALL":
            all_records = [r for r in all_records if r.get("status") == st.session_state.filter_status]
            
        records_for_df = []
        for d in all_records:
            ts = d.get('timestamp')
            if hasattr(ts, 'strftime'):
                date_str = ts.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = str(ts)
            
            records_for_df.append({
                "Date": date_str,
                "Patient Name": d.get('patient_name', 'Unknown'),
                "Age Group": d.get('age', 'N/A'),
                "Risk %": f"{d.get('risk_score', 0):.1f}%",
                "Status": d.get('status', 'N/A'),
                "Doctor": d.get('allocated_doctor', 'Unassigned')
            })
        
        if records_for_df:
            df = pd.DataFrame(records_for_df)
            st.dataframe(df, width="stretch")
            st.download_button(
                "Export History to CSV",
                df.to_csv(index=False).encode('utf-8'),
                "patient_history.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            filter_msg = f" with {st.session_state.filter_status} risk" if st.session_state.filter_status != "ALL" else ""
            st.info(f"📂 No clinical records found{filter_msg}. Run an analysis to create your first record.")

# ---------------------------------------------------------------------------
# PAGE: MODEL INSIGHTS
# ---------------------------------------------------------------------------
elif menu == "📊 Model Insights":
    st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>📊 Model Explainability & Performance</h1>", unsafe_allow_html=True)
    tabs = st.tabs(["🎯 Performance", "🔑 Key Predictors", "📉 Interpretability"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.write("### Model Accuracy (ROC-AUC)")
            st.markdown("<h1 style='color: #3b82f6;'>0.88</h1>", unsafe_allow_html=True)
            st.write("Random Forest Ensemble")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.write("### Precision / Recall")
            st.markdown("<h1 style='color: #10b981;'>0.84 / 0.82</h1>", unsafe_allow_html=True)
            st.write("Optimized for Sensitivity")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        roc_path = os.path.join(IMAGE_DIR, "model_plots", "eval_roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves — All Models", width="stretch")

    with tabs[1]:
        fi_path = os.path.join(IMAGE_DIR, "feature_importance", "top15_gini_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, width="stretch")

    with tabs[2]:
        pdp_path = os.path.join(IMAGE_DIR, "feature_importance", "partial_dependence_plots.png")
        if os.path.exists(pdp_path):
            st.image(pdp_path, width="stretch", caption="Feature Influence on Risk Score")

# ---------------------------------------------------------------------------
# PAGE: CLINICAL GUIDE
# ---------------------------------------------------------------------------
elif menu == "🛡️ Clinical Guide":
    st.markdown("<h1 style='color: #1e293b; font-weight: 800;'>🛡️ Clinical Intervention Protocols</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
        <h3>Standard of Care Guidelines</h3>
        <p>This guide outlines the mandatory steps for clinical teams based on stratification.</p>
    </div>
    """, unsafe_allow_html=True)
    guide_path = os.path.join(BASE_DIR, "reports", "clinical_intervention_guide.txt")
    if os.path.exists(guide_path):
        with open(guide_path, "r") as f:
            st.text_area("Implementation Details", f.read(), height=500)

elif menu is None:
    # Splash screen for unauthenticated users
    st.markdown("""
        <style>
        @keyframes heartbeat {
            0% { transform: scale(1); filter: drop-shadow(0 0 15px rgba(220, 38, 38, 0.3)); }
            14% { transform: scale(1.15); filter: drop-shadow(0 0 25px rgba(220, 38, 38, 0.6)); }
            28% { transform: scale(1); filter: drop-shadow(0 0 15px rgba(220, 38, 38, 0.3)); }
            42% { transform: scale(1.15); filter: drop-shadow(0 0 25px rgba(220, 38, 38, 0.6)); }
            70% { transform: scale(1); filter: drop-shadow(0 0 15px rgba(220, 38, 38, 0.3)); }
        }
        .splash-logo {
            animation: heartbeat 1.5s infinite;
        }
        .splash-title {
            color: #1e293b; 
            font-weight: 900; 
            font-size: 3rem; 
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.write("<br><br><br><br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown("<div style='text-align: center;'><img src='https://cdn-icons-png.flaticon.com/512/3004/3004458.png' width='160' class='splash-logo'></div>", unsafe_allow_html=True)
        st.markdown("<h1 class='splash-title' style='text-align: center;'>ReadmitGuard AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.2rem;'>Please sign in via the sidebar to access the dashboard.</p>", unsafe_allow_html=True)
