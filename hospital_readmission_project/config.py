"""
config.py
=========
Central configuration for the Hospital Readmission Prediction project.
All file paths, column names, hyperparameters, and visualization settings
are defined here so every module can import from a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# ROOT DIRECTORY
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# DIRECTORY PATHS
# ---------------------------------------------------------------------------
DATA_DIR                = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR            = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR      = os.path.join(DATA_DIR, "processed")

REPORTS_DIR             = os.path.join(BASE_DIR, "reports")
LOGS_DIR                = os.path.join(BASE_DIR, "logs")

VISUALIZATIONS_DIR      = os.path.join(BASE_DIR, "visualizations")
EDA_DIR                 = os.path.join(VISUALIZATIONS_DIR, "eda")
MODEL_PLOTS_DIR         = os.path.join(VISUALIZATIONS_DIR, "model_plots")

MODELS_DIR              = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# FILE PATHS — RAW DATA
# ---------------------------------------------------------------------------
RAW_DATA_FILE           = os.path.join(RAW_DATA_DIR, "diabetic_data.csv")

# ---------------------------------------------------------------------------
# FILE PATHS — PROCESSED DATA
# ---------------------------------------------------------------------------
DATA_AFTER_IMPUTATION   = os.path.join(PROCESSED_DATA_DIR, "data_after_imputation.csv")
DATA_DEDUPLICATED       = os.path.join(PROCESSED_DATA_DIR, "data_deduplicated.csv")
DATA_OUTLIERS_HANDLED   = os.path.join(PROCESSED_DATA_DIR, "data_outliers_handled.csv")
PREPROCESSED_DATA       = os.path.join(PROCESSED_DATA_DIR, "preprocessed_data.csv")

# Train / Validation / Test splits
TRAIN_DATA_FILE         = os.path.join(PROCESSED_DATA_DIR, "train.csv")
VAL_DATA_FILE           = os.path.join(PROCESSED_DATA_DIR, "val.csv")
TEST_DATA_FILE          = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# ---------------------------------------------------------------------------
# FILE PATHS — REPORTS & LOGS
# ---------------------------------------------------------------------------
PREPROCESSING_LOG       = os.path.join(LOGS_DIR,    "preprocessing_log.txt")
MODELING_LOG            = os.path.join(LOGS_DIR,    "modeling_log.txt")

DATA_QUALITY_INITIAL    = os.path.join(REPORTS_DIR, "data_quality_initial.csv")
ENCODING_MAPPINGS       = os.path.join(REPORTS_DIR, "encoding_mappings.json")
PREPROCESSING_SUMMARY   = os.path.join(REPORTS_DIR, "preprocessing_summary.txt")
MODEL_RESULTS           = os.path.join(REPORTS_DIR, "model_results.csv")

# ---------------------------------------------------------------------------
# TARGET COLUMN
# ---------------------------------------------------------------------------
TARGET_COLUMN = "readmitted"

# ---------------------------------------------------------------------------
# FEATURE COLUMN NAMES
# (Update this list after one-hot encoding if needed; these are raw feature names)
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]

# Columns to drop before modelling (identifiers / leakage risk)
COLUMNS_TO_DROP = [
    "encounter_id",
    "patient_nbr",
]

# ---------------------------------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT RATIOS
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # implicit: 1 - TRAIN_RATIO - VAL_RATIO

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# OUTLIER TREATMENT
# ---------------------------------------------------------------------------
OUTLIER_METHOD        = "IQR"          # "IQR" or "zscore"
IQR_MULTIPLIER        = 1.5
CAPPING_LOWER_PCTILE  = 1              # 1st  percentile
CAPPING_UPPER_PCTILE  = 99             # 99th percentile

# ---------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
MODEL_HYPERPARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": RANDOM_SEED,
    },
}

# ---------------------------------------------------------------------------
# VISUALIZATION SETTINGS
# ---------------------------------------------------------------------------
VIZ = {
    "figure_size"      : (14, 8),      # default (width, height) in inches
    "figure_size_sq"   : (10, 10),     # square figures (correlation matrix, etc.)
    "figure_size_wide" : (18, 6),      # wide multi-panel figures
    "dpi"              : 150,
    "color_palette"    : "Set2",       # seaborn palette name
    "heatmap_cmap"     : "YlOrRd",
    "hist_color"       : "#4C72B0",
    "bar_color"        : "#55A868",
    "title_fontsize"   : 14,
    "label_fontsize"   : 11,
    "tick_fontsize"    : 9,
    "style"            : "seaborn-v0_8-whitegrid",
}
