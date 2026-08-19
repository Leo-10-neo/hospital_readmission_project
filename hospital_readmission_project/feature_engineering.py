# -*- coding: utf-8 -*-
"""
feature_engineering.py
======================
Step 3 - Feature Engineering for Hospital Readmission Risk Scorer.
Loads pre-encoding data (data_outliers_handled.csv), engineers features,
and saves to data/processed/featured_data.csv.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from config import (
    DATA_OUTLIERS_HANDLED, PROCESSED_DATA_DIR,
    REPORTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR,
    TARGET_COLUMN, VIZ,
)
from utils import (
    log_message, log_section, generate_timestamp,
    save_data, print_step,
)

LOG       = os.path.join(LOGS_DIR,    "feature_engineering_log.txt")
FE_DIR    = os.path.join(VISUALIZATIONS_DIR, "feature_importance")
OUT_PATH  = os.path.join(PROCESSED_DATA_DIR, "featured_data.csv")
REPORT    = os.path.join(REPORTS_DIR, "feature_engineering_report.csv")

NEW_FEATURES = []   # track names of engineered features

def _track(df, name):
    if name not in NEW_FEATURES:
        NEW_FEATURES.append(name)
    return df

def _age_midpoint(age_str):
    """Convert '[60-70)' -> 65, '[70-80)' -> 75, etc."""
    try:
        s = str(age_str).strip("[)(>")
        parts = s.replace(")", "").replace("]", "").split("-")
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
        return float(s)
    except:
        return np.nan

# ICD-9 code range helpers
def _in_range(code, lo, hi):
    try:
        c = float(str(code).split(".")[0])
        return lo <= c <= hi
    except:
        return False

def _is_diabetes(code):
    return _in_range(code, 250, 250)

def _is_heart(code):
    return _in_range(code, 390, 429)

def _is_respiratory(code):
    return _in_range(code, 460, 519)

def _is_kidney(code):
    return _in_range(code, 580, 629)

# ============================================================
# LOAD
# ============================================================
def load_data():
    df = pd.read_csv(DATA_OUTLIERS_HANDLED, low_memory=False)
    log_message(f"Loaded: {df.shape}", LOG)
    return df

# ============================================================
# A: DIAGNOSTIC FEATURES
# ============================================================
def section_a(df):
    print_step("A", "Diagnostic Features")
    log_section("SECTION A - DIAGNOSTIC FEATURES", LOG)

    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]

    df["total_diagnoses_count"] = df[diag_cols].notna().sum(axis=1)
    _track(df, "total_diagnoses_count")

    df["has_primary_diagnosis"] = df["diag_1"].notna().astype(int) if "diag_1" in df.columns else 0
    _track(df, "has_primary_diagnosis")

    df["diagnosis_diversity"] = df[diag_cols].apply(
        lambda row: row.dropna().nunique(), axis=1)
    _track(df, "diagnosis_diversity")

    def _any_chronic(row):
        for c in diag_cols:
            v = row.get(c)
            if pd.notna(v) and any([_is_diabetes(v), _is_heart(v),
                                    _is_respiratory(v), _is_kidney(v)]):
                return 1
        return 0

    df["chronic_diagnosis_flag"] = df.apply(_any_chronic, axis=1)
    _track(df, "chronic_diagnosis_flag")

    log_message("Section A done.", LOG)
    return df

# ============================================================
# B: COMORBIDITY FEATURES
# ============================================================
def section_b(df):
    print_step("B", "Comorbidity Features")
    log_section("SECTION B - COMORBIDITY FEATURES", LOG)

    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]

    for flag, fn in [("has_diabetes", _is_diabetes),
                     ("has_heart_disease", _is_heart),
                     ("has_respiratory_disease", _is_respiratory),
                     ("has_kidney_disease", _is_kidney)]:
        df[flag] = df[diag_cols].apply(
            lambda row: int(any(fn(v) for v in row.dropna())), axis=1)
        _track(df, flag)

    chronic_flags = ["has_diabetes", "has_heart_disease",
                     "has_respiratory_disease", "has_kidney_disease"]
    df["chronic_disease_count"] = df[chronic_flags].sum(axis=1)
    _track(df, "chronic_disease_count")

    # Weighted comorbidity index (simplified Charlson)
    weights = {"has_diabetes": 1, "has_heart_disease": 2,
               "has_respiratory_disease": 1, "has_kidney_disease": 2}
    df["comorbidity_index"] = sum(df[c] * w for c, w in weights.items())
    _track(df, "comorbidity_index")

    log_message("Section B done.", LOG)
    return df

# ============================================================
# C: MEDICATION FEATURES
# ============================================================
def section_c(df):
    print_step("C", "Medication Features")
    log_section("SECTION C - MEDICATION FEATURES", LOG)

    if "num_medications" in df.columns:
        df["total_medications"] = df["num_medications"]
        _track(df, "total_medications")

        df["medication_complexity_score"] = pd.cut(
            df["num_medications"], bins=[-1, 5, 10, 999],
            labels=[1, 2, 3]).astype(float)
        _track(df, "medication_complexity_score")

    if "insulin" in df.columns:
        df["insulin_usage"] = (df["insulin"].astype(str).str.lower() != "no").astype(int)
        _track(df, "insulin_usage")

    if "diabetesMed" in df.columns:
        df["diabetic_med_flag"] = (df["diabetesMed"].astype(str).str.lower() == "yes").astype(int)
        _track(df, "diabetic_med_flag")

    if "change" in df.columns:
        df["medication_change_flag"] = (df["change"].astype(str).str.lower() == "ch").astype(int)
        _track(df, "medication_change_flag")

    log_message("Section C done.", LOG)
    return df

# ============================================================
# D: PROCEDURE FEATURES
# ============================================================
def section_d(df):
    print_step("D", "Procedure Features")
    log_section("SECTION D - PROCEDURE FEATURES", LOG)

    if "num_procedures" in df.columns:
        df["total_procedures"] = df["num_procedures"]
        _track(df, "total_procedures")

        df["procedure_complexity_score"] = pd.cut(
            df["num_procedures"], bins=[-1, 0, 2, 5, 999],
            labels=[0, 1, 2, 3]).astype(float)
        _track(df, "procedure_complexity_score")

        df["has_surgery"] = (df["num_procedures"] > 0).astype(int)
        _track(df, "has_surgery")

    if "num_procedures" in df.columns and "time_in_hospital" in df.columns:
        df["procedure_to_stay_ratio"] = (
            df["num_procedures"] / df["time_in_hospital"].replace(0, 1))
        _track(df, "procedure_to_stay_ratio")

    log_message("Section D done.", LOG)
    return df

# ============================================================
# E: HOSPITAL STAY FEATURES
# ============================================================
def section_e(df):
    print_step("E", "Hospital Stay Features")
    log_section("SECTION E - HOSPITAL STAY FEATURES", LOG)

    if "time_in_hospital" not in df.columns:
        return df

    df["hospital_stay_category"] = pd.cut(
        df["time_in_hospital"], bins=[0, 3, 7, 14, 999],
        labels=[0, 1, 2, 3]).astype(float)
    _track(df, "hospital_stay_category")

    df["length_of_stay_squared"] = df["time_in_hospital"] ** 2
    _track(df, "length_of_stay_squared")

    df["length_of_stay_log"] = np.log1p(df["time_in_hospital"])
    _track(df, "length_of_stay_log")

    if "num_procedures" in df.columns:
        df["average_daily_procedures"] = (
            df["num_procedures"] / df["time_in_hospital"].replace(0, 1))
        _track(df, "average_daily_procedures")

    df["time_in_hospital_category"] = df["hospital_stay_category"]
    _track(df, "time_in_hospital_category")

    log_message("Section E done.", LOG)
    return df

# ============================================================
# F: ADMISSION FEATURES
# ============================================================
def section_f(df):
    print_step("F", "Admission Features")
    log_section("SECTION F - ADMISSION FEATURES", LOG)

    if "number_inpatient" in df.columns:
        df["readmission_history_flag"] = (df["number_inpatient"] > 0).astype(int)
        _track(df, "readmission_history_flag")

    if "number_emergency" in df.columns:
        df["emergency_admission_flag"] = (df["number_emergency"] > 0).astype(int)
        _track(df, "emergency_admission_flag")

    if "admission_type_id" in df.columns:
        # 1=Emergency,2=Urgent,3=Elective; higher risk for 1
        risk_map = {1: 3, 2: 2, 3: 1}
        df["admission_type_risk_score"] = (
            pd.to_numeric(df["admission_type_id"], errors="coerce")
            .map(risk_map).fillna(1))
        _track(df, "admission_type_risk_score")

    if "admission_source_id" in df.columns:
        # Source 7=Emergency Room -> higher risk
        df["admission_source_risk"] = (
            pd.to_numeric(df["admission_source_id"], errors="coerce")
            .apply(lambda x: 3 if x == 7 else (2 if x in [1, 2] else 1)))
        _track(df, "admission_source_risk")

    if "number_inpatient" in df.columns:
        df["days_since_last_admission"] = df["number_inpatient"].apply(
            lambda x: 0 if x == 0 else 30)   # proxy; 0=no history
        _track(df, "days_since_last_admission")

    log_message("Section F done.", LOG)
    return df

# ============================================================
# G: AGE & DEMOGRAPHIC FEATURES
# ============================================================
def section_g(df):
    print_step("G", "Age & Demographic Features")
    log_section("SECTION G - AGE & DEMOGRAPHIC FEATURES", LOG)

    if "age" in df.columns:
        df["age_midpoint"] = df["age"].apply(_age_midpoint)

        df["age_risk_group"] = pd.cut(
            df["age_midpoint"], bins=[-1, 40, 60, 75, 999],
            labels=[0, 1, 2, 3]).astype(float)
        _track(df, "age_risk_group")

        df["elderly_flag"] = (df["age_midpoint"] > 65).astype(int)
        _track(df, "elderly_flag")

        df["very_elderly_flag"] = (df["age_midpoint"] > 75).astype(int)
        _track(df, "very_elderly_flag")

        if "num_medications" in df.columns:
            df["age_medication_interaction"] = (
                df["age_midpoint"].fillna(0) * df["num_medications"].fillna(0))
            _track(df, "age_medication_interaction")

        if "chronic_disease_count" in df.columns:
            df["age_comorbidity_interaction"] = (
                df["age_midpoint"].fillna(0) * df["chronic_disease_count"].fillna(0))
            _track(df, "age_comorbidity_interaction")

    log_message("Section G done.", LOG)
    return df

# ============================================================
# H: LAB RESULT FEATURES
# ============================================================
def section_h(df):
    print_step("H", "Lab Result Features")
    log_section("SECTION H - LAB FEATURES", LOG)

    if "num_lab_procedures" in df.columns:
        df["lab_test_frequency"] = df["num_lab_procedures"]
        _track(df, "lab_test_frequency")

    if "max_glu_serum" in df.columns:
        glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
        df["glucose_level_category"] = (
            df["max_glu_serum"].astype(str).map(glu_map).fillna(0))
        _track(df, "glucose_level_category")

    if "A1Cresult" in df.columns:
        a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
        df["A1C_result_category"] = (
            df["A1Cresult"].astype(str).map(a1c_map).fillna(0))
        _track(df, "A1C_result_category")

    # Abnormal lab proxy: sum of glucose+A1C flags
    ab_cols = [c for c in ["glucose_level_category", "A1C_result_category"] if c in df.columns]
    if ab_cols:
        df["abnormal_lab_count"] = df[ab_cols].apply(
            lambda row: (row > 1).sum(), axis=1)
        _track(df, "abnormal_lab_count")

    log_message("Section H done.", LOG)
    return df

# ============================================================
# I: DISCHARGE FEATURES
# ============================================================
def section_i(df):
    print_step("I", "Discharge Features")
    log_section("SECTION I - DISCHARGE FEATURES", LOG)

    if "discharge_disposition_id" in df.columns:
        ddi = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")

        df["discharge_to_home_flag"]      = (ddi == 1).astype(int)
        df["discharge_with_services_flag"]= (ddi == 6).astype(int)  # home health
        df["ama_discharge_flag"]          = (ddi == 7).astype(int)  # AMA

        # Risk score: higher for AMA/SNF, lower for home
        def _risk(x):
            if x == 1: return 1
            if x in [3, 5, 6]: return 2
            if x == 7: return 3
            return 1
        df["discharge_disposition_risk"] = ddi.apply(_risk)

        for c in ["discharge_to_home_flag", "discharge_with_services_flag",
                  "ama_discharge_flag", "discharge_disposition_risk"]:
            _track(df, c)

    log_message("Section I done.", LOG)
    return df

# ============================================================
# J: INTERACTION FEATURES
# ============================================================
def section_j(df):
    print_step("J", "Interaction Features")
    log_section("SECTION J - INTERACTION FEATURES", LOG)

    if "age_midpoint" in df.columns and "time_in_hospital" in df.columns:
        df["age_length_stay"] = df["age_midpoint"].fillna(0) * df["time_in_hospital"].fillna(0)
        _track(df, "age_length_stay")

    if "has_diabetes" in df.columns and "diabetic_med_flag" in df.columns:
        df["diabetes_medication_combo"] = df["has_diabetes"] * df["diabetic_med_flag"]
        _track(df, "diabetes_medication_combo")

    if "total_procedures" in df.columns and "total_medications" in df.columns:
        df["procedures_medications"] = df["total_procedures"] * df["total_medications"]
        _track(df, "procedures_medications")

    if "comorbidity_index" in df.columns and "age_risk_group" in df.columns:
        df["comorbidity_age"] = df["comorbidity_index"] * df["age_risk_group"].fillna(0)
        _track(df, "comorbidity_age")

    if "emergency_admission_flag" in df.columns and "elderly_flag" in df.columns:
        df["emergency_elderly"] = df["emergency_admission_flag"] * df["elderly_flag"]
        _track(df, "emergency_elderly")

    log_message("Section J done.", LOG)
    return df

# ============================================================
# K: FEATURE VALIDATION
# ============================================================
def section_k(df, target_num):
    print_step("K", "Feature Validation")
    log_section("SECTION K - VALIDATION", LOG)

    rows = []
    for feat in NEW_FEATURES:
        if feat not in df.columns:
            continue
        col = df[feat]
        miss_pct = col.isna().mean() * 100
        corr     = col.corr(target_num) if col.dtype in [float, int] else np.nan
        rows.append({
            "feature"      : feat,
            "dtype"        : str(col.dtype),
            "missing_pct"  : round(miss_pct, 2),
            "unique_values": col.nunique(),
            "mean"         : round(col.mean(), 4) if col.dtype in [float, int] else "-",
            "std"          : round(col.std(), 4)  if col.dtype in [float, int] else "-",
            "corr_target"  : round(corr, 4) if not np.isnan(corr) else "-",
        })

    val_df = pd.DataFrame(rows)
    log_message(f"Validation table:\n{val_df.to_string()}", LOG)
    print("\nFeature Validation (top 10):")
    print(val_df.head(10).to_string(index=False))
    return val_df

# ============================================================
# L: FEATURE SUMMARY REPORT
# ============================================================
def section_l(val_df):
    print_step("L", "Feature Summary Report")

    def _ftype(dtype, n_unique):
        if n_unique == 2: return "binary"
        if "float" in str(dtype) or "int" in str(dtype): return "continuous"
        return "categorical"

    val_df["feature_type"] = val_df.apply(
        lambda r: _ftype(r["dtype"], r["unique_values"]), axis=1)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    val_df.to_csv(REPORT, index=False)
    log_message(f"Feature report saved -> {REPORT}", LOG)
    print(f"Feature report saved -> {REPORT}")
    return val_df

# ============================================================
# M: VISUALIZATIONS
# ============================================================
def section_m(df, val_df, target_num):
    print_step("M", "Visualizations")
    os.makedirs(FE_DIR, exist_ok=True)

    try:
        plt.style.use(VIZ["style"])
    except:
        plt.style.use("seaborn-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1 - Correlation bar chart
    corr_df = val_df[val_df["corr_target"] != "-"].copy()
    corr_df["corr_target"] = pd.to_numeric(corr_df["corr_target"], errors="coerce")
    corr_df = corr_df.dropna(subset=["corr_target"]).sort_values("corr_target", key=abs, ascending=True)
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in corr_df["corr_target"]]
    axes[0,0].barh(corr_df["feature"], corr_df["corr_target"], color=colors)
    axes[0,0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0,0].set_title("Engineered Feature Correlations with Target", fontsize=12)
    axes[0,0].set_xlabel("Pearson Correlation")

    # 2 - Distribution of top 6 features
    top6 = corr_df["feature"].iloc[-6:].tolist()
    ax2 = axes[0,1]
    ax2.set_title("Top Features Value Counts (normalized)", fontsize=12)
    for feat in top6:
        if feat in df.columns and df[feat].nunique() <= 20:
            vc = df[feat].value_counts(normalize=True).sort_index()
            ax2.plot(vc.index.astype(str), vc.values, marker="o", label=feat, alpha=0.7)
    ax2.legend(fontsize=7); ax2.set_xlabel("Value"); ax2.set_ylabel("Proportion")

    # 3 - Correlation heatmap of new features
    num_new = [f for f in NEW_FEATURES if f in df.columns and
               df[f].dtype in ["float64", "int64", "float32", "int32"]][:15]
    if len(num_new) >= 2:
        corr_mat = df[num_new].corr()
        sns.heatmap(corr_mat, ax=axes[1,0], cmap="coolwarm", center=0,
                    annot=len(num_new) <= 10, fmt=".1f", linewidths=0.3,
                    annot_kws={"size": 7})
        axes[1,0].set_title("New Feature Correlation Heatmap", fontsize=12)
        axes[1,0].tick_params(axis="x", rotation=45, labelsize=7)
        axes[1,0].tick_params(axis="y", labelsize=7)

    # 4 - Feature count comparison
    orig = df.shape[1] - len(NEW_FEATURES)
    axes[1,1].bar(["Original Features", "Engineered Features", "Total"],
                  [orig, len(NEW_FEATURES), df.shape[1]],
                  color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate([orig, len(NEW_FEATURES), df.shape[1]]):
        axes[1,1].text(i, v + 1, str(v), ha="center", fontsize=11)
    axes[1,1].set_title("Feature Count Comparison", fontsize=12)
    axes[1,1].set_ylabel("Count")

    fig.suptitle("Feature Engineering Summary", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FE_DIR, "engineered_features.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    log_message(f"Visualization saved -> {path}", LOG)
    print(f"Visualization saved -> {path}")

# ============================================================
# N: SAVE
# ============================================================
def section_n(df):
    print_step("N", "Save Engineered Dataset")
    save_data(df, OUT_PATH)

    print(f"\nOriginal feature count : {df.shape[1] - len(NEW_FEATURES)}")
    print(f"New features created   : {len(NEW_FEATURES)}")
    print(f"Final feature count    : {df.shape[1]}")
    print(f"\nSample (new cols only):")
    print(df[NEW_FEATURES[:6]].head(5).to_string())

    log_message(f"featured_data.csv saved -> {OUT_PATH}", LOG)
    log_message(f"New features: {NEW_FEATURES}", LOG)

# ============================================================
# MAIN
# ============================================================
def run_feature_engineering():
    print("\n" + "=" * 65)
    print("  HOSPITAL READMISSION -- FEATURE ENGINEERING")
    print(f"  Started: {generate_timestamp()}")
    print("=" * 65)

    os.makedirs(FE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    log_message("=" * 65, LOG)
    log_message(f"Feature engineering started: {generate_timestamp()}", LOG)

    df = load_data()
    orig_cols = df.shape[1]

    df = section_a(df)
    df = section_b(df)
    df = section_c(df)
    df = section_d(df)
    df = section_e(df)
    df = section_f(df)
    df = section_g(df)
    df = section_h(df)
    df = section_i(df)
    df = section_j(df)

    # Encode target for correlation
    le = LabelEncoder()
    target_num = pd.Series(
        le.fit_transform(df[TARGET_COLUMN].astype(str)),
        index=df.index, name=TARGET_COLUMN
    )

    val_df = section_k(df, target_num)
    val_df = section_l(val_df)
    section_m(df, val_df, target_num)
    section_n(df)

    # Top 5 correlated
    top5 = (val_df[val_df["corr_target"] != "-"]
            .assign(abs_corr=lambda x: pd.to_numeric(x["corr_target"], errors="coerce").abs())
            .nlargest(5, "abs_corr")[["feature", "corr_target"]])

    print("\n" + "=" * 65)
    print("  [DONE] FEATURE ENGINEERING COMPLETE")
    print(f"  Finished : {generate_timestamp()}")
    print(f"  Original features  : {orig_cols}")
    print(f"  New features       : {len(NEW_FEATURES)}")
    print(f"  Final features     : {df.shape[1]}")
    print(f"\n  Top 5 correlated new features:")
    print(top5.to_string(index=False))
    print(f"\n  Output : {OUT_PATH}")
    print(f"  Report : {REPORT}")
    print("=" * 65 + "\n")

    log_message(f"Feature engineering complete: {generate_timestamp()}", LOG)
    return df

if __name__ == "__main__":
    run_feature_engineering()
