# -*- coding: utf-8 -*-
"""
model_training.py
=================
STEP 5 - Model Training for Hospital Readmission Risk Scorer.

Sections:
  A  – Load Training Data
  B  – Handle Class Imbalance with SMOTE
  (More sections to be added)
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR,
    TARGET_COLUMN, RANDOM_SEED, VIZ,
)
from utils import log_message, log_section, generate_timestamp, print_step

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
LOG          = os.path.join(LOGS_DIR, "modeling_log.txt")
PLOT_DIR     = os.path.join(os.path.dirname(PROCESSED_DATA_DIR),
                            "visualizations", "model_plots")

X_TRAIN_PKL  = os.path.join(PROCESSED_DATA_DIR, "X_train.pkl")
Y_TRAIN_PKL  = os.path.join(PROCESSED_DATA_DIR, "y_train.pkl")
X_VAL_PKL    = os.path.join(PROCESSED_DATA_DIR, "X_val.pkl")
Y_VAL_PKL    = os.path.join(PROCESSED_DATA_DIR, "y_val.pkl")
X_TEST_PKL   = os.path.join(PROCESSED_DATA_DIR, "X_test.pkl")
Y_TEST_PKL   = os.path.join(PROCESSED_DATA_DIR, "y_test.pkl")

X_TRAIN_BAL  = os.path.join(PROCESSED_DATA_DIR, "X_train_balanced.pkl")
Y_TRAIN_BAL  = os.path.join(PROCESSED_DATA_DIR, "y_train_balanced.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# SECTION A: LOAD TRAINING DATA
# ============================================================
def section_a_load():
    print_step("A", "Load Training Data")
    log_section("SECTION A – LOAD TRAINING DATA", LOG)

    # Load from pickle files
    X_train = pd.read_pickle(X_TRAIN_PKL)
    y_train = pd.read_pickle(Y_TRAIN_PKL)
    X_val   = pd.read_pickle(X_VAL_PKL)
    y_val   = pd.read_pickle(Y_VAL_PKL)
    X_test  = pd.read_pickle(X_TEST_PKL)
    y_test  = pd.read_pickle(Y_TEST_PKL)

    # Fix target column: extract Series if DataFrame
    for name, obj in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        if isinstance(obj, pd.DataFrame):
            if TARGET_COLUMN in obj.columns:
                pass   # will unpack below

    def _extract_y(y):
        if isinstance(y, pd.DataFrame):
            return y[TARGET_COLUMN] if TARGET_COLUMN in y.columns else y.iloc[:, 0]
        return y

    y_train = _extract_y(y_train)
    y_val   = _extract_y(y_val)
    y_test  = _extract_y(y_test)

    # Display shapes
    print(f"\n  {'Split':<12} {'X shape':<20} {'y shape':<15}")
    print(f"  {'-'*47}")
    print(f"  {'Train':<12} {str(X_train.shape):<20} {str(y_train.shape):<15}")
    print(f"  {'Validation':<12} {str(X_val.shape):<20} {str(y_val.shape):<15}")
    print(f"  {'Test':<12} {str(X_test.shape):<20} {str(y_test.shape):<15}")

    # Data integrity — NaN check
    print(f"\n  NaN Check:")
    nan_issues = False
    for name, df in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        n_nan = df.isna().sum().sum()
        status = "OK (0 NaN)" if n_nan == 0 else f"WARN ({n_nan:,} NaN values)"
        print(f"    {name:<12}: {status}")
        if n_nan > 0:
            nan_issues = True
            log_message(f"[WARN] {name} has {n_nan} NaN values", LOG)

    for name, s in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        n_nan = s.isna().sum()
        status = "OK (0 NaN)" if n_nan == 0 else f"WARN ({n_nan} NaN)"
        print(f"    {name:<12}: {status}")

    if not nan_issues:
        print(f"\n  Data integrity: ALL CHECKS PASSED — no NaN values found")

    log_message(f"Loaded: X_train{X_train.shape} X_val{X_val.shape} X_test{X_test.shape}", LOG)
    log_message(f"NaN issues present: {nan_issues}", LOG)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# SECTION B: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================
def section_b_smote(X_train, y_train):
    print_step("B", "Handle Class Imbalance with SMOTE")
    log_section("SECTION B – SMOTE", LOG)

    # Class distribution BEFORE
    counts_before = y_train.value_counts().sort_index()
    total_before  = len(y_train)
    print(f"\n  Class distribution BEFORE SMOTE  (n={total_before:,}):")
    print(f"  {'Class':<20} {'Count':>8}   {'Ratio':>8}")
    print(f"  {'-'*40}")
    for cls, cnt in counts_before.items():
        label = "Readmitted (<30d)" if cls == 1 else "Not Readmitted"
        print(f"  {label:<20} {cnt:>8,}   {cnt/total_before*100:>7.2f}%")

    log_message(f"Before SMOTE: {counts_before.to_dict()}", LOG)

    # Apply SMOTE (train only)
    print(f"\n  Applying SMOTE (random_state={RANDOM_SEED}) ...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    X_bal = pd.DataFrame(X_bal, columns=X_train.columns)
    y_bal = pd.Series(y_bal, name=TARGET_COLUMN)

    # Class distribution AFTER
    counts_after = y_bal.value_counts().sort_index()
    total_after  = len(y_bal)
    n_synthetic  = total_after - total_before

    print(f"\n  Class distribution AFTER SMOTE  (n={total_after:,}):")
    print(f"  {'Class':<20} {'Count':>8}   {'Ratio':>8}")
    print(f"  {'-'*40}")
    for cls, cnt in counts_after.items():
        label = "Readmitted (<30d)" if cls == 1 else "Not Readmitted"
        print(f"  {label:<20} {cnt:>8,}   {cnt/total_after*100:>7.2f}%")

    print(f"\n  Synthetic samples created: {n_synthetic:,}")
    print(f"  Dataset size: {total_before:,}  ->  {total_after:,}  (+{n_synthetic:,})")

    log_message(f"After SMOTE: {counts_after.to_dict()}", LOG)
    log_message(f"Synthetic samples created: {n_synthetic:,}", LOG)

    # Save balanced training data
    X_bal.to_pickle(X_TRAIN_BAL)
    y_bal.to_pickle(Y_TRAIN_BAL)
    print(f"\n  Saved balanced X_train -> {X_TRAIN_BAL}")
    print(f"  Saved balanced y_train -> {Y_TRAIN_BAL}")
    log_message(f"Balanced data saved -> {X_TRAIN_BAL}", LOG)

    # Visualization: before vs after bar chart
    _plot_smote_comparison(counts_before, counts_after)

    return X_bal, y_bal


def _plot_smote_comparison(counts_before, counts_after):
    """Side-by-side bar chart comparing class distribution before/after SMOTE."""
    try:
        plt.style.use(VIZ["style"])
    except Exception:
        plt.style.use("seaborn-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Class Imbalance — Before vs After SMOTE", fontsize=14, fontweight="bold")

    labels = ["Not Readmitted", "Readmitted (<30d)"]
    colors = ["#4C72B0", "#C44E52"]

    for ax, counts, title in [
        (axes[0], counts_before, "Before SMOTE"),
        (axes[1], counts_after,  "After SMOTE"),
    ]:
        vals = [counts.get(0, 0), counts.get(1, 0)]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        total = sum(vals)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.01,
                    f"{val:,}\n({val/total*100:.1f}%)",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.18)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "smote_class_balance.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
    log_message(f"SMOTE plot saved -> {path}", LOG)


# ============================================================
# MAIN
# ============================================================
def run_model_training():
    print("\n" + "=" * 65)
    print("  HOSPITAL READMISSION -- STEP 5: MODEL TRAINING")
    print(f"  Started: {generate_timestamp()}")
    print("=" * 65)
    log_message(f"Model training (Step 5) started: {generate_timestamp()}", LOG)

    X_train, y_train, X_val, y_val, X_test, y_test = section_a_load()
    X_bal, y_bal = section_b_smote(X_train, y_train)

    print("\n" + "=" * 65)
    print("  [DONE] Sections A & B Complete")
    print(f"  Original train  : {len(X_train):,} samples")
    print(f"  Balanced train  : {len(X_bal):,} samples  (after SMOTE)")
    print(f"  Validation      : {len(X_val):,} samples")
    print(f"  Test            : {len(X_test):,} samples")
    print(f"  Finished: {generate_timestamp()}")
    print("=" * 65 + "\n")
    log_message(f"Sections A & B complete: {generate_timestamp()}", LOG)

    return X_bal, y_bal, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    run_model_training()
