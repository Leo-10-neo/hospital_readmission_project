# -*- coding: utf-8 -*-
"""
feature_importance.py
=====================
STEP 7 - Feature Importance Analysis for Hospital Readmission Risk Scorer.

- Loads best model (Random Forest) and test data
- Top-15 feature importance bar chart (Gini Importance)
- Permutation Importance (more robust)
- Partial Dependence Plots (PDP) for top 3 features
- All plots saved to visualizations/feature_importance/
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance, PartialDependenceDisplay

warnings.filterwarnings("ignore")

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR,
    VISUALIZATIONS_DIR, TARGET_COLUMN, VIZ,
)
from utils import log_message, log_section, generate_timestamp, print_step

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
LOG      = os.path.join(LOGS_DIR, "feature_importance_log.txt")
FI_DIR   = os.path.join(VISUALIZATIONS_DIR, "feature_importance")
FI_CSV   = os.path.join(REPORTS_DIR, "feature_importance.csv")

X_TEST_CSV = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
Y_TEST_CSV = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")

# Best model determined from Step 6
BEST_MODEL_FILE = "random_forest.pkl"

os.makedirs(FI_DIR,    exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Seaborn palette
COLORS = {
    "top":    "#C44E52",
    "mid":    "#4C72B0",
    "shap":   "#55A868",
    "bar":    "#8172B2",
}


# ============================================================
# LOAD DATA & MODEL
# ============================================================
def load_data_and_model():
    print_step("1", "Load Test Data & Best Model")
    log_section("LOAD DATA & MODEL", LOG)

    X_test = pd.read_csv(X_TEST_CSV)
    y_test = pd.read_csv(Y_TEST_CSV)
    if TARGET_COLUMN in y_test.columns:
        y_test = y_test[TARGET_COLUMN].astype(int)
    else:
        y_test = y_test.iloc[:, 0].astype(int)

    model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILE)
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    model_name = BEST_MODEL_FILE.replace(".pkl", "").replace("_", " ").title()
    print(f"  Model      : {model_name}")
    print(f"  X_test     : {X_test.shape}")
    print(f"  y_test     : {y_test.shape}  (pos rate={y_test.mean():.3f})")
    log_message(f"Loaded model: {model_name} | X_test {X_test.shape}", LOG)

    # Extract the underlying sklearn estimator (last step of pipeline)
    estimator = pipeline.named_steps.get("model", pipeline)

    # Apply imputation if pipeline has imputer (keep X consistent)
    imputer = pipeline.named_steps.get("imputer", None)
    if imputer is not None:
        X_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns)
    else:
        X_imp = X_test.copy()

    feature_names = list(X_test.columns)
    return X_test, X_imp, y_test, estimator, feature_names, model_name


# ============================================================
# BUILT-IN FEATURE IMPORTANCE (Top 15 Bar Chart)
# ============================================================
def plot_top15_importance(estimator, feature_names, model_name):
    print_step("2", "Top-15 Gini Feature Importance")
    log_section("BUILT-IN FEATURE IMPORTANCE", LOG)

    importances = estimator.feature_importances_
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top15 = imp_df.head(15).sort_values("importance", ascending=True)

    # Save full importance CSV
    imp_df.to_csv(FI_CSV, index=False)
    print(f"  Full importance saved: {FI_CSV}")
    log_message(f"Feature importance CSV saved -> {FI_CSV}", LOG)

    # --- Plot ---
    try:
        plt.style.use(VIZ["style"])
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color gradient: darker = more important
    norm_vals = (top15["importance"] - top15["importance"].min()) / \
                (top15["importance"].max() - top15["importance"].min() + 1e-9)
    colors = plt.cm.YlOrRd(0.35 + norm_vals * 0.65)

    bars = ax.barh(top15["feature"], top15["importance"],
                   color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, top15["importance"]):
        ax.text(val + top15["importance"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Gini Importance (Mean Decrease in Impurity)", fontsize=12)
    ax.set_title(f"Top-15 Predictors of Hospital Readmission\n({model_name})",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, top15["importance"].max() * 1.18)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(FI_DIR, "top15_gini_importance.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    log_message(f"Top-15 Gini chart saved -> {path}", LOG)

    return imp_df


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================
def plot_permutation_importance(estimator, X_imp, y_test, feature_names, model_name):
    print_step("3", "Permutation Importance (on Test Set)")
    log_section("PERMUTATION IMPORTANCE", LOG)

    # Use a sample to speed up (permutation importance can be slow)
    sample_size = min(1000, len(X_imp))
    X_samp = X_imp.sample(sample_size, random_state=42)
    y_samp = y_test.loc[X_samp.index]

    print(f"  Computing permutation importance on {sample_size} samples...")
    result = permutation_importance(
        estimator, X_samp, y_samp, n_repeats=5, random_state=42, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    top15_perm = perm_df.head(15).sort_values("importance_mean", ascending=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    norm_vals = (top15_perm["importance_mean"] - top15_perm["importance_mean"].min()) / \
                (top15_perm["importance_mean"].max() - top15_perm["importance_mean"].min() + 1e-9)
    colors = plt.cm.Greens(0.35 + norm_vals * 0.65)

    ax.barh(top15_perm["feature"], top15_perm["importance_mean"],
            xerr=top15_perm["importance_std"], color=colors, edgecolor="white")

    ax.set_xlabel("Mean Accuracy Decrease", fontsize=12)
    ax.set_title(f"Permutation Importance — Top 15\n({model_name})",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(FI_DIR, "permutation_importance.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    log_message(f"Permutation importance chart saved -> {path}", LOG)

    return perm_df


# ============================================================
# PARTIAL DEPENDENCE PLOTS (PDP)
# ============================================================
def plot_pdp(estimator, X_imp, feature_names, top_features, model_name):
    print_step("4", "Partial Dependence Plots (Top Features)")
    log_section("PARTIAL DEPENDENCE PLOTS", LOG)

    # Limit to top 3-4 features
    features_to_plot = top_features[:4]
    
    print(f"  Generating PDP for: {features_to_plot}")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    PartialDependenceDisplay.from_estimator(
        estimator, X_imp, features_to_plot, ax=ax, grid_resolution=50
    )
    
    fig.suptitle(f"Partial Dependence Plots\n({model_name})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    path = os.path.join(FI_DIR, "partial_dependence_plots.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    log_message(f"Partial dependence plots saved -> {path}", LOG)


# ============================================================
# MAIN
# ============================================================
def run_feature_importance():
    print("\n" + "=" * 65)
    print("  HOSPITAL READMISSION -- STEP 7: FEATURE IMPORTANCE")
    print(f"  Started: {generate_timestamp()}")
    print("=" * 65)
    log_message(f"Feature importance started: {generate_timestamp()}", LOG)

    X_test, X_imp, y_test, estimator, feature_names, model_name = load_data_and_model()

    # 1. Gini Importance
    imp_df = plot_top15_importance(estimator, feature_names, model_name)

    # 2. Permutation Importance
    perm_df = plot_permutation_importance(estimator, X_imp, y_test, feature_names, model_name)

    # 3. Partial Dependence Plots
    top_features = perm_df["feature"].tolist()
    plot_pdp(estimator, X_imp, feature_names, top_features, model_name)

    # Summary
    print(f"\n  Top 5 Predictors (Permutation):")
    for i, feat in enumerate(top_features[:5]):
        print(f"    {i+1}. {feat}")

    print("\n" + "=" * 65)
    print("  [DONE] STEP 7: FEATURE IMPORTANCE COMPLETE")
    print(f"  Finished: {generate_timestamp()}")
    print("=" * 65 + "\n")
    log_message(f"Feature importance complete: {generate_timestamp()}", LOG)


if __name__ == "__main__":
    run_feature_importance()
