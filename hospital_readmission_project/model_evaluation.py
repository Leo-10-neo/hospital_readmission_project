# -*- coding: utf-8 -*-
"""
model_evaluation.py
===================
STEP 6 - Model Evaluation for Hospital Readmission Risk Scorer.

- Loads X_test.csv / y_test.csv
- Loads all saved models from models/
- Computes: Accuracy, Precision, Recall, F1, ROC-AUC, Brier Score, R²
- Plots: ROC curves, Confusion matrix heatmaps, Metric comparison table
- Identifies best model by ROC-AUC and Accuracy
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    roc_curve, classification_report,
)
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR,
    TARGET_COLUMN, VIZ,
)
from utils import log_message, log_section, generate_timestamp, print_step

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
LOG       = os.path.join(LOGS_DIR,    "model_evaluation_log.txt")
PLOT_DIR  = os.path.join(os.path.dirname(PROCESSED_DATA_DIR),
                         "visualizations", "model_plots")
EVAL_CSV  = os.path.join(REPORTS_DIR, "model_evaluation_results.csv")
EVAL_TXT  = os.path.join(REPORTS_DIR, "model_evaluation_report.txt")

X_TEST_CSV = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
Y_TEST_CSV = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")

os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#55A868",
    "Gradient Boosting":   "#C44E52",
    "XGBoost":             "#8172B2",
    "SVM":                 "#CCB974",
}
DEFAULT_COLOR = "#64B5CD"


# ============================================================
# LOAD TEST DATA
# ============================================================
def load_test_data():
    print_step("1", "Load Test Data")
    log_section("LOAD TEST DATA", LOG)

    X_test = pd.read_csv(X_TEST_CSV)
    y_test = pd.read_csv(Y_TEST_CSV)

    # Extract target series
    if TARGET_COLUMN in y_test.columns:
        y_test = y_test[TARGET_COLUMN]
    else:
        y_test = y_test.iloc[:, 0]

    y_test = y_test.astype(int)

    print(f"  X_test shape : {X_test.shape}")
    print(f"  y_test shape : {y_test.shape}")
    print(f"  Positive rate: {y_test.mean()*100:.2f}%  "
          f"({y_test.sum():,} readmitted / {len(y_test):,} total)")

    log_message(f"X_test {X_test.shape} | y_test {y_test.shape} | "
                f"pos_rate={y_test.mean():.4f}", LOG)
    return X_test, y_test


# ============================================================
# LOAD MODELS
# ============================================================
def load_models():
    print_step("2", "Load Saved Models")
    log_section("LOAD MODELS", LOG)

    model_files = [f for f in os.listdir(MODELS_DIR)
                   if f.endswith(".pkl") and f != "scaler.pkl"]
    models = {}
    for fname in sorted(model_files):
        path = os.path.join(MODELS_DIR, fname)
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        # Pretty name from filename
        name = fname.replace(".pkl", "").replace("_", " ").title()
        models[name] = pipeline
        print(f"  Loaded: {name}  ({fname})")
        log_message(f"Loaded model: {name} from {fname}", LOG)

    print(f"\n  Total models loaded: {len(models)}")
    return models


# ============================================================
# EVALUATE ALL MODELS
# ============================================================
def evaluate_models(models, X_test, y_test):
    print_step("3", "Evaluate All Models")
    log_section("MODEL EVALUATION", LOG)

    results = {}
    for name, pipeline in models.items():
        print(f"\n  Evaluating: {name} ...")

        y_pred = pipeline.predict(X_test)
        y_prob = (pipeline.predict_proba(X_test)[:, 1]
                  if hasattr(pipeline, "predict_proba") else None)

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        auc   = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        brier = brier_score_loss(y_test, y_prob) if y_prob is not None else np.nan
        r2    = r2_score(y_test, y_prob)   if y_prob is not None else np.nan
        cm    = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "roc_auc":   auc,
            "brier":     brier,
            "r2":        r2,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "cm":        cm,
        }

        tn, fp, fn, tp = cm.ravel()
        print(f"    Accuracy  : {acc:.4f}   Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}   F1-Score  : {f1:.4f}")
        print(f"    ROC-AUC   : {auc:.4f}   Brier     : {brier:.4f}   R²: {r2:.4f}")
        print(f"    TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

        log_message(f"{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
                    f"f1={f1:.4f} auc={auc:.4f} brier={brier:.4f} r2={r2:.4f}", LOG)

    return results


# ============================================================
# CONFUSION MATRIX HEATMAPS
# ============================================================
def plot_confusion_matrices(results, y_test):
    print_step("4", "Confusion Matrix Heatmaps")

    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    try:
        plt.style.use(VIZ["style"])
    except Exception:
        pass

    for i, (name, res) in enumerate(results.items()):
        cm = res["cm"]
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                            [f"FN\n{fn:,}", f"TP\n{tp:,}"]])

        color = PALETTE.get(name, DEFAULT_COLOR)
        cmap  = sns.light_palette(color, as_cmap=True)

        sns.heatmap(cm, annot=labels, fmt="", ax=axes[i],
                    cmap=cmap, linewidths=2, linecolor="white",
                    xticklabels=["Not Readmitted", "Readmitted"],
                    yticklabels=["Not Readmitted", "Readmitted"],
                    cbar=False, annot_kws={"size": 12, "weight": "bold"})

        axes[i].set_title(name, fontsize=13, fontweight="bold", pad=10)
        axes[i].set_xlabel("Predicted", fontsize=10)
        axes[i].set_ylabel("Actual",    fontsize=10)
        axes[i].tick_params(labelsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrix Heatmaps — Test Set",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "eval_confusion_matrices.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    log_message(f"Confusion matrices saved -> {path}", LOG)


# ============================================================
# ROC CURVES
# ============================================================
def plot_roc_curves(results, y_test):
    print_step("5", "ROC Curve Plot")

    try:
        plt.style.use(VIZ["style"])
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, res in results.items():
        if res["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["roc_auc"]
        color = PALETTE.get(name, DEFAULT_COLOR)
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random Baseline (AUC = 0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models (Test Set)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "eval_roc_curves.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    log_message(f"ROC curves saved -> {path}", LOG)


# ============================================================
# METRIC COMPARISON TABLE PLOT
# ============================================================
def plot_metric_comparison(results):
    print_step("6", "Metric Comparison Table")

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier", "r2"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score",
               "ROC-AUC", "Brier Score", "R²"]

    names = list(results.keys())
    x     = np.arange(len(metrics))
    width = 0.8 / len(names)

    try:
        plt.style.use(VIZ["style"])
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, name in enumerate(names):
        vals  = [results[name].get(m, 0) or 0 for m in metrics]
        color = PALETTE.get(name, DEFAULT_COLOR)
        offset = x + i * width - (len(names) - 1) * width / 2
        bars  = ax.bar(offset, vals, width * 0.88,
                       label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Metric Comparison — Test Set",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "eval_metric_comparison.png")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    log_message(f"Metric comparison saved -> {path}", LOG)


# ============================================================
# COMPARISON TABLE (CSV + PRINT)
# ============================================================
def save_comparison_table(results):
    print_step("7", "Comparison Table")

    rows = []
    for name, res in results.items():
        rows.append({
            "Model":       name,
            "Accuracy":    round(res["accuracy"],  4),
            "Precision":   round(res["precision"], 4),
            "Recall":      round(res["recall"],    4),
            "F1-Score":    round(res["f1"],        4),
            "ROC-AUC":     round(res["roc_auc"],   4),
            "Brier Score": round(res["brier"],     4),
            "R²":          round(res["r2"],        4),
        })

    df = (pd.DataFrame(rows)
            .sort_values("ROC-AUC", ascending=False)
            .reset_index(drop=True))

    df.to_csv(EVAL_CSV, index=False)

    # Pretty print
    print(f"\n  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'AUC':>7} {'Brier':>7} {'R²':>7}")
    print(f"  {'-'*73}")
    for _, row in df.iterrows():
        print(f"  {row['Model']:<22} {row['Accuracy']:>7.4f} "
              f"{row['Precision']:>7.4f} {row['Recall']:>7.4f} "
              f"{row['F1-Score']:>7.4f} {row['ROC-AUC']:>7.4f} "
              f"{row['Brier Score']:>7.4f} {row['R²']:>7.4f}")

    print(f"\n  Saved: {EVAL_CSV}")
    log_message(f"Comparison table saved -> {EVAL_CSV}", LOG)
    return df


# ============================================================
# BEST MODEL IDENTIFICATION
# ============================================================
def identify_best_model(df, results):
    print_step("8", "Best Model Identification")

    best_auc = df.loc[df["ROC-AUC"].idxmax(), "Model"]
    best_acc = df.loc[df["Accuracy"].idxmax(), "Model"]

    # Combined rank: rank by AUC + rank by accuracy, pick lowest sum
    df["rank_auc"] = df["ROC-AUC"].rank(ascending=False)
    df["rank_acc"] = df["Accuracy"].rank(ascending=False)
    df["combined_rank"] = df["rank_auc"] + df["rank_acc"]
    overall_best = df.loc[df["combined_rank"].idxmin(), "Model"]

    print(f"\n  Best by ROC-AUC  : {best_auc}  ({df.loc[df.Model==best_auc,'ROC-AUC'].values[0]:.4f})")
    print(f"  Best by Accuracy : {best_acc}  ({df.loc[df.Model==best_acc,'Accuracy'].values[0]:.4f})")
    print(f"\n  ★  OVERALL BEST MODEL: {overall_best}")
    print(f"     (Lowest combined rank across ROC-AUC + Accuracy)")

    # Print its full metrics
    r = results[overall_best]
    print(f"\n  {overall_best} — Test Set Metrics:")
    print(f"    Accuracy  : {r['accuracy']:.4f}")
    print(f"    Precision : {r['precision']:.4f}")
    print(f"    Recall    : {r['recall']:.4f}")
    print(f"    F1-Score  : {r['f1']:.4f}")
    print(f"    ROC-AUC   : {r['roc_auc']:.4f}")
    print(f"    Brier     : {r['brier']:.4f}")
    print(f"    R²        : {r['r2']:.4f}")

    log_message(f"Best by AUC: {best_auc} | Best by Acc: {best_acc} | "
                f"Overall best: {overall_best}", LOG)
    return overall_best


# ============================================================
# SAVE REPORT
# ============================================================
def save_report(df, overall_best, results):
    print_step("9", "Save Evaluation Report")

    lines = [
        "=" * 70,
        "  HOSPITAL READMISSION — MODEL EVALUATION REPORT (STEP 6)",
        f"  Generated : {generate_timestamp()}",
        "=" * 70,
        "",
        "TEST SET",
        f"  File       : X_test.csv / y_test.csv",
        f"  Samples    : {sum(r['cm'].sum() for r in [list(results.values())[0]])//1}",
        "",
        "MODELS EVALUATED",
    ]
    for name in results:
        lines.append(f"  - {name}")

    lines += ["", "METRIC DEFINITIONS",
              "  Accuracy    : (TP+TN) / Total",
              "  Precision   : TP / (TP+FP) — how many predicted positives are correct",
              "  Recall      : TP / (TP+FN) — how many actual positives were caught",
              "  F1-Score    : Harmonic mean of Precision & Recall",
              "  ROC-AUC     : Area under ROC curve (higher = better discriminator)",
              "  Brier Score : Mean squared error of probability predictions (lower = better)",
              "  R²          : Goodness-of-fit of predicted probabilities vs actual labels",
              "", "COMPARISON TABLE (sorted by ROC-AUC)",
              df[["Model","Accuracy","Precision","Recall","F1-Score",
                  "ROC-AUC","Brier Score","R²"]].to_string(index=False),
              "", f"★  OVERALL BEST MODEL: {overall_best}",
              f"   (Highest combined rank on ROC-AUC + Accuracy)",
              "", "SAVED OUTPUTS",
              f"  Comparison CSV        : {EVAL_CSV}",
              f"  Confusion matrices    : {PLOT_DIR}/eval_confusion_matrices.png",
              f"  ROC curves            : {PLOT_DIR}/eval_roc_curves.png",
              f"  Metric comparison     : {PLOT_DIR}/eval_metric_comparison.png",
              "=" * 70,
    ]

    report = "\n".join(lines)
    with open(EVAL_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved: {EVAL_TXT}")
    log_message(f"Evaluation report saved -> {EVAL_TXT}", LOG)


# ============================================================
# MAIN
# ============================================================
def run_model_evaluation():
    print("\n" + "=" * 65)
    print("  HOSPITAL READMISSION -- STEP 6: MODEL EVALUATION")
    print(f"  Started: {generate_timestamp()}")
    print("=" * 65)
    log_message(f"Model evaluation started: {generate_timestamp()}", LOG)

    X_test, y_test = load_test_data()
    models          = load_models()
    results         = evaluate_models(models, X_test, y_test)

    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_metric_comparison(results)

    df           = save_comparison_table(results)
    overall_best = identify_best_model(df, results)
    save_report(df, overall_best, results)

    print("\n" + "=" * 65)
    print(f"  [DONE] STEP 6: MODEL EVALUATION COMPLETE")
    print(f"  Models evaluated : {len(models)}")
    print(f"  Best model       : {overall_best}")
    print(f"  Report           : {EVAL_TXT}")
    print(f"  Plots            : {PLOT_DIR}")
    print(f"  Finished: {generate_timestamp()}")
    print("=" * 65 + "\n")
    log_message(f"Model evaluation complete: {generate_timestamp()}", LOG)


if __name__ == "__main__":
    run_model_evaluation()
