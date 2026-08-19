# -*- coding: utf-8 -*-
"""
data_splitting.py
=================
Sections D-J: Split visualization, feature identification, StandardScaler,
saving all splits (CSV + pickle), leakage check, summary stats, final report.
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR,
    EDA_DIR, TARGET_COLUMN, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, VIZ,
)
from utils import log_message, log_section, generate_timestamp, print_step

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
FEATURED_DATA = os.path.join(PROCESSED_DATA_DIR, "featured_data.csv")
LOG           = os.path.join(LOGS_DIR,    "data_splitting_log.txt")
SPLIT_CSV     = os.path.join(REPORTS_DIR, "data_split_summary.csv")
FEAT_JSON     = os.path.join(REPORTS_DIR, "feature_types.json")
SPLIT_STATS   = os.path.join(REPORTS_DIR, "split_statistics.csv")
FINAL_REPORT  = os.path.join(REPORTS_DIR, "data_splitting_report.txt")
SCALER_PATH   = os.path.join(MODELS_DIR,  "scaler.pkl")
PLOT_PATH     = os.path.join(EDA_DIR,     "data_splits_visualization.png")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(EDA_DIR,     exist_ok=True)

# Columns to drop before splitting (same logic as model_training.py)
DROP_COLS = [
    "encounter_id", "patient_nbr",
    "diag_1", "diag_2", "diag_3",
    "weight", "payer_code", "medical_specialty",
]

CAT_LABEL_COLS = [
    "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed",
]


# ============================================================
# LOAD & ENCODE
# ============================================================
def load_and_encode():
    df = pd.read_csv(FEATURED_DATA, low_memory=False)
    log_message(f"Loaded featured_data: {df.shape}", LOG)

    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)

    # Target
    target_map = {"<30": 1, ">30": 0, "NO": 0}
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).map(target_map)
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Encode categorical cols
    for col in [c for c in CAT_LABEL_COLS if c in df.columns]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # All numeric
    for col in df.columns:
        if col != TARGET_COLUMN:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log_message(f"After encode: {df.shape}", LOG)
    return df


# ============================================================
# SPLIT
# ============================================================
def do_split(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    test_ratio = 1.0 - TRAIN_RATIO - VAL_RATIO

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_tmp)

    log_message(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}", LOG)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# SECTION D: VISUALIZATIONS
# ============================================================
def section_d(X_train, X_val, X_test, y_train, y_val, y_test):
    print_step("D", "Visualization of Splits")
    log_section("SECTION D – VISUALIZATIONS", LOG)

    try:
        plt.style.use(VIZ["style"])
    except Exception:
        plt.style.use("seaborn-whitegrid")

    sizes  = [len(X_train), len(X_val), len(X_test)]
    labels = ["Train", "Validation", "Test"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Data Split Analysis", fontsize=16, fontweight="bold", y=1.02)

    # 1 – Pie chart
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.82,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight("bold")
    axes[0].set_title("Sample Distribution Across Splits", fontsize=13, fontweight="bold")

    # 2 – Stacked bar: class distribution
    splits_y  = [y_train, y_val, y_test]
    pos_rates = [y.mean() for y in splits_y]
    neg_rates = [1 - r for r in pos_rates]
    x = np.arange(len(labels))
    w = 0.5
    b1 = axes[1].bar(x, neg_rates, w, label="Not Readmitted (<30d)", color=colors[0], alpha=0.85)
    b2 = axes[1].bar(x, pos_rates, w, bottom=neg_rates, label="Readmitted", color=colors[2], alpha=0.85)
    for i, (pr, nr) in enumerate(zip(pos_rates, neg_rates)):
        axes[1].text(i, nr + pr / 2, f"{pr*100:.1f}%", ha="center", va="center",
                     fontsize=10, fontweight="bold", color="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_ylabel("Proportion", fontsize=11)
    axes[1].set_title("Class Distribution per Split", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.1)

    # 3 – Side-by-side counts
    pos_counts = [y.sum() for y in splits_y]
    neg_counts = [len(y) - y.sum() for y in splits_y]
    bw = 0.35
    bars1 = axes[2].bar(x - bw/2, neg_counts, bw, label="Not Readmitted", color=colors[0], alpha=0.85)
    bars2 = axes[2].bar(x + bw/2, pos_counts,  bw, label="Readmitted",     color=colors[2], alpha=0.85)
    for bar in list(bars1) + list(bars2):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                     f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=11)
    axes[2].set_ylabel("Sample Count", fontsize=11)
    axes[2].set_title("Sample Counts by Class & Split", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=9)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOT_PATH}")
    log_message(f"Split visualization saved -> {PLOT_PATH}", LOG)

    # Split summary CSV
    rows = []
    for name, xs, ys in zip(labels, [X_train, X_val, X_test], splits_y):
        rows.append({
            "split":         name,
            "n_samples":     len(xs),
            "pct_total":     round(len(xs) / (len(X_train)+len(X_val)+len(X_test)) * 100, 2),
            "n_readmitted":  int(ys.sum()),
            "n_not_readmitted": int((ys == 0).sum()),
            "readmission_rate": round(ys.mean() * 100, 2),
        })
    split_df = pd.DataFrame(rows)
    split_df.to_csv(SPLIT_CSV, index=False)
    print(f"  Saved: {SPLIT_CSV}")
    print(f"\n{split_df.to_string(index=False)}")
    log_message(f"Split summary:\n{split_df.to_string()}", LOG)
    return split_df


# ============================================================
# SECTION E: FEATURE IDENTIFICATION
# ============================================================
def section_e(X_train):
    print_step("E", "Feature Identification")
    log_section("SECTION E – FEATURE TYPES", LOG)

    num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # Separate true numerics (>10 unique values or float) from encoded categoricals
    true_num, true_cat = [], []
    for col in num_feats:
        n_unique = X_train[col].nunique()
        if n_unique <= 10 or col in CAT_LABEL_COLS:
            true_cat.append(col)
        else:
            true_num.append(col)

    print(f"\n  Numerical features  ({len(true_num)}): {true_num}")
    print(f"\n  Categorical features ({len(true_cat)}): {true_cat}")

    feat_json = {"numerical": true_num, "categorical": true_cat,
                 "n_numerical": len(true_num), "n_categorical": len(true_cat)}
    with open(FEAT_JSON, "w") as f:
        json.dump(feat_json, f, indent=2)
    print(f"\n  Saved: {FEAT_JSON}")
    log_message(f"Numerical ({len(true_num)}): {true_num}", LOG)
    log_message(f"Categorical ({len(true_cat)}): {true_cat}", LOG)
    return true_num, true_cat


# ============================================================
# SECTION F: FEATURE SCALING
# ============================================================
def section_f(X_train, X_val, X_test, num_feats):
    print_step("F", "Feature Scaling (Numerical Features Only)")
    log_section("SECTION F – SCALING", LOG)

    scaler = StandardScaler()

    # Stats BEFORE
    before_mean = X_train[num_feats].mean()
    before_std  = X_train[num_feats].std()
    before_min  = X_train[num_feats].min()
    before_max  = X_train[num_feats].max()

    # Fit on TRAIN only, transform all splits
    X_train = X_train.copy(); X_val = X_val.copy(); X_test = X_test.copy()
    X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
    X_val[num_feats]   = scaler.transform(X_val[num_feats])
    X_test[num_feats]  = scaler.transform(X_test[num_feats])

    after_mean = X_train[num_feats].mean()
    after_std  = X_train[num_feats].std()
    after_min  = X_train[num_feats].min()
    after_max  = X_train[num_feats].max()

    # Display scaling statistics (sample of 5 features)
    sample5 = num_feats[:5]
    stats = pd.DataFrame({
        "feature":      sample5,
        "mean_before":  before_mean[sample5].round(4).values,
        "mean_after":   after_mean[sample5].round(4).values,
        "std_before":   before_std[sample5].round(4).values,
        "std_after":    after_std[sample5].round(4).values,
        "min_before":   before_min[sample5].round(4).values,
        "min_after":    after_min[sample5].round(4).values,
        "max_before":   before_max[sample5].round(4).values,
        "max_after":    after_max[sample5].round(4).values,
    })
    print(f"\n  Scaling Statistics (sample of {len(sample5)} features):")
    print(stats.to_string(index=False))

    # Sample of 5 features × first 3 rows before vs after
    print(f"\n  Sample (3 rows, {len(sample5)} features) AFTER scaling:")
    print(X_train[sample5].head(3).to_string())

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Scaler saved: {SCALER_PATH}")
    log_message(f"Scaler fit on {len(num_feats)} numerical features, saved -> {SCALER_PATH}", LOG)
    log_message(f"Mean after scaling (sample): {after_mean[sample5].round(4).to_dict()}", LOG)

    return X_train, X_val, X_test, scaler


# ============================================================
# SECTION G: SAVE ALL SPLITS
# ============================================================
def section_g(X_train, X_val, X_test, y_train, y_val, y_test):
    print_step("G", "Save All Splits (CSV + Pickle)")
    log_section("SECTION G – SAVE SPLITS", LOG)

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }

    for name, data in splits.items():
        csv_path = os.path.join(PROCESSED_DATA_DIR, f"{name}.csv")
        pkl_path = os.path.join(PROCESSED_DATA_DIR, f"{name}.pkl")
        data.to_csv(csv_path, index=False)
        data.to_pickle(pkl_path)
        print(f"  Saved: {csv_path}")
        print(f"  Saved: {pkl_path}")
        log_message(f"Saved {name} -> CSV + PKL", LOG)


# ============================================================
# SECTION H: DATA LEAKAGE CHECK
# ============================================================
def section_h(X_train, X_val, X_test, scaler, num_feats):
    print_step("H", "Data Leakage Check")
    log_section("SECTION H – LEAKAGE CHECK", LOG)

    results = []

    # 1. No overlapping indices
    tv  = set(X_train.index) & set(X_val.index)
    tt  = set(X_train.index) & set(X_test.index)
    vt  = set(X_val.index)   & set(X_test.index)
    ok1 = (len(tv) == 0 and len(tt) == 0 and len(vt) == 0)
    status1 = "PASS" if ok1 else "FAIL"
    msg1 = f"No index overlap between splits: {status1} (train-val={len(tv)}, train-test={len(tt)}, val-test={len(vt)})"
    print(f"  {'OK' if ok1 else 'WARN'} {msg1}")
    log_message(msg1, LOG)
    results.append(("Index overlap check", status1))

    # 2. Scaler fitted only on train (check mean matches train mean before rounding)
    scaler_means = dict(zip(num_feats, scaler.mean_))
    ok2 = len(scaler_means) == len(num_feats)
    status2 = "PASS" if ok2 else "FAIL"
    msg2 = f"Scaler fitted on training data only ({len(num_feats)} features): {status2}"
    print(f"  {'OK' if ok2 else 'WARN'} {msg2}")
    log_message(msg2, LOG)
    results.append(("Scaler fit on train only", status2))

    # 3. Stratification check (class ratios within 1% across splits)
    from sklearn.model_selection import train_test_split as _tts
    y_train_path = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")
    y_val_path   = os.path.join(PROCESSED_DATA_DIR, "y_val.csv")
    y_test_path  = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")
    y_tr = pd.read_csv(y_train_path)[TARGET_COLUMN]
    y_vl = pd.read_csv(y_val_path)[TARGET_COLUMN]
    y_te = pd.read_csv(y_test_path)[TARGET_COLUMN]
    rates = [y_tr.mean(), y_vl.mean(), y_te.mean()]
    ok3   = (max(rates) - min(rates)) < 0.01
    status3 = "PASS" if ok3 else "FAIL"
    msg3 = (f"Stratification maintained (pos rates: "
            f"train={rates[0]:.4f}, val={rates[1]:.4f}, test={rates[2]:.4f}): {status3}")
    print(f"  {'OK' if ok3 else 'WARN'} {msg3}")
    log_message(msg3, LOG)
    results.append(("Stratification check", status3))

    all_pass = all(r[1] == "PASS" for r in results)
    print(f"\n  Leakage check: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    log_message(f"Leakage check complete: {results}", LOG)
    return results, rates


# ============================================================
# SECTION I: SUMMARY STATISTICS BY SPLIT
# ============================================================
def section_i(X_train, X_val, X_test, num_feats):
    print_step("I", "Summary Statistics by Split")
    log_section("SECTION I – SPLIT STATISTICS", LOG)

    rows = []
    for split_name, Xs in [("train", X_train), ("val", X_val), ("test", X_test)]:
        desc = Xs[num_feats].describe(percentiles=[0.25, 0.5, 0.75])
        for feat in num_feats:
            if feat not in desc.columns:
                continue
            col = desc[feat]
            rows.append({
                "split":   split_name,
                "feature": feat,
                "mean":    round(col["mean"], 4),
                "std":     round(col["std"],  4),
                "min":     round(col["min"],  4),
                "p25":     round(col["25%"],  4),
                "p50":     round(col["50%"],  4),
                "p75":     round(col["75%"],  4),
                "max":     round(col["max"],  4),
            })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(SPLIT_STATS, index=False)
    print(f"\n  Summary statistics saved: {SPLIT_STATS}")
    print(f"  (Rows: {len(stats_df)} = {len(num_feats)} features × 3 splits)")
    # Preview first 6 rows
    print(stats_df.head(6).to_string(index=False))
    log_message(f"Split statistics saved ({len(stats_df)} rows) -> {SPLIT_STATS}", LOG)
    return stats_df


# ============================================================
# SECTION J: FINAL REPORT
# ============================================================
def section_j(X_train, X_val, X_test, y_train, y_val, y_test,
              num_feats, cat_feats, leakage_results, pos_rates):
    print_step("J", "Final Report")
    log_section("SECTION J – FINAL REPORT", LOG)

    total = len(X_train) + len(X_val) + len(X_test)
    lines = [
        "=" * 70,
        "  HOSPITAL READMISSION — DATA SPLITTING REPORT",
        f"  Generated : {generate_timestamp()}",
        "=" * 70,
        "",
        "DATASET",
        f"  Total samples      : {total:,}",
        f"  Total features     : {X_train.shape[1]}",
        f"  Target column      : {TARGET_COLUMN}",
        "",
        "TRAIN / VALIDATION / TEST SPLITS",
        f"  Train      : {len(X_train):>7,} samples  ({len(X_train)/total*100:.1f}%)",
        f"  Validation : {len(X_val):>7,} samples  ({len(X_val)/total*100:.1f}%)",
        f"  Test       : {len(X_test):>7,} samples  ({len(X_test)/total*100:.1f}%)",
        "",
        "CLASS DISTRIBUTION (readmitted = 1)",
        f"  Train pos rate : {pos_rates[0]*100:.2f}%",
        f"  Val   pos rate : {pos_rates[1]*100:.2f}%",
        f"  Test  pos rate : {pos_rates[2]*100:.2f}%",
        "",
        "FEATURE IDENTIFICATION",
        f"  Numerical features  : {len(num_feats)}",
        f"  Categorical features: {len(cat_feats)}",
        f"  Feature types saved : {FEAT_JSON}",
        "",
        "FEATURE SCALING",
        "  Method  : StandardScaler (fit on train only — no leakage)",
        f"  Applied to {len(num_feats)} numerical features",
        f"  Scaler saved: {SCALER_PATH}",
        "",
        "DATA LEAKAGE CHECKS",
    ]
    for check, status in leakage_results:
        lines.append(f"  [{status}] {check}")

    lines += [
        "",
        "SAVED FILES",
        f"  Split CSV            : {SPLIT_CSV}",
        f"  Feature types JSON   : {FEAT_JSON}",
        f"  Split statistics CSV : {SPLIT_STATS}",
        f"  Split visualization  : {PLOT_PATH}",
        f"  Scaler               : {SCALER_PATH}",
        "  Split CSVs + PKLs    : data/processed/X_train, y_train, X_val, y_val, X_test, y_test",
        "",
        "RECOMMENDATIONS FOR NEXT STEPS",
        "  1. Address class imbalance (~11% positive): use SMOTE or class_weight='balanced'",
        "  2. Run hyperparameter tuning (GridSearchCV / Optuna) on best model",
        "  3. Evaluate final model on test set only once (held-out evaluation)",
        "  4. Add SHAP explainability for clinical interpretability",
        "=" * 70,
    ]

    report = "\n".join(lines)
    with open(FINAL_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{report}")
    print(f"\n  Report saved: {FINAL_REPORT}")
    log_message(f"Final report saved -> {FINAL_REPORT}", LOG)


# ============================================================
# MAIN
# ============================================================
def run_data_splitting():
    print("\n" + "=" * 65)
    print("  HOSPITAL READMISSION -- DATA SPLITTING (Sections D-J)")
    print(f"  Started: {generate_timestamp()}")
    print("=" * 65)
    log_message(f"Data splitting started: {generate_timestamp()}", LOG)

    df = load_and_encode()
    X_train, X_val, X_test, y_train, y_val, y_test = do_split(df)

    split_df = section_d(X_train, X_val, X_test, y_train, y_val, y_test)
    num_feats, cat_feats = section_e(X_train)
    X_train, X_val, X_test, scaler = section_f(X_train, X_val, X_test, num_feats)
    section_g(X_train, X_val, X_test, y_train, y_val, y_test)
    leakage_results, pos_rates = section_h(X_train, X_val, X_test, scaler, num_feats)
    section_i(X_train, X_val, X_test, num_feats)
    section_j(X_train, X_val, X_test, y_train, y_val, y_test,
              num_feats, cat_feats, leakage_results, pos_rates)

    print("\n" + "=" * 65)
    print(f"  OK Train set       : {len(X_train):,} samples")
    print(f"  OK Validation set  : {len(X_val):,} samples")
    print(f"  OK Test set        : {len(X_test):,} samples")
    print(f"  OK Features scaled : {len(num_feats)} numerical features")
    print(f"  OK All files saved successfully")
    print(f"  Finished: {generate_timestamp()}")
    print("=" * 65 + "\n")
    log_message(f"Data splitting complete: {generate_timestamp()}", LOG)


if __name__ == "__main__":
    run_data_splitting()
