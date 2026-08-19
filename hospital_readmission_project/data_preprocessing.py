"""
data_preprocessing.py
=====================
Full preprocessing pipeline for the Hospital Readmission dataset.
Parts A–I as specified in the project brief.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
from config import (
    RAW_DATA_FILE, PROCESSED_DATA_DIR,
    DATA_AFTER_IMPUTATION, DATA_DEDUPLICATED,
    DATA_OUTLIERS_HANDLED, PREPROCESSED_DATA,
    DATA_QUALITY_INITIAL, ENCODING_MAPPINGS,
    PREPROCESSING_SUMMARY, PREPROCESSING_LOG,
    EDA_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    COLUMNS_TO_DROP, TARGET_COLUMN,
    CAPPING_LOWER_PCTILE, CAPPING_UPPER_PCTILE, VIZ,
)
from utils import (
    create_directories, save_data, load_data,
    log_message, log_section, save_json,
    plot_save, new_figure,
    print_step, print_completion, generate_timestamp,
)

LOG = PREPROCESSING_LOG   # shorthand


# ════════════════════════════════════════════════════════════════════════════
#  PART A — DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def part_a_load_data() -> pd.DataFrame:
    """Load dataset and display basic metadata."""
    print_step("A", "Data Loading")
    log_section("PART A — DATA LOADING", LOG)

    df = load_data(RAW_DATA_FILE, na_values=["?", "Unknown", ""])

    info_lines = [
        f"Dataset shape  : {df.shape[0]} rows × {df.shape[1]} columns",
        f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB",
        "Data types:",
    ]
    for col, dtype in df.dtypes.items():
        info_lines.append(f"  {col:<40s} {dtype}")

    for line in info_lines:
        log_message(line, LOG)

    print("\n".join(info_lines))
    print("\nFirst 10 rows:")
    print(df.head(10).to_string())

    log_message("First 10 rows logged.", LOG)
    log_message(df.head(10).to_string(), LOG)

    return df


# ════════════════════════════════════════════════════════════════════════════
#  PART B — INITIAL DATA QUALITY CHECK
# ════════════════════════════════════════════════════════════════════════════

def part_b_quality_check(df: pd.DataFrame) -> None:
    """Missing values, duplicates, basic statistics."""
    print_step("B", "Initial Data Quality Check")
    log_section("PART B — INITIAL DATA QUALITY CHECK", LOG)

    # Missing values
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)
    missing_df    = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct":   missing_pct,
    }).sort_values("missing_pct", ascending=False)

    print("\nMissing values (top 20):")
    print(missing_df[missing_df["missing_count"] > 0].head(20).to_string())
    log_message(missing_df.to_string(), LOG)

    # Duplicates
    dup_count = df.duplicated().sum()
    log_message(f"Duplicate rows: {dup_count}", LOG)
    print(f"\nDuplicate rows: {dup_count}")

    # Basic statistics — save
    stats = df.describe(include="all").T
    os.makedirs(os.path.dirname(DATA_QUALITY_INITIAL), exist_ok=True)
    stats.to_csv(DATA_QUALITY_INITIAL)
    log_message(f"Basic stats saved → {DATA_QUALITY_INITIAL}", LOG)

    # Missing value heatmap (before)
    _save_missing_heatmap(df, "missing_heatmap_before.png")


def _save_missing_heatmap(df: pd.DataFrame, filename: str) -> None:
    cols_with_na = df.columns[df.isnull().any()].tolist()
    if not cols_with_na:
        log_message("No missing values — skipping heatmap.", LOG)
        return

    sample = df[cols_with_na].isnull().astype(int)
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)

    fig, ax = plt.subplots(figsize=VIZ["figure_size"])
    sns.heatmap(sample, cbar=False, cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Value Heatmap", fontsize=VIZ["title_fontsize"])
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows (sample)")
    plot_save(fig, filename, EDA_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  PART C — MISSING VALUE TREATMENT
# ════════════════════════════════════════════════════════════════════════════

def part_c_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation for numerical; mode for categorical."""
    print_step("C", "Missing Value Treatment")
    log_section("PART C — MISSING VALUE TREATMENT", LOG)

    before = df.isnull().sum()
    imputation_log = {}

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    for col in num_cols:
        if df[col].isnull().any():
            val = df[col].median()
            df[col].fillna(val, inplace=True)
            imputation_log[col] = {"strategy": "median", "value": val}

    for col in cat_cols:
        if df[col].isnull().any():
            val = df[col].mode()[0]
            df[col].fillna(val, inplace=True)
            imputation_log[col] = {"strategy": "mode", "value": str(val)}

    after = df.isnull().sum()
    comparison = pd.DataFrame({"before": before, "after": after})
    comparison = comparison[comparison["before"] > 0]

    print("\nImputation summary:")
    print(comparison.to_string())

    for col, info in imputation_log.items():
        log_message(f"  {col}: {info['strategy']} → {info['value']}", LOG)

    save_data(df, DATA_AFTER_IMPUTATION)
    print_completion([DATA_AFTER_IMPUTATION])
    return df


# ════════════════════════════════════════════════════════════════════════════
#  PART D — DATA TYPE CORRECTIONS
# ════════════════════════════════════════════════════════════════════════════

def part_d_dtype_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to correct dtypes."""
    print_step("D", "Data Type Corrections")
    log_section("PART D — DATA TYPE CORRECTIONS", LOG)

    conversions = []

    # Force known numerical columns to numeric
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            original = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if str(original) != str(df[col].dtype):
                conversions.append(f"{col}: {original} → {df[col].dtype}")

    # Force known categorical columns to object
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(str)
            conversions.append(f"{col}: numeric → object (categorical)")

    if conversions:
        for c in conversions:
            log_message(f"  Converted — {c}", LOG)
        print("\nConversions performed:")
        print("\n".join(conversions))
    else:
        log_message("No dtype conversions needed.", LOG)
        print("All dtypes already correct.")

    return df


# ════════════════════════════════════════════════════════════════════════════
#  PART E — DUPLICATE REMOVAL
# ════════════════════════════════════════════════════════════════════════════

def part_e_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and remove duplicate rows."""
    print_step("E", "Duplicate Removal")
    log_section("PART E — DUPLICATE REMOVAL", LOG)

    dup_mask  = df.duplicated()
    dup_count = dup_mask.sum()

    log_message(f"Duplicate rows found: {dup_count}", LOG)
    print(f"\nDuplicate rows found: {dup_count}")

    if dup_count > 0:
        print("\nSample of duplicates (up to 5):")
        print(df[dup_mask].head(5).to_string())
        df = df.drop_duplicates()
        log_message(f"Removed {dup_count} duplicates. Remaining rows: {len(df)}", LOG)

    save_data(df, DATA_DEDUPLICATED)
    print_completion([DATA_DEDUPLICATED])
    return df


# ════════════════════════════════════════════════════════════════════════════
#  PART F — OUTLIER DETECTION & TREATMENT
# ════════════════════════════════════════════════════════════════════════════

def part_f_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """IQR-based outlier detection; cap at 1st/99th percentiles."""
    print_step("F", "Outlier Detection & Treatment")
    log_section("PART F — OUTLIER DETECTION", LOG)

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    outlier_stats = {}

    rows_per_fig = 3
    n_cols_plot  = 3

    for i in range(0, len(num_cols), rows_per_fig * n_cols_plot):
        batch = num_cols[i: i + rows_per_fig * n_cols_plot]
        n_rows = (len(batch) + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows, n_cols_plot,
                                 figsize=(18, 5 * n_rows))
        axes = np.array(axes).flatten()

        for j, col in enumerate(batch):
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo  = Q1 - 1.5 * IQR
            hi  = Q3 + 1.5 * IQR
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()

            axes[j].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                            boxprops=dict(facecolor="#4C72B0", alpha=0.6))
            axes[j].set_title(f"{col}\n({n_out} outliers)", fontsize=9)
            axes[j].set_xlabel("")

            # Cap
            lo_cap = df[col].quantile(CAPPING_LOWER_PCTILE / 100)
            hi_cap = df[col].quantile(CAPPING_UPPER_PCTILE / 100)
            df[col] = df[col].clip(lower=lo_cap, upper=hi_cap)

            outlier_stats[col] = {
                "Q1": round(Q1, 4), "Q3": round(Q3, 4), "IQR": round(IQR, 4),
                "lower_fence": round(lo, 4), "upper_fence": round(hi, 4),
                "outliers_found": int(n_out),
                "cap_lower": round(lo_cap, 4), "cap_upper": round(hi_cap, 4),
            }

            log_message(f"  {col}: {n_out} outliers → capped [{lo_cap:.3f}, {hi_cap:.3f}]", LOG)

        for k in range(len(batch), len(axes)):
            axes[k].set_visible(False)

        fig.suptitle("Boxplots — Outlier Detection", fontsize=13, y=1.01)
        plt.tight_layout()
        plot_save(fig, f"boxplots_batch_{i // (rows_per_fig * n_cols_plot) + 1}.png", EDA_DIR)

    # Print outlier table
    out_df = pd.DataFrame(outlier_stats).T
    print("\nOutlier statistics:")
    print(out_df[["Q1", "Q3", "IQR", "outliers_found"]].to_string())

    save_data(df, DATA_OUTLIERS_HANDLED)
    print_completion([DATA_OUTLIERS_HANDLED])
    return df, outlier_stats


# ════════════════════════════════════════════════════════════════════════════
#  PART G — CATEGORICAL ENCODING
# ════════════════════════════════════════════════════════════════════════════

def part_g_encoding(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode binary; one-hot-encode multi-class categories."""
    print_step("G", "Categorical Encoding")
    log_section("PART G — CATEGORICAL ENCODING", LOG)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # Do not encode the target
    if TARGET_COLUMN in cat_cols:
        cat_cols.remove(TARGET_COLUMN)

    encoding_maps = {}
    ohe_cols      = []
    le_cols       = []

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= 2:
            # Binary → Label Encode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoding_maps[col] = {
                "method": "label_encoding",
                "mapping": {str(cls): int(i) for i, cls in enumerate(le.classes_)},
            }
            le_cols.append(col)
            log_message(f"  LabelEncoded  {col}: {encoding_maps[col]['mapping']}", LOG)
        else:
            ohe_cols.append(col)

    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
        for col in ohe_cols:
            encoding_maps[col] = {"method": "one_hot_encoding", "new_columns": "see dataframe"}
        log_message(f"  One-Hot Encoded: {ohe_cols}", LOG)

    # Encode target separately if needed
    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype == object:
        le = LabelEncoder()
        df[TARGET_COLUMN] = le.fit_transform(df[TARGET_COLUMN].astype(str))
        encoding_maps[TARGET_COLUMN] = {
            "method": "label_encoding",
            "mapping": {str(cls): int(i) for i, cls in enumerate(le.classes_)},
        }
        log_message(f"  Target encoded: {encoding_maps[TARGET_COLUMN]['mapping']}", LOG)

    print(f"\n  Label-encoded columns  : {len(le_cols)}")
    print(f"  One-hot-encoded columns: {len(ohe_cols)}")
    print(f"  Final shape            : {df.shape}")

    save_json(encoding_maps, ENCODING_MAPPINGS)
    save_data(df, PREPROCESSED_DATA)
    print_completion([ENCODING_MAPPINGS, PREPROCESSED_DATA])
    return df, encoding_maps


# ════════════════════════════════════════════════════════════════════════════
#  PART H — FINAL DATA QUALITY REPORT
# ════════════════════════════════════════════════════════════════════════════

def part_h_report(
    df_original: pd.DataFrame,
    df_final: pd.DataFrame,
    outlier_stats: dict,
    encoding_maps: dict,
) -> None:
    """Write a comprehensive preprocessing summary report."""
    print_step("H", "Final Data Quality Report")
    log_section("PART H — PREPROCESSING REPORT", LOG)

    miss_orig  = df_original.isnull().sum().sum()
    miss_final = df_final.isnull().sum().sum()

    # Custom quality score: penalise remaining nulls + outliers
    total_cells    = df_final.shape[0] * df_final.shape[1]
    remaining_null = df_final.isnull().sum().sum()
    quality_score  = round((1 - remaining_null / max(total_cells, 1)) * 100, 2)

    lines = [
        "=" * 65,
        "  HOSPITAL READMISSION — PREPROCESSING SUMMARY REPORT",
        f"  Generated: {generate_timestamp()}",
        "=" * 65,
        "",
        "SHAPE",
        f"  Original  : {df_original.shape[0]} rows × {df_original.shape[1]} cols",
        f"  Final     : {df_final.shape[0]} rows × {df_final.shape[1]} cols",
        f"  Rows removed : {df_original.shape[0] - df_final.shape[0]}",
        "",
        "MISSING VALUES",
        f"  Before treatment : {miss_orig}",
        f"  After treatment  : {miss_final}",
        "",
        "OUTLIERS HANDLED",
    ]
    for col, s in outlier_stats.items():
        lines.append(f"  {col:<35s}  {s['outliers_found']:>5d} outlier(s) capped")

    lines += [
        "",
        "ENCODING APPLIED",
        f"  Total columns encoded : {len(encoding_maps)}",
    ]
    for col, info in encoding_maps.items():
        lines.append(f"  {col:<35s}  {info['method']}")

    lines += [
        "",
        f"DATA QUALITY SCORE (custom): {quality_score} / 100",
        "",
        "=" * 65,
    ]

    os.makedirs(os.path.dirname(PREPROCESSING_SUMMARY), exist_ok=True)
    with open(PREPROCESSING_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    log_message(f"Preprocessing summary saved → {PREPROCESSING_SUMMARY}", LOG)
    print_completion([PREPROCESSING_SUMMARY])


# ════════════════════════════════════════════════════════════════════════════
#  PART I — VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

def part_i_visualizations(df_original: pd.DataFrame, df_final: pd.DataFrame) -> None:
    """Generate and save all EDA visualizations."""
    print_step("I", "EDA Visualizations")
    log_section("PART I — VISUALIZATIONS", LOG)

    saved = []

    # 1. Missing value heatmap AFTER imputation
    _save_missing_heatmap(df_final, "missing_heatmap_after.png")
    saved.append(os.path.join(EDA_DIR, "missing_heatmap_after.png"))

    # 2. Data-type distribution pie chart (original dataset)
    dtype_counts = df_original.dtypes.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=VIZ["figure_size_sq"])
    ax.pie(
        dtype_counts.values,
        labels=dtype_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette(VIZ["color_palette"], len(dtype_counts)),
    )
    ax.set_title("Data Type Distribution", fontsize=VIZ["title_fontsize"])
    path = plot_save(fig, "dtype_distribution.png", EDA_DIR)
    saved.append(path)

    # 3. Numerical feature histograms
    num_cols = [c for c in NUMERICAL_FEATURES if c in df_final.columns]
    if num_cols:
        n_cols_plot = 3
        n_rows = (len(num_cols) + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows, n_cols_plot,
                                 figsize=(18, 4 * n_rows))
        axes = np.array(axes).flatten()
        for j, col in enumerate(num_cols):
            axes[j].hist(df_final[col].dropna(), bins=30,
                         color=VIZ["hist_color"], edgecolor="white", alpha=0.8)
            axes[j].set_title(col, fontsize=9)
            axes[j].set_xlabel(col, fontsize=8)
            axes[j].set_ylabel("Count", fontsize=8)
        for k in range(len(num_cols), len(axes)):
            axes[k].set_visible(False)
        fig.suptitle("Numerical Feature Distributions", fontsize=13)
        plt.tight_layout()
        path = plot_save(fig, "numerical_distributions.png", EDA_DIR)
        saved.append(path)

    # 4. Categorical feature count plots (top-10 values, first 9 columns)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df_original.columns][:9]
    if cat_cols:
        n_cols_plot = 3
        n_rows = (len(cat_cols) + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows, n_cols_plot,
                                 figsize=(18, 5 * n_rows))
        axes = np.array(axes).flatten()
        for j, col in enumerate(cat_cols):
            top10 = df_original[col].value_counts().head(10)
            axes[j].barh(top10.index.astype(str), top10.values,
                         color=VIZ["bar_color"], alpha=0.85)
            axes[j].set_title(col, fontsize=9)
            axes[j].set_xlabel("Count", fontsize=8)
            axes[j].invert_yaxis()
        for k in range(len(cat_cols), len(axes)):
            axes[k].set_visible(False)
        fig.suptitle("Categorical Feature Distributions (top-10 values)", fontsize=13)
        plt.tight_layout()
        path = plot_save(fig, "categorical_distributions.png", EDA_DIR)
        saved.append(path)

    log_message(f"Saved {len(saved)} visualizations to {EDA_DIR}", LOG)
    print_completion(saved)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline() -> pd.DataFrame:
    """Execute Parts A–I in sequence."""
    print("\n" + "█" * 65)
    print("  HOSPITAL READMISSION — PREPROCESSING PIPELINE")
    print(f"  Started: {generate_timestamp()}")
    print("█" * 65)

    # Create all project directories
    create_directories()
    log_message("=" * 65, LOG)
    log_message(f"Pipeline started: {generate_timestamp()}", LOG)

    # A — Load
    df = part_a_load_data()
    df_original = df.copy()

    # B — Quality check
    part_b_quality_check(df)

    # C — Impute
    df = part_c_impute(df)

    # D — Dtype corrections
    df = part_d_dtype_corrections(df)

    # E — Deduplicate
    df = part_e_remove_duplicates(df)

    # F — Outliers
    df, outlier_stats = part_f_outliers(df)

    # G — Encoding
    df, encoding_maps = part_g_encoding(df)

    # H — Report
    part_h_report(df_original, df, outlier_stats, encoding_maps)

    # I — Visualisations
    part_i_visualizations(df_original, df)

    log_message(f"Pipeline complete: {generate_timestamp()}", LOG)

    print("\n" + "█" * 65)
    print("  ✅  ALL PREPROCESSING STEPS COMPLETE")
    print(f"  Finished: {generate_timestamp()}")
    print("  Key outputs:")
    print(f"    • Preprocessed data  : {PREPROCESSED_DATA}")
    print(f"    • Summary report     : {PREPROCESSING_SUMMARY}")
    print(f"    • Encoding mappings  : {ENCODING_MAPPINGS}")
    print(f"    • Visualizations     : {EDA_DIR}")
    print(f"    • Full log           : {PREPROCESSING_LOG}")
    print("█" * 65 + "\n")

    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()
