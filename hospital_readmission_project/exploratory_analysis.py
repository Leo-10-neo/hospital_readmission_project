"""
exploratory_analysis.py — Step 2: EDA for Hospital Readmission
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
from config import (
    DATA_OUTLIERS_HANDLED, PREPROCESSED_DATA,
    NUMERICAL_FEATURES, TARGET_COLUMN,
    EDA_DIR, REPORTS_DIR, LOGS_DIR, VIZ,
)
from utils import log_message, log_section, plot_save, generate_timestamp, create_directories, print_step

LOG = os.path.join(LOGS_DIR, "eda_log.txt")
DEMO_DIR = os.path.join(EDA_DIR, "demographics")
CLIN_DIR = os.path.join(EDA_DIR, "clinical_features")
HOSP_DIR = os.path.join(EDA_DIR, "hospital_stay")
COMORB_DIR = os.path.join(EDA_DIR, "comorbidities")
CORR_DIR  = os.path.join(EDA_DIR, "correlations")

STYLE = VIZ["style"]
DPI   = VIZ["dpi"]

# ── helpers ──────────────────────────────────────────────────────────────────
def _style():
    try: plt.style.use(STYLE)
    except: plt.style.use("seaborn-whitegrid")

def _save(fig, name, folder):
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log_message(f"Saved → {p}", LOG)
    return p

def _calculate_readmission_rate(series):
    """Return % of records that are readmitted (not 'NO')."""
    return (series != "NO").sum() / len(series) * 100

# ── load data ─────────────────────────────────────────────────────────────────
def load_datasets():
    df_raw = pd.read_csv(DATA_OUTLIERS_HANDLED)
    # Ensure target is original string form if available
    log_message(f"Raw (pre-enc) loaded: {df_raw.shape}", LOG)
    return df_raw

# ══ A: TARGET VARIABLE ════════════════════════════════════════════════════════
def section_a(df):
    print_step("A", "Target Variable Analysis")
    log_section("SECTION A — TARGET VARIABLE", LOG)

    counts = df[TARGET_COLUMN].value_counts()
    pcts   = (counts / len(df) * 100).round(2)
    imbalance = counts.max() / counts.min()

    log_message(f"Class distribution:\n{pd.concat([counts, pcts], axis=1).to_string()}", LOG)
    log_message(f"Imbalance ratio: {imbalance:.2f}", LOG)
    print(f"\nClass distribution:\n{pd.concat([counts, pcts.rename('%')], axis=1)}")
    print(f"Imbalance ratio: {imbalance:.2f}x")

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # pie
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("Set2", len(counts)), startangle=90)
    axes[0].set_title("Readmission Distribution (Pie)", fontsize=13)

    # count plot
    palette = sns.color_palette("Set2", len(counts))
    bars = axes[1].bar(counts.index.astype(str), counts.values, color=palette)
    for bar, pct in zip(bars, pcts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f"{pct}%", ha="center", fontsize=10)
    axes[1].set_title("Readmission Count Plot", fontsize=13)
    axes[1].set_xlabel("Readmission Status"); axes[1].set_ylabel("Count")

    # imbalance info
    axes[2].axis("off")
    info = (f"Total Records : {len(df):,}\n"
            f"Imbalance Ratio : {imbalance:.2f}x\n\n"
            + "\n".join([f"{k}: {v:,} ({pcts[k]}%)" for k, v in counts.items()]))
    axes[2].text(0.1, 0.5, info, transform=axes[2].transAxes,
                 fontsize=12, va="center", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    axes[2].set_title("Summary", fontsize=13)

    fig.suptitle("Target Variable — Readmission Status", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "target_distribution.png", EDA_DIR)

# ══ B: DEMOGRAPHICS ═══════════════════════════════════════════════════════════
def section_b(df):
    print_step("B", "Demographic Analysis")
    log_section("SECTION B — DEMOGRAPHICS", LOG)
    _style()

    # ── Age ──
    age_col = next((c for c in df.columns if c == "age"), None)
    if age_col:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        order = sorted(df[age_col].unique())
        sns.countplot(data=df, x=age_col, order=order, palette="Blues_d", ax=axes[0])
        axes[0].set_title("Age Group Distribution"); axes[0].tick_params(axis="x", rotation=45)

        for cls, color in zip(df[TARGET_COLUMN].unique(), sns.color_palette("Set2")):
            subset = df[df[TARGET_COLUMN] == cls][age_col]
            axes[1].hist([list(order).index(v) for v in subset if v in order],
                         alpha=0.5, label=str(cls), color=color, bins=len(order))
        axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(order, rotation=45)
        axes[1].set_title("Age by Readmission"); axes[1].legend()

        sns.countplot(data=df, x=age_col, hue=TARGET_COLUMN, order=order,
                      palette="Set2", ax=axes[2])
        axes[2].set_title("Stacked Age vs Readmission"); axes[2].tick_params(axis="x", rotation=45)

        fig.suptitle("Age Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, "age_analysis.png", DEMO_DIR)

    # ── Gender ──
    gen_col = next((c for c in df.columns if c == "gender"), None)
    if gen_col:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sns.countplot(data=df, x=gen_col, palette="Set2", ax=axes[0])
        axes[0].set_title("Gender Distribution")

        readm_rate = df.groupby(gen_col)[TARGET_COLUMN].apply(
            lambda x: (x != "NO").sum() / len(x) * 100).reset_index()
        readm_rate.columns = [gen_col, "readmission_rate"]
        axes[1].bar(readm_rate[gen_col], readm_rate["readmission_rate"],
                    color=sns.color_palette("Set2"))
        axes[1].set_title("Readmission Rate by Gender (%)"); axes[1].set_ylabel("%")

        pivot = df.groupby([gen_col, TARGET_COLUMN]).size().unstack(fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=axes[2], colormap="Set2")
        axes[2].set_title("Gender vs Readmission (Stacked)"); axes[2].tick_params(axis="x", rotation=0)

        fig.suptitle("Gender Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, "gender_analysis.png", DEMO_DIR)

    # ── Race ──
    race_col = next((c for c in df.columns if c == "race"), None)
    if race_col:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        race_counts = df[race_col].value_counts()
        axes[0].barh(race_counts.index, race_counts.values, color=sns.color_palette("Set2", len(race_counts)))
        axes[0].set_title("Race Distribution"); axes[0].set_xlabel("Count"); axes[0].invert_yaxis()

        race_rate = df.groupby(race_col)[TARGET_COLUMN].apply(
            lambda x: (x != "NO").sum() / len(x) * 100).sort_values()
        axes[1].barh(race_rate.index, race_rate.values, color=sns.color_palette("RdYlGn", len(race_rate)))
        axes[1].set_title("Readmission Rate by Race (%)"); axes[1].set_xlabel("%"); axes[1].invert_yaxis()

        fig.suptitle("Race Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, "race_analysis.png", DEMO_DIR)

# ══ C: CLINICAL FEATURES ══════════════════════════════════════════════════════
def section_c(df):
    print_step("C", "Clinical Features Analysis")
    log_section("SECTION C — CLINICAL FEATURES", LOG)
    _style()

    # ── Diagnoses ──
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        if d_col not in df.columns: continue
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        top15 = df[d_col].value_counts().head(15)
        axes[0].barh(top15.index, top15.values, color=sns.color_palette("Blues_d", 15))
        axes[0].set_title(f"Top 15 {d_col} Codes"); axes[0].invert_yaxis()

        top10_codes = top15.index[:10]
        rates = df[df[d_col].isin(top10_codes)].groupby(d_col)[TARGET_COLUMN].apply(_calculate_readmission_rate)
        axes[1].barh(rates.index, rates.values, color=sns.color_palette("Oranges_d", len(rates)))
        axes[1].set_title(f"Readmission Rate by {d_col} (Top 10)"); axes[1].set_xlabel("%")
        axes[1].invert_yaxis()

        fig.suptitle(f"Diagnosis Analysis — {d_col}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, f"{d_col}_analysis.png", CLIN_DIR)

    # ── Number of diagnoses ──
    num_diag = "number_diagnoses"
    if num_diag in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(df[num_diag].dropna(), bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
        axes[0].set_title("Number of Diagnoses Distribution"); axes[0].set_xlabel("Count")

        target_str = df[TARGET_COLUMN].astype(str)
        for cls in target_str.unique():
            axes[1].hist(df[target_str == cls][num_diag].dropna(), alpha=0.6, label=cls, bins=15)
        axes[1].set_title("Diagnoses Count by Readmission"); axes[1].legend()

        fig.suptitle("Number of Diagnoses", fontsize=14)
        plt.tight_layout()
        _save(fig, "number_diagnoses.png", CLIN_DIR)

    # ── Medications ──
    med_col = "num_medications"
    if med_col in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(df[med_col].dropna(), bins=25, color="#55A868", edgecolor="white", alpha=0.8)
        axes[0].set_title("Number of Medications Distribution")

        try:
            sns.violinplot(data=df, x=TARGET_COLUMN, y=med_col, palette="Set2", ax=axes[1])
        except:
            sns.boxplot(data=df, x=TARGET_COLUMN, y=med_col, palette="Set2", ax=axes[1])
        axes[1].set_title("Medications vs Readmission"); axes[1].tick_params(axis="x", rotation=30)

        fig.suptitle("Medications Analysis", fontsize=14)
        plt.tight_layout()
        _save(fig, "medications_analysis.png", CLIN_DIR)

    # ── Procedures ──
    proc_col = "num_procedures"
    if proc_col in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(df[proc_col].dropna(), bins=15, color="#C44E52", edgecolor="white", alpha=0.8)
        axes[0].set_title("Number of Procedures Distribution")

        sns.boxplot(data=df, x=TARGET_COLUMN, y=proc_col, palette="Set2", ax=axes[1])
        axes[1].set_title("Procedures vs Readmission"); axes[1].tick_params(axis="x", rotation=30)

        fig.suptitle("Procedures Analysis", fontsize=14)
        plt.tight_layout()
        _save(fig, "procedures_analysis.png", CLIN_DIR)

# ══ D: HOSPITAL STAY ══════════════════════════════════════════════════════════
def section_d(df):
    print_step("D", "Hospital Stay Analysis")
    log_section("SECTION D — HOSPITAL STAY", LOG)
    _style()

    stay_col = "time_in_hospital"
    if stay_col not in df.columns:
        log_message("time_in_hospital not found.", LOG); return

    stats_tbl = df.groupby(TARGET_COLUMN)[stay_col].agg(["mean","median","std"]).round(2)
    log_message(f"Length of stay stats:\n{stats_tbl.to_string()}", LOG)
    print(f"\nLength of stay:\n{stats_tbl}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0,0].hist(df[stay_col].dropna(), bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0,0].set_title("Time in Hospital Distribution"); axes[0,0].set_xlabel("Days")

    sns.boxplot(data=df, x=TARGET_COLUMN, y=stay_col, palette="Set2", ax=axes[0,1])
    axes[0,1].set_title("Time in Hospital by Readmission"); axes[0,1].tick_params(axis="x", rotation=30)

    means = df.groupby(TARGET_COLUMN)[stay_col].mean()
    axes[1,0].bar(means.index.astype(str), means.values, color=sns.color_palette("Set2", len(means)))
    for i, v in enumerate(means.values):
        axes[1,0].text(i, v + 0.05, f"{v:.2f}", ha="center")
    axes[1,0].set_title("Average Length of Stay"); axes[1,0].set_ylabel("Days")

    med_col = "num_medications"
    if med_col in df.columns:
        sample = df.sample(min(3000, len(df)), random_state=42)
        sc = axes[1,1].scatter(sample[stay_col], sample[med_col],
                               alpha=0.3, c="#4C72B0", s=10)
        axes[1,1].set_xlabel("Time in Hospital"); axes[1,1].set_ylabel("Medications")
        axes[1,1].set_title("Length of Stay vs Medications")
    else:
        axes[1,1].axis("off")

    fig.suptitle("Hospital Stay Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "hospital_stay_analysis.png", HOSP_DIR)

# ══ E: COMORBIDITIES ══════════════════════════════════════════════════════════
def section_e(df):
    print_step("E", "Comorbidity Analysis")
    log_section("SECTION E — COMORBIDITIES", LOG)
    _style()

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]

    # Comorbidity proxy: number_diagnoses
    if "number_diagnoses" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(df["number_diagnoses"].dropna(), bins=15, color="#8172B2", edgecolor="white", alpha=0.8)
        axes[0].set_title("Number of Diagnoses (Comorbidity Proxy)")

        sns.boxplot(data=df, x=TARGET_COLUMN, y="number_diagnoses", palette="Set2", ax=axes[1])
        axes[1].set_title("Diagnoses Count vs Readmission"); axes[1].tick_params(axis="x", rotation=30)

        fig.suptitle("Comorbidity Analysis", fontsize=14)
        plt.tight_layout()
        _save(fig, "comorbidity_analysis.png", COMORB_DIR)

    # Heatmap of numerical comorbidity-related features
    if len(num_cols) >= 3:
        corr_subset = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title("Numerical Feature Correlation (Comorbidities)", fontsize=13)
        plt.tight_layout()
        _save(fig, "comorbidity_heatmap.png", COMORB_DIR)

# ══ F: CORRELATION ANALYSIS ═══════════════════════════════════════════════════
def section_f(df):
    print_step("F", "Correlation Analysis")
    log_section("SECTION F — CORRELATION", LOG)
    _style()

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]

    # ── Correlation heatmap ──
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, square=True, linewidths=0.5, mask=mask,
                annot_kws={"size": 9})
    ax.set_title("Numerical Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png", CORR_DIR)

    # ── Feature vs target correlations ──
    target_num = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if target_num.isna().all():
        from sklearn.preprocessing import LabelEncoder
        target_num = pd.Series(LabelEncoder().fit_transform(df[TARGET_COLUMN].astype(str)),
                               index=df.index)

    feat_corr = df[num_cols].apply(lambda col: col.corr(target_num)).sort_values(key=abs, ascending=False)
    log_message(f"Feature-target correlations:\n{feat_corr.to_string()}", LOG)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in feat_corr.values]
    ax.barh(feat_corr.index[::-1], feat_corr.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Feature Correlation with Readmission (Target)", fontsize=13)
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    _save(fig, "feature_target_correlation.png", CORR_DIR)

    return feat_corr

# ══ G: STATISTICAL SUMMARY TABLES ════════════════════════════════════════════
def section_g(df, feat_corr):
    print_step("G", "Statistical Summary Tables")
    log_section("SECTION G — STATISTICAL TABLES", LOG)

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    rows = []

    for col in num_cols:
        for cls in df[TARGET_COLUMN].unique():
            subset = df[df[TARGET_COLUMN] == cls][col].dropna()
            rows.append({
                "column": col, "readmission_status": cls,
                "mean": round(subset.mean(), 3),
                "median": round(subset.median(), 3),
                "std": round(subset.std(), 3),
                "min": round(subset.min(), 3),
                "max": round(subset.max(), 3),
            })

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(REPORTS_DIR, "eda_statistical_summary.csv")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    log_message(f"Statistical summary saved → {out_path}", LOG)
    print(f"Statistical summary saved → {out_path}")

    # Top 10 risk factors
    top10 = feat_corr.abs().nlargest(10)
    log_message(f"Top 10 risk factors:\n{top10.to_string()}", LOG)
    print(f"\nTop 10 correlated features:\n{top10}")

# ══ H: KEY INSIGHTS ═══════════════════════════════════════════════════════════
def section_h(df, feat_corr):
    print_step("H", "Key Insights Extraction")
    log_section("SECTION H — INSIGHTS", LOG)

    counts = df[TARGET_COLUMN].value_counts()
    imbalance = counts.max() / counts.min()
    top_feat = feat_corr.abs().nlargest(3).index.tolist()

    stay_col = "time_in_hospital"
    if stay_col in df.columns:
        stay_means = df.groupby(TARGET_COLUMN)[stay_col].mean().to_dict()
    else:
        stay_means = {}

    lines = [
        "=" * 65,
        "  HOSPITAL READMISSION EDA — KEY INSIGHTS",
        f"  Generated: {generate_timestamp()}",
        "=" * 65,
        "",
        "CLASS IMBALANCE",
        f"  Imbalance ratio        : {imbalance:.2f}x",
        f"  Class distribution     : {dict(counts)}",
        f"  Recommendation         : Use SMOTE or class_weight='balanced'",
        "",
        "TOP CORRELATED FEATURES",
        f"  Top 3 features         : {', '.join(top_feat)}",
        "",
        "HOSPITAL STAY PATTERNS",
    ]
    for cls, mean_val in stay_means.items():
        lines.append(f"  Avg stay ({cls:>5s})     : {mean_val:.2f} days")

    lines += [
        "",
        "COMORBIDITY INSIGHTS",
        "  higher number_diagnoses correlates with readmission risk.",
        "",
        "RECOMMENDATIONS",
        "  1. Address class imbalance before modelling.",
        "  2. Feature-engineer diagnosis codes into ICD categories.",
        "  3. Consider dimensionality reduction (PCA/UMAP) on OHE columns.",
        "  4. Include interaction terms for top correlated features.",
        "",
        "=" * 65,
    ]

    out = os.path.join(REPORTS_DIR, "eda_insights.txt")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    log_message(f"Insights saved → {out}", LOG)

# ══ I: SUMMARY DASHBOARD ══════════════════════════════════════════════════════
def section_i(df):
    print_step("I", "EDA Summary Dashboard")
    log_section("SECTION I — DASHBOARD", LOG)
    _style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hospital Readmission EDA Summary", fontsize=16, fontweight="bold", y=1.01)

    # 1 — Target distribution
    counts = df[TARGET_COLUMN].value_counts()
    axes[0,0].pie(counts.values, labels=counts.index.astype(str),
                  autopct="%1.1f%%", colors=sns.color_palette("Set2", len(counts)))
    axes[0,0].set_title("Readmission Distribution")

    # 2 — Age group distribution
    age_col = "age"
    if age_col in df.columns:
        order = sorted(df[age_col].unique())
        age_counts = df[age_col].value_counts().reindex(order)
        axes[0,1].bar(range(len(order)), age_counts.values,
                      color=sns.color_palette("Blues_d", len(order)))
        axes[0,1].set_xticks(range(len(order)))
        axes[0,1].set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        axes[0,1].set_title("Age Group Distribution")
    else:
        axes[0,1].axis("off")

    # 3 — Top diagnoses
    if "diag_1" in df.columns:
        top10 = df["diag_1"].value_counts().head(10)
        axes[1,0].barh(top10.index, top10.values, color=sns.color_palette("Oranges_d", 10))
        axes[1,0].set_title("Top 10 Primary Diagnoses"); axes[1,0].invert_yaxis()
    else:
        axes[1,0].axis("off")

    # 4 — Length of stay
    if "time_in_hospital" in df.columns:
        for cls, color in zip(df[TARGET_COLUMN].unique(), sns.color_palette("Set2")):
            axes[1,1].hist(df[df[TARGET_COLUMN] == cls]["time_in_hospital"].dropna(),
                           alpha=0.6, label=str(cls), bins=14, color=color)
        axes[1,1].set_title("Length of Stay by Readmission")
        axes[1,1].set_xlabel("Days"); axes[1,1].legend()
    else:
        axes[1,1].axis("off")

    plt.tight_layout()
    _save(fig, "eda_summary_dashboard.png", EDA_DIR)

# ══ MAIN ══════════════════════════════════════════════════════════════════════
def run_eda():
    print("\n" + "=" * 65)
    print("  HOSPITAL READMISSION -- EDA PIPELINE")
    print(f"  Started: {generate_timestamp()}")
    print("=" * 65)

    for d in [DEMO_DIR, CLIN_DIR, HOSP_DIR, COMORB_DIR, CORR_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)

    log_message("=" * 65, LOG)
    log_message(f"EDA Pipeline started: {generate_timestamp()}", LOG)

    df = load_datasets()

    section_a(df)
    section_b(df)
    section_c(df)
    section_d(df)
    section_e(df)
    feat_corr = section_f(df)
    section_g(df, feat_corr)
    section_h(df, feat_corr)
    section_i(df)

    log_message(f"EDA complete: {generate_timestamp()}", LOG)

    print("\n" + "=" * 65)
    print("  [DONE] EDA COMPLETE")
    print(f"  Finished: {generate_timestamp()}")
    print("  Key outputs:")
    print(f"    - Visualizations  : {EDA_DIR}")
    print(f"    - Stats summary   : {os.path.join(REPORTS_DIR, 'eda_statistical_summary.csv')}")
    print(f"    - Insights report : {os.path.join(REPORTS_DIR, 'eda_insights.txt')}")
    print(f"    - EDA log         : {LOG}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_eda()
