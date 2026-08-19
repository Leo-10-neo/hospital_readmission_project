"""
utils.py
========
Shared utility functions used across all pipeline modules.
"""

import os
import json
import logging
import datetime
import traceback

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

from config import LOGS_DIR, VIZ

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_console_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DIRECTORY MANAGEMENT
# ---------------------------------------------------------------------------

def create_directories(dirs: list | None = None) -> None:
    """
    Create all required project folders.

    Parameters
    ----------
    dirs : list of str, optional
        If supplied, only these directories are created.
        If None, the full set of project directories from config is created.
    """
    from config import (
        RAW_DATA_DIR, PROCESSED_DATA_DIR,
        REPORTS_DIR, LOGS_DIR,
        EDA_DIR, MODEL_PLOTS_DIR,
        MODELS_DIR,
    )

    default_dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        EDA_DIR,
        MODEL_PLOTS_DIR,
        MODELS_DIR,
    ]

    target_dirs = dirs if dirs is not None else default_dirs

    for directory in target_dirs:
        os.makedirs(directory, exist_ok=True)
        _console_logger.debug("Directory ready: %s", directory)

    _console_logger.info("✅  All %d directories verified / created.", len(target_dirs))


# ---------------------------------------------------------------------------
# DATA I/O
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    df       : DataFrame to save.
    filepath : Destination CSV path (parent directory must exist).
    index    : Whether to write the row index. Default False.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=index)
        _console_logger.info("💾  Saved  (%d rows × %d cols)  →  %s",
                             len(df), df.shape[1], filepath)
    except Exception as exc:
        _console_logger.error("❌  save_data failed for '%s': %s", filepath, exc)
        raise


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    filepath : Path to the CSV file.
    **kwargs : Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, **kwargs)
    _console_logger.info("📂  Loaded (%d rows × %d cols)  ←  %s",
                         len(df), df.shape[1], filepath)
    return df


def save_json(obj: dict, filepath: str) -> None:
    """Serialize *obj* to a pretty-printed JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=4, default=str)
    _console_logger.info("📄  JSON saved  →  %s", filepath)


def load_json(filepath: str) -> dict:
    """Load a JSON file and return the parsed object."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def log_message(message: str, filepath: str, level: str = "INFO") -> None:
    """
    Append a timestamped message to a plain-text log file AND echo to console.

    Parameters
    ----------
    message  : Text to log.
    filepath : Path to the log file (created if missing).
    level    : Log level label string (INFO / WARNING / ERROR).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    timestamp = generate_timestamp()
    line = f"[{timestamp}]  [{level.upper():7s}]  {message}\n"

    with open(filepath, "a", encoding="utf-8") as fh:
        fh.write(line)

    # Mirror to console
    if level.upper() == "ERROR":
        _console_logger.error(message)
    elif level.upper() == "WARNING":
        _console_logger.warning(message)
    else:
        _console_logger.info(message)


def log_section(title: str, filepath: str, width: int = 70) -> None:
    """Write a visual section header to the log file."""
    separator = "=" * width
    header = f"\n{separator}\n  {title.upper()}\n{separator}"
    log_message(header, filepath)


# ---------------------------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------------------------

def plot_save(fig: plt.Figure, filename: str, folder: str) -> str:
    """
    Save a Matplotlib figure to *folder/filename* and close it.

    Parameters
    ----------
    fig      : Matplotlib Figure object.
    filename : Filename including extension (e.g. ``"boxplot.png"``).
    folder   : Destination directory (created if missing).

    Returns
    -------
    str : Full path where the figure was saved.
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    _console_logger.info("🖼️   Figure saved  →  %s", filepath)
    return filepath


def new_figure(wide: bool = False, square: bool = False) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a styled Matplotlib figure using settings from config.VIZ.

    Returns
    -------
    (fig, ax) tuple
    """
    try:
        plt.style.use(VIZ["style"])
    except OSError:
        plt.style.use("seaborn-whitegrid")   # fallback for older matplotlib

    if square:
        size = VIZ["figure_size_sq"]
    elif wide:
        size = VIZ["figure_size_wide"]
    else:
        size = VIZ["figure_size"]

    fig, ax = plt.subplots(figsize=size)
    return fig, ax


# ---------------------------------------------------------------------------
# TIMESTAMP
# ---------------------------------------------------------------------------

def generate_timestamp(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Return the current local date-time as a formatted string.

    Parameters
    ----------
    fmt : ``strftime`` format string. Defaults to ``"%Y-%m-%d %H:%M:%S"``.

    Returns
    -------
    str
    """
    return datetime.datetime.now().strftime(fmt)


def generate_run_id() -> str:
    """Return a compact timestamp suitable for use in filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# PROGRESS INDICATOR
# ---------------------------------------------------------------------------

def print_step(step: str, description: str) -> None:
    """Print a formatted pipeline step banner to stdout."""
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  STEP {step}: {description}")
    print(f"{bar}")


def print_completion(files: list[str]) -> None:
    """Print a formatted completion message listing output file paths."""
    print("\n" + "=" * 60)
    print("  ✅  PIPELINE STEP COMPLETE")
    print("=" * 60)
    print("  Output files:")
    for f in files:
        print(f"    • {f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# SAFE WRAPPER
# ---------------------------------------------------------------------------

def safe_run(func, *args, log_path: str | None = None, **kwargs):
    """
    Execute *func* with *args*/*kwargs*, catching and logging any exception.

    Returns the function result on success, or None on failure.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        msg = f"Error in {func.__name__}: {exc}\n{traceback.format_exc()}"
        if log_path:
            log_message(msg, log_path, level="ERROR")
        else:
            _console_logger.error(msg)
        return None
