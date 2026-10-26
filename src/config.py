"""Central configuration for the DriftGuard pipeline.

All thresholds live here so they can be tuned without touching the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RANDOM_SEED = 42

# ---- Simulation ----------------------------------------------------------
REFERENCE_SIZE = 5000      # rows used to train the first model
BATCH_SIZE = 2000          # rows arriving each "week" in production
NUM_WEEKS = 12             # how many production weeks to simulate

FEATURES = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "num_credit_lines",
]
TARGET = "default"

# ---- Drift thresholds ----------------------------------------------------
# Population Stability Index (industry rule of thumb used in banking):
#   PSI < 0.10  -> no significant change
#   0.10-0.25   -> moderate change (warning)
#   PSI > 0.25  -> significant change (critical)
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25

# Kolmogorov-Smirnov test: feature is "drifted" if p-value below this
KS_P_VALUE = 0.01

# Dataset-level drift: retrain if this share of features is critical
DRIFT_SHARE_THRESHOLD = 0.30

# Performance drift: retrain if ROC-AUC falls more than this below baseline
AUC_DROP_THRESHOLD = 0.05

# ---- Retraining ----------------------------------------------------------
RETRAIN_WINDOW_WEEKS = 2   # retrain on the most recent N weeks of labelled data
MIN_IMPROVEMENT = 0.0      # candidate must beat production by at least this AUC
