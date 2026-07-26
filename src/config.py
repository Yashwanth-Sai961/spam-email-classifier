"""
config.py

Centralized configuration for the Email Security Assistant.

Every path, filename, and tunable constant used across the project lives
here so that other modules never hardcode a string literal like
"data/raw/spam.csv". This means:
  - Paths are resolved relative to the project root regardless of which
    directory a script is launched from (previously every script assumed
    it was run from the project root).
  - Changing a threshold or a filename is a one-line edit here instead of
    a find-and-replace across a dozen files.

Nothing in this file performs I/O. It only defines constants.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Project root
# --------------------------------------------------------------------------
# This file lives at <project_root>/src/config.py, so the project root is
# one level up from this file's directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Top-level directories
# --------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure directories that are written to actually exist. Directories that
# are only ever read from (raw data) are NOT created here, since a missing
# raw data directory should raise a clear FileNotFoundError, not silently
# produce an empty one.
for _directory in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset files
# --------------------------------------------------------------------------
SPAM_RAW_CSV: Path = RAW_DATA_DIR / "spam.csv"
SPAM_PROCESSED_CSV: Path = PROCESSED_DATA_DIR / "cleaned_spam.csv"

# Populated in the phishing/URL phase once a public dataset is sourced.
PHISHING_URL_RAW_CSV: Path = RAW_DATA_DIR / "phishing_urls.csv"
PHISHING_URL_PROCESSED_CSV: Path = PROCESSED_DATA_DIR / "cleaned_phishing_urls.csv"

# --------------------------------------------------------------------------
# Model / vectorizer artifact files
# --------------------------------------------------------------------------
TFIDF_VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.joblib"
COUNT_VECTORIZER_PATH: Path = MODELS_DIR / "count_vectorizer.joblib"
SPAM_CLASSIFIER_PATH: Path = MODELS_DIR / "spam_classifier.joblib"

# Populated in later phases.
PHISHING_CLASSIFIER_PATH: Path = MODELS_DIR / "phishing_classifier.joblib"
URL_CLASSIFIER_PATH: Path = MODELS_DIR / "url_classifier.joblib"

# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
DATABASE_PATH: Path = PROJECT_ROOT / "email_security.db"

# --------------------------------------------------------------------------
# Modeling constants
# --------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Label used by the spam classifier for the positive ("bad") class. Kept as
# a named constant because sklearn's precision/recall/F1 calls require an
# explicit pos_label, and this string appears in several files.
SPAM_POSITIVE_LABEL: str = "spam"

# --------------------------------------------------------------------------
# Risk scoring thresholds (0-100 scale)
# --------------------------------------------------------------------------
# These bands translate a numeric risk_score into a human-readable risk
# category. Defined once here so the Streamlit UI, the risk engine, and any
# report generator all agree on the same cutoffs.
RISK_BANDS: dict[str, tuple[int, int]] = {
    "Safe": (0, 19),
    "Low Risk": (20, 39),
    "Spam": (40, 59),
    "Suspicious": (60, 74),
    "High Risk": (75, 89),
    "Critical Risk": (90, 100),
}

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_FILE: Path = LOGS_DIR / "app.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# --------------------------------------------------------------------------
# File upload support
# --------------------------------------------------------------------------
ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (".txt", ".csv", ".eml")
MAX_UPLOAD_SIZE_MB: int = 10
