"""Central configuration loaded from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    return default if val is None else val.strip().lower() in {"1", "true", "yes"}


DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")
SQL_DIR = PROJECT_ROOT / "sql"

TRAIN_CSV = DATA_DIR / "application_train.csv"
TEST_CSV = DATA_DIR / "application_test.csv"
DUCKDB_PATH = DATA_DIR / "credit_risk.duckdb"

MODEL_PATH = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
RULES_PATH = MODELS_DIR / "rules.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

RISK_LOW_THRESHOLD = float(os.getenv("RISK_LOW_THRESHOLD", "0.2"))
RISK_HIGH_THRESHOLD = float(os.getenv("RISK_HIGH_THRESHOLD", "0.5"))

N_SYNTHETIC_ROWS = int(os.getenv("N_SYNTHETIC_ROWS", "50000"))
RANDOM_SEED = 42

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

COLUMN_DICTIONARY_CSV = SQL_DIR / "HomeCredit_columns_description.csv"

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
APPLICATIONS_TABLE = "applications"
