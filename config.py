"""Shared settings for the whole project. Owner: Member 1."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "intent_model.joblib"
DOCS_DIR = BASE_DIR / "docs"

# If the model is less confident than this, the answer becomes UNKNOWN.
# Member 5 tunes this value after evaluation (try 0.4, 0.5, 0.6).
THRESHOLD = 0.5

INTENTS = [
    "OPEN_APPLICATION",
    "CLOSE_APPLICATION",
    "CREATE_FOLDER",
    "CREATE_FILE",
    "LIST_FILES",
    "OPEN_FOLDER",
    "SYSTEM_INFO",
    "SHOW_DATE_TIME",
    "UNKNOWN",
]
