"""Shared settings for the whole project. Owner: Member 1."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "intent_model.joblib"
MODEL_METADATA_PATH = BASE_DIR / "models" / "intent_model_metadata.json"
DOCS_DIR = BASE_DIR / "docs"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# If the model is less confident than this, the answer becomes UNKNOWN.
# Tuned against the held-out test set: keeps normal commands usable while
# rejecting low-confidence unsupported inputs.
THRESHOLD = 0.35

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
    "TAKE_SCREENSHOT",
    "READ_SCREEN",
    "DESCRIBE_SCREEN",
]

# Screen commands have explicit routing because they trigger local capture/OCR
# capabilities rather than ordinary text classification.
CLASSIFIER_INTENTS = [
    intent for intent in INTENTS
    if intent not in {"TAKE_SCREENSHOT", "READ_SCREEN", "DESCRIBE_SCREEN"}
]
