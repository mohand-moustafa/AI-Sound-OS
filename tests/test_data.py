import csv
from pathlib import Path

VALID_INTENTS = {
    "OPEN_APPLICATION",
    "CLOSE_APPLICATION",
    "CREATE_FOLDER",
    "CREATE_FILE",
    "LIST_FILES",
    "OPEN_FOLDER",
    "SYSTEM_INFO",
    "SHOW_DATE_TIME",
    "UNKNOWN",
}


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def test_dataset_has_required_columns():
    file_path = Path("data/raw/_template.csv")

    with open(file_path, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["text", "intent"]


def test_dataset_has_no_empty_values():
    rows = load_dataset("data/raw/_template.csv")

    for row in rows:
        assert row["text"].strip() != ""
        assert row["intent"].strip() != ""


def test_dataset_has_valid_intents():
    rows = load_dataset("data/raw/_template.csv")

    for row in rows:
        assert row["intent"] in VALID_INTENTS


def test_dataset_has_no_duplicate_commands():
    rows = load_dataset("data/raw/_template.csv")

    commands = [row["text"].strip().lower() for row in rows]

    assert len(commands) == len(set(commands))