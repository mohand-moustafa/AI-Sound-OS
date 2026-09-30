"""Text -> intent -> details -> (dry-run) action."""
import pytest

from commands.executor import execute
from nlp.classifier import IntentClassifier
from nlp.extractor import extract

CASES = [
    ("Open Chrome", "OPEN_APPLICATION"),
    ("Close Firefox", "CLOSE_APPLICATION"),
    ("Create a folder called AI Project", "CREATE_FOLDER"),
    ("Create a file named notes.txt", "CREATE_FILE"),
    ("Show the files in Downloads", "LIST_FILES"),
    ("Open the Downloads folder", "OPEN_FOLDER"),
    ("Show system information", "SYSTEM_INFO"),
    ("What time is it", "SHOW_DATE_TIME"),
]


@pytest.fixture(scope="module")
def classifier():
    return IntentClassifier()


@pytest.mark.parametrize("text, expected", CASES)
def test_intent(classifier, text, expected):
    intent, _ = classifier.predict(text)
    assert intent == expected


def test_full_pipeline_dry_run(classifier):
    text = "Create a folder called AI Project"
    intent, _ = classifier.predict(text)
    params = extract(intent, text)
    assert params["name"] == "AI Project"
    assert execute(intent, params, dry_run=True).success
