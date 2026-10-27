import pandas as pd

from preprocessing.clean import clean_text, cleaned_text_column


def test_lowercase_and_punctuation():
    assert clean_text("Open Chrome!") == "open chrome"


def test_extra_spaces():
    assert clean_text("  create   a folder. ") == "create a folder"


def test_cleaned_column_matches_inference_cleaning():
    frame = pd.DataFrame({"text": [" Open Chrome! ", "CREATE a folder"]})
    assert cleaned_text_column(frame).tolist() == [
        clean_text(value) for value in frame["text"]
    ]
