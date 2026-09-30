from preprocessing.clean import clean_text


def test_lowercase_and_punctuation():
    assert clean_text("Open Chrome!") == "open chrome"


def test_extra_spaces():
    assert clean_text("  create   a folder. ") == "create a folder"
