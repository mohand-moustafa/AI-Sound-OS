from speech.evaluate_wer import corpus_wer, edit_distance, normalize_text, wer


def test_normalize():
    assert normalize_text("Please, open Chrome!") == "please open chrome"


def test_wer_values():
    assert wer("open chrome", "open chrome") == 0
    assert wer("open chrome", "open crome") == 0.5
    assert wer("open chrome now", "open chrome") == 1 / 3
    assert wer("open chrome", "") == 1


def test_edit_distance():
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_corpus_wer_weights_by_words():
    # 1 error in 2 words + 0 errors in 4 words = 1/6
    pairs = [("open chrome", "open crome"), ("create a folder called ai", "create a folder called ai")]
    assert abs(corpus_wer(pairs) - 1 / 7) < 1e-9
