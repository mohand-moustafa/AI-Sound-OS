"""Transcriber tests with a fake Whisper model (no model download needed)."""
import numpy as np

import speech.transcriber as tr


class FakeSeg:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def transcribe(self, samples, **kw):
        assert samples.dtype == np.float32
        return [FakeSeg(" Please open"), FakeSeg(" Chrome. ")], None


def test_transcribe_joins_segments():
    tr._models["fake"] = FakeModel()
    audio = np.ones(16000, dtype="float32") * 0.1
    assert tr.transcribe(audio, model_size="fake") == "Please open Chrome."


def test_transcribe_empty_or_short():
    tr._models["fake"] = FakeModel()
    assert tr.transcribe(None, model_size="fake") == ""
    assert tr.transcribe(np.zeros(100, dtype="float32"), model_size="fake") == ""
