"""Tests for audio_capture + wake_word. No microphone needed (fake stream)."""
import numpy as np

from speech.audio_capture import (
    SAMPLE_RATE, calibrate_threshold, collect_utterance,
    normalize_audio, record_until_silence, rms, trim_silence,
)
from speech.wake_word import WakeWordListener, is_wake_phrase


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 440 * t)).astype("float32")


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype="float32")


class FakeStream:
    """Pretends to be a microphone: plays a prepared signal, then silence."""
    def __init__(self, signal):
        self.signal, self.pos = signal, 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        chunk = self.signal[self.pos:self.pos + n]
        self.pos += n
        if len(chunk) < n:
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), dtype="float32")])
        return chunk.reshape(-1, 1), False


# ---------------------------------------------------------------- preprocessing
def test_normalize_peak():
    out = normalize_audio(tone(1, amp=0.1))
    assert abs(np.max(np.abs(out)) - 0.9) < 1e-3


def test_normalize_empty_and_silent():
    assert normalize_audio(np.array([], dtype="float32")).size == 0
    assert np.all(normalize_audio(silence(0.5)) == 0)


def test_trim_removes_silence():
    audio = np.concatenate([silence(1), tone(1), silence(1)])
    out = trim_silence(audio, threshold=0.01)
    assert 1.2 < len(out) / SAMPLE_RATE < 1.6      # 1 s tone + 2 x 0.2 s margin


def test_trim_all_silence_is_empty():
    assert trim_silence(silence(1), 0.01).size == 0


def test_calibration_has_floor():
    assert calibrate_threshold(FakeStream(silence(1))) == 0.01


# ---------------------------------------------------------------- recording
def test_record_returns_speech():
    sig = np.concatenate([silence(1), tone(1.5), silence(2)])
    audio = record_until_silence(stream_factory=lambda: FakeStream(sig))
    assert audio is not None and 1.3 < len(audio) / SAMPLE_RATE < 2.2
    assert abs(np.max(np.abs(audio)) - 0.9) < 1e-3


def test_record_no_speech_returns_none():
    audio = record_until_silence(wait_timeout=0.5,
                                 stream_factory=lambda: FakeStream(silence(5)))
    assert audio is None


def test_too_short_click_is_ignored():
    sig = np.concatenate([silence(1), tone(0.1), silence(3)])
    assert record_until_silence(stream_factory=lambda: FakeStream(sig)) is None


# ---------------------------------------------------------------- wake word
def test_wake_phrase_matching():
    assert is_wake_phrase("Hey, sound!")
    assert is_wake_phrase("hey sound open chrome")
    assert is_wake_phrase("hey sand")
    assert not is_wake_phrase("open chrome")
    assert not is_wake_phrase("what is the weather")
    assert not is_wake_phrase("")


def test_listener_ignores_other_speech_then_wakes():
    sig = np.concatenate([
        silence(1), tone(1.0),            # speech #1 (not the wake word)
        silence(1.5), tone(1.0),          # speech #2 = wake word
        silence(0.8), tone(1.5),          # the command
        silence(2),
    ])
    heard = iter(["", "what is the weather", "hey sound"])   # first "" = warm-up call
    statuses = []
    listener = WakeWordListener(lambda a: next(heard), on_status=statuses.append,
                                stream_factory=lambda: FakeStream(sig))
    cmd = listener.listen_for_command()
    assert cmd is not None and 1.3 < len(cmd) / SAMPLE_RATE < 2.2
    assert "WAKE_WORD_DETECTED" in statuses


def test_listener_stop():
    listener = WakeWordListener(lambda a: "", stream_factory=lambda: FakeStream(silence(1)))
    # listen_for_command resets the flag at start, so stop from a status callback
    listener.on_status = lambda msg: listener.stop()
    assert listener.listen_for_command() is None
