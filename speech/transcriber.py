"""Speech-to-Text with Whisper. Owner: Member 6.

Whisper is already trained. We only use it.
"""
SAMPLE_RATE = 16000
_model = None


def _get_model():
    """Load Whisper only once (loading is slow)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _model


def listen(seconds=4):
    """Record from the microphone and return the text."""
    import sounddevice as sd

    input("Press Enter and speak... ")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    segments, _ = _get_model().transcribe(audio.flatten(), language="en", vad_filter=True)
    return " ".join(segment.text for segment in segments).strip()
