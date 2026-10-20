"""
speech/transcriber.py  (Owner: Salma Mohamed)

Offline speech-to-text with faster-whisper.
Contract:  transcribe(audio) -> str      e.g. "please open chrome"
audio = float32 numpy array (16 kHz mono)  OR  path to a WAV file.
Models are loaded once and cached (so the wake word re-uses them).
"""
import logging
import os
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small.en")
WAKE_MODEL_SIZE = "tiny.en"             # fast model for the wake-phrase check
MIN_SAMPLES = int(0.2 * SAMPLE_RATE)    # ignore clips shorter than 0.2 s

COMMAND_PROMPT = None
_models = {}


def get_model(size: str = None):
    """Load a Whisper model once (CPU, int8) and cache it."""
    size = size or DEFAULT_MODEL_SIZE
    if size not in _models:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper '%s' ...", size)
        _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _models[size]


_get_model = get_model      # old name, kept so existing code does not break


def _prepare(audio: Union[np.ndarray, str]) -> np.ndarray:
    if isinstance(audio, (str, os.PathLike)):
        import soundfile as sf
        data, sr = sf.read(str(audio), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:                      # simple resample
            n = int(len(data) * SAMPLE_RATE / sr)
            data = np.interp(np.linspace(0, len(data) - 1, n),
                             np.arange(len(data)), data)
        return data.astype("float32")
    return np.asarray(audio, dtype="float32").flatten()


def transcribe(audio: Union[np.ndarray, str], model_size: str = None,
               beam_size: int = 5, prompt: str = COMMAND_PROMPT) -> str:
    """Convert audio to plain text. Returns '' if nothing was understood."""
    if audio is None:
        return ""
    samples = _prepare(audio)
    if samples.size < MIN_SAMPLES:
        return ""
    model = get_model(model_size)
    segments, _ = model.transcribe(
        samples, language="en", beam_size=beam_size,
        temperature=0.0,
        initial_prompt=prompt or None,
        condition_on_previous_text=False, vad_filter=False)
    text = " ".join(s.text.strip() for s in segments).strip()
    logger.debug("Heard: %r", text)
    return text


def transcribe_wake(audio) -> str:
    """Fast tiny-model transcription used only for the wake-phrase check."""
    return transcribe(audio, model_size=WAKE_MODEL_SIZE, beam_size=1,
                      prompt=None)


def listen(seconds=4):
    """Old API kept for compatibility: record from the mic and return the text.
    (`seconds` is now the max command length; recording stops at silence.)"""
    from speech.audio_capture import record_until_silence
    audio = record_until_silence(max_duration=float(seconds))
    return transcribe(audio) if audio is not None else ""


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1:
        print(transcribe(sys.argv[1]))
    else:
        print("Heard:", listen(8) or "(nothing)")