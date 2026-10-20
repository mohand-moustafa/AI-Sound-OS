"""
speech/audio_capture.py  (Owner: Menna Zaki)

Microphone capture + silence detection + audio preprocessing.
Output contract: float32 numpy array, mono, 16 kHz  (or None if no speech).
sounddevice is imported lazily so the pure functions can be tested without a mic.
"""
import collections
import logging
import time
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000      # Whisper expects 16 kHz
CHANNELS = 1             # mono
CHUNK_DURATION = 0.1     # we read the mic in 100 ms pieces
PRE_CHUNKS = 5           # keep 0.5 s before speech starts
MIN_THRESHOLD = 0.01


# ---------------------------------------------------------------- helpers
def list_input_devices() -> list:
    """Return [(index, name), ...] of all microphones."""
    import sounddevice as sd
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def open_stream(device: Optional[int] = None, sample_rate: int = SAMPLE_RATE):
    """Open one input stream (use it as a context manager)."""
    import sounddevice as sd
    return sd.InputStream(samplerate=sample_rate, channels=CHANNELS,
                          dtype="float32", device=device)


def rms(chunk: np.ndarray) -> float:
    """Loudness of a chunk (root-mean-square energy)."""
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype("float64") ** 2)))


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Scale the audio so its loudest point equals target_peak."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    return (audio * (target_peak / peak)).astype("float32")


def trim_silence(audio: np.ndarray, threshold: float,
                 sample_rate: int = SAMPLE_RATE, keep: float = 0.2) -> np.ndarray:
    """Cut silence from the start and end, keeping `keep` seconds of margin."""
    if audio.size == 0:
        return audio
    frame = int(0.02 * sample_rate)                   # 20 ms frames
    n = len(audio) // frame
    if n == 0:
        return audio
    energies = np.array([rms(audio[i * frame:(i + 1) * frame]) for i in range(n)])
    voiced = np.where(energies > threshold)[0]
    if voiced.size == 0:
        return audio[:0]
    margin = int(keep * sample_rate)
    start = max(0, voiced[0] * frame - margin)
    end = min(len(audio), (voiced[-1] + 1) * frame + margin)
    return audio[start:end]


def calibrate_threshold(stream, n_chunks: int = 5,
                        sample_rate: int = SAMPLE_RATE) -> float:
    """Measure room noise for ~0.5 s and return a speech threshold.
    Uses the QUIETEST chunk (so early speech does not inflate it) and a cap."""
    size = int(CHUNK_DURATION * sample_rate)
    noise = [rms(stream.read(size)[0].flatten()) for _ in range(n_chunks)]
    return min(0.05, max(MIN_THRESHOLD, float(np.min(noise)) * 3))


# ---------------------------------------------------------------- core
def collect_utterance(
    stream,
    threshold: float,
    first_chunks: Sequence[np.ndarray] = (),
    max_len: float = 8.0,
    end_silence: float = 1.0,
    wait_timeout: Optional[float] = None,
    min_speech: float = 0.3,
    stop_check: Optional[Callable[[], bool]] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    Read ONE utterance from an already-open stream.

    - waits (up to wait_timeout seconds) for speech to start
    - stops after `end_silence` seconds of silence, or `max_len` seconds of speech
    - returns trimmed + normalized float32 audio, or None (no speech / too short)

    first_chunks: audio already read by the caller (last item may be the loud
    chunk that triggered us), so no audio is lost.
    """
    size = int(CHUNK_DURATION * sample_rate)
    pre = collections.deque(first_chunks, maxlen=PRE_CHUNKS)
    recorded = []
    started = bool(len(first_chunks)) and rms(first_chunks[-1]) > threshold
    speech = CHUNK_DURATION if started else 0.0
    silence = 0.0
    if started:
        recorded.extend(pre)
    t_wait = t_speech = time.monotonic()

    while not (stop_check and stop_check()):
        data, overflowed = stream.read(size)
        if overflowed:
            logger.warning("Audio buffer overflow")
        chunk = data.flatten()
        energy = rms(chunk)

        if not started:
            pre.append(chunk)
            if energy > threshold:
                started = True
                t_speech = time.monotonic()
                recorded.extend(pre)
                speech = CHUNK_DURATION
            elif wait_timeout is not None and time.monotonic() - t_wait > wait_timeout:
                return None
            continue

        recorded.append(chunk)
        if energy < threshold:
            silence += CHUNK_DURATION
        else:
            silence = 0.0
            speech += CHUNK_DURATION
        if silence >= end_silence or time.monotonic() - t_speech > max_len:
            break

    if not started or speech < min_speech:
        return None
    # trim with a LOWER threshold so quiet word starts/ends are kept
    audio = trim_silence(np.concatenate(recorded), max(0.005, threshold * 0.5),
                         sample_rate, keep=0.3)
    return normalize_audio(audio) if audio.size else None


def record_until_silence(
    silence_duration: float = 1.0,
    threshold: Optional[float] = None,      # None = auto calibration
    wait_timeout: float = 6.0,              # max wait for speech to START
    max_duration: float = 8.0,              # max length of the command
    min_speech_duration: float = 0.3,
    device: Optional[int] = None,
    sample_rate: int = SAMPLE_RATE,
    on_status: Optional[Callable[[str], None]] = None,
    stream_factory: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """
    Push-to-talk style recording: record one command, stop when speaker is silent.
    Returns float32 mono 16 kHz array, or None if nothing valid was said.
    """
    def status(msg):
        logger.info(msg)
        if on_status:
            on_status(msg)

    factory = stream_factory or (lambda: open_stream(device, sample_rate))
    with factory() as stream:
        if threshold is None:
            threshold = calibrate_threshold(stream, sample_rate=sample_rate)
        status("LISTENING")
        audio = collect_utterance(
            stream, threshold, max_len=max_duration, end_silence=silence_duration,
            wait_timeout=wait_timeout, min_speech=min_speech_duration,
            sample_rate=sample_rate)
    status("CAPTURED" if audio is not None else "NO_SPEECH")
    return audio


def save_audio(filename: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import soundfile as sf
    sf.write(filename, audio, sample_rate)
    logger.info("Saved audio to %s", filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Microphones:", list_input_devices())
    print("Speak now...")
    clip = record_until_silence()
    if clip is None:
        print("No valid speech captured.")
    else:
        save_audio("test_recording.wav", clip)
        print(f"Captured {len(clip) / SAMPLE_RATE:.2f} s -> test_recording.wav")