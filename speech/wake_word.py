"""
speech/wake_word.py  (Owner: Menna Zaki)

Wake word = energy-based voice detection + tiny-Whisper phrase check ("hey sound").
Push-to-talk fallback = push_to_talk().
ONE continuous microphone stream is used, so no audio is lost between steps.
"""
import collections
import logging
import os
from difflib import SequenceMatcher
from typing import Callable, Optional

import numpy as np

from speech.audio_capture import (
    CHUNK_DURATION, PRE_CHUNKS, SAMPLE_RATE, calibrate_threshold,
    collect_utterance, open_stream, record_until_silence, rms,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

WAKE_PHRASES = ("hey sound", "hi sound", "hey sounds")
MATCH_RATIO = 0.75


def is_wake_phrase(text: str, phrases=WAKE_PHRASES, ratio: float = MATCH_RATIO) -> bool:
    """Fuzzy check; tolerates Whisper mistakes like 'hey sand' or 'Hey, sound!'."""
    t = "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()
    if not t:
        return False
    for p in phrases:
        if p in t:
            return True
        head = " ".join(t.split()[: len(p.split())])
        if SequenceMatcher(None, head, p).ratio() >= ratio:
            return True
    return False


class WakeWordListener:
    def __init__(
        self,
        transcribe_fn: Optional[Callable[[np.ndarray], str]] = None,
        device: Optional[int] = None,
        silence_duration: float = 1.0,
        command_wait_timeout: float = 6.0,
        max_command: float = 8.0,
        max_phrase: float = 2.5,
        on_status: Optional[Callable[[str], None]] = None,
        stream_factory: Optional[Callable] = None,   # used by tests
    ):
        if transcribe_fn is None:                    # default: tiny model
            from speech.transcriber import transcribe_wake
            transcribe_fn = transcribe_wake
        self.transcribe_fn = transcribe_fn
        self.silence_duration = silence_duration
        self.command_wait_timeout = command_wait_timeout
        self.max_command = max_command
        self.max_phrase = max_phrase
        self.on_status = on_status
        self._factory = stream_factory or (lambda: open_stream(device))
        self._stop = False

    def _status(self, msg: str):
        logger.info(msg)
        if self.on_status:
            self.on_status(msg)

    def stop(self):
        """Call from another thread (e.g. the GUI) to stop listening."""
        self._stop = True

    def _warm_up(self):
        """Load the Whisper model BEFORE we start listening."""
        self._status("LOADING_MODEL")
        try:
            self.transcribe_fn(np.zeros(SAMPLE_RATE, dtype="float32"))
        except Exception as e:
            logger.warning("Warm-up failed: %s", e)

    def listen_for_command(self) -> Optional[np.ndarray]:
        """
        Block until the wake phrase is heard, then return the command audio
        (float32, 16 kHz). Returns None only if stop() was called.
        """
        self._stop = False
        size = int(CHUNK_DURATION * SAMPLE_RATE)
        pre = collections.deque(maxlen=PRE_CHUNKS)

        self._warm_up()

        with self._factory() as stream:
            threshold = calibrate_threshold(stream)
            self._status("WAITING_FOR_WAKE_WORD")

            while not self._stop:
                chunk = stream.read(size)[0].flatten()
                pre.append(chunk)
                if rms(chunk) <= threshold:
                    continue

                # loud sound -> collect a short phrase and check it
                phrase = collect_utterance(
                    stream, threshold, first_chunks=list(pre),
                    max_len=self.max_phrase, end_silence=0.6,
                    stop_check=lambda: self._stop)
                pre.clear()
                if phrase is None:
                    continue

                text = self.transcribe_fn(phrase)
                logger.debug("Wake check heard: %r", text)
                if not is_wake_phrase(text):
                    continue

                self._status("WAKE_WORD_DETECTED")
                self._status("LISTENING")
                command = collect_utterance(
                    stream, threshold, max_len=self.max_command,
                    end_silence=self.silence_duration,
                    wait_timeout=self.command_wait_timeout,
                    stop_check=lambda: self._stop)
                if command is None:
                    self._status("NO_SPEECH_TIMEOUT")
                    self._status("WAITING_FOR_WAKE_WORD")
                    continue
                return command
        return None


def push_to_talk(on_status: Optional[Callable[[str], None]] = None,
                 device: Optional[int] = None) -> Optional[np.ndarray]:
    """Fallback: called by the GUI mic button. Returns audio or None."""
    return record_until_silence(on_status=on_status, device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    from speech.transcriber import transcribe
    listener = WakeWordListener()
    print('Say "hey sound", wait for LISTENING, then say your command. Ctrl+C to stop.')
    try:
        while True:
            audio = listener.listen_for_command()
            if audio is None:
                break
            print("Command:", transcribe(audio))
    except KeyboardInterrupt:
        print("\nStopped.")