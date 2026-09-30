"""Sound Operating System - main program. Owner: Member 1.

Run:
    python main.py                      # voice mode (microphone + Whisper)
    python main.py --text               # type commands instead of speaking
    python main.py --text --dry-run     # type commands, do NOT run the Linux actions
"""
import argparse

from commands.executor import execute
from nlp.classifier import IntentClassifier
from nlp.extractor import extract

EXIT_WORDS = ("exit", "quit", "stop listening")


def get_text(text_mode):
    if text_mode:
        return input("> ").strip()
    # Imported here so text mode works even without a microphone / Whisper installed.
    from speech.transcriber import listen
    return listen()


def run(text_mode=False, dry_run=False):
    classifier = IntentClassifier()
    print("Sound Operating System is ready. Say (or type) 'exit' to stop.\n")
    while True:
        try:
            text = get_text(text_mode)
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower().strip(" .!?") in EXIT_WORDS:
            break

        intent, confidence = classifier.predict(text)
        params = extract(intent, text)
        result = execute(intent, params, dry_run=dry_run)

        # This printout is what we show in the demo: it proves our model is working.
        print(f"Heard:   {text}")
        print(f"Intent:  {intent} ({confidence:.2f})")
        print(f"Details: {params}")
        print(f"Result:  {result.message}\n")
    print("Bye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sound Operating System")
    parser.add_argument("--text", action="store_true", help="type commands instead of speaking")
    parser.add_argument("--dry-run", action="store_true", help="do not execute Linux actions")
    args = parser.parse_args()
    run(text_mode=args.text, dry_run=args.dry_run)
