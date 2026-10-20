"""
speech/record_test_set.py  (Menna + Yousef Waleed)

Guided recording of the audio test set. Each member runs it once per condition:
  python -m speech.record_test_set --speaker menna --condition clean
  python -m speech.record_test_set --speaker menna --condition noisy   (fan/music/street)
It saves WAV files + appends the ground-truth transcript to data/audio_test/transcripts.csv.
Press Enter to record each sentence, 's' to skip, 'q' to quit.
"""
import argparse
import csv
from pathlib import Path

from speech.audio_capture import record_until_silence, save_audio

COMMANDS = [
    ("please open chrome", "polite"), ("close chrome", "short"),
    ("create a folder called AI", "normal"), ("create a file named notes.txt", "normal"),
    ("show files in downloads", "normal"), ("open downloads folder", "short"),
    ("show system information", "normal"), ("what time is it", "question"),
    ("what is the weather today", "question"), ("launch firefox", "synonym"),
    ("could you please close firefox", "polite"), ("open the terminal", "normal"),
    ("what is the date today", "question"), ("make a new folder named projects", "synonym"),
    ("list the files in documents", "synonym"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--condition", choices=["clean", "noisy"], required=True)
    ap.add_argument("--out", default="data/audio_test")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "transcripts.csv"
    new = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "speaker", "condition", "style", "transcript"])
        for i, (text, style) in enumerate(COMMANDS, 1):
            while True:
                ans = input(f'\n[{i}/{len(COMMANDS)}] Say: "{text}"  (Enter=record, s=skip, q=quit) ')
                if ans == "q":
                    return
                if ans == "s":
                    break
                audio = record_until_silence()
                if audio is None:
                    print("Nothing captured, try again.")
                    continue
                name = f"{args.speaker}_{args.condition}_{i:02d}.wav"
                save_audio(str(out / name), audio)
                w.writerow([name, args.speaker, args.condition, style, text])
                f.flush()
                break


if __name__ == "__main__":
    main()
