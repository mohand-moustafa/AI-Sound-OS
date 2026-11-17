# Ubuntu demo walkthrough

## Prepare

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git libportaudio2 libsndfile1 \
  xdg-utils tesseract-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build and verify the intent model

```bash
python -m preprocessing.build_dataset
python -m training.compare_models
python -m training.train
python -m training.evaluate
python -m pytest -q
```

The model is trained from `data/processed/train.csv` and saved to
`models/intent_model.joblib`. Evaluation writes the classification report to
the terminal, plus `docs/confusion_matrix.png` and `docs/errors.csv`.

## Run the assistant

Use text mode first:

```bash
python main.py --text --dry-run
```

Use real Linux actions after validating the dry run:

```bash
python main.py --text
```

For microphone input with push-to-talk-style silence detection:

```bash
python main.py
```

For wake-word mode:

```bash
python main.py --wake-word
```

Say “hey sound”, wait for the listening status, then say one command. The
wake-word mode returns to waiting after each command. It requires a working
microphone, PortAudio-compatible audio input, and the Whisper model download.

## Custom Home folders and Vision

Folder commands accept any safe relative path below the user's Home folder,
including nested folders:

```text
open the AI Project folder
list files in projects
create a folder called reports in projects
create a file called notes.txt inside projects
```

Absolute paths and paths containing `..` are rejected for safety.

Vision commands:

```text
take a screenshot
read the text on the screen
describe the screen
```

Screenshots are saved under `screenshots/`. `READ_SCREEN` and
`DESCRIBE_SCREEN` capture the screen and run Tesseract OCR. They require a
desktop session and `tesseract-ocr`; these commands read visible text but do
not yet identify arbitrary objects or click interface controls. Screen intents
are routed explicitly before the text classifier because they activate local
capture/OCR capabilities.

## Speech evaluation

Collect real recordings; do not create synthetic transcripts:

```bash
python -m speech.record_test_set --speaker <name> --condition clean
python -m speech.record_test_set --speaker <name> --condition noisy
python -m speech.evaluate_wer
```

The evaluator expects `data/audio_test/transcripts.csv`, WAV files, and
optional `wake/` and `other/` directories. It writes
`docs/speech_results.md`. Speech metrics are not valid until real recordings
have been collected on the target Ubuntu machine.

## Troubleshooting

- If no microphone is listed, check Linux audio permissions and
  `python -c "import sounddevice as sd; print(sd.query_devices())"`.
- If Whisper model loading fails, verify network access for the first model
  download and available disk space.
- If OCR fails, install the engine with `sudo apt install tesseract-ocr`.
- Keep `--text --dry-run` as the safe fallback when testing the classifier.
- The executor is Linux-specific: `pkill`, `xdg-open`, `/proc/meminfo`, and
  application names are not portable to Windows.
