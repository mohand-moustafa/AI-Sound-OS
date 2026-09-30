# Sound Operating System

Control Linux with your **voice**.
Say *"Open Chrome"* or *"Create a folder called AI Project"* and the system does it.

```text
Voice → Speech-to-Text (Whisper) → Our ML model (intent) → Details → Linux action
```

The main AI part is the **intent classifier** that our team built and trained (TF-IDF + Logistic Regression).
Read [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full plan, steps, and team roles.

## Supported commands

`OPEN_APPLICATION` · `CLOSE_APPLICATION` · `CREATE_FOLDER` · `CREATE_FILE` · `LIST_FILES` · `OPEN_FOLDER` · `SYSTEM_INFO` · `SHOW_DATE_TIME` · `UNKNOWN`

## Quick start

```bash
git clone <repo-url>
cd sound-operating-system
bash scripts/setup_linux.sh
source .venv/bin/activate

python main.py --text --dry-run   # type commands, nothing is executed
python main.py --text             # type commands, Linux actions run
python main.py                    # voice mode (microphone)
```

Until the real model is trained, a temporary keyword stub is used, so everything already runs.

## How to build the model

Run all commands from the project root:

```bash
python -m preprocessing.build_dataset   # merge data/raw/*.csv, clean, split train/test
python -m training.compare_models       # compare 4 models
python -m training.train                # train and save models/intent_model.joblib
python -m training.evaluate             # metrics, confusion matrix, errors
python -m pytest                        # run all tests
```

## Folder structure

```text
sound-operating-system/
├── main.py                 # runs the whole system
├── config.py               # paths, threshold, intents
├── PROJECT_PLAN.md
├── requirements.txt
│
├── data/
│   ├── raw/                # one CSV per member (see _template.csv)
│   └── processed/          # train.csv, test.csv (created by build_dataset)
├── preprocessing/          # clean.py, build_dataset.py
├── training/               # train.py, compare_models.py, evaluate.py
├── models/                 # intent_model.joblib
├── nlp/                    # classifier.py, extractor.py
├── speech/                 # transcriber.py (Whisper)
├── commands/               # executor.py (Linux actions)
├── tests/
├── scripts/                # setup_linux.sh
├── docs/                   # report, slides, results
└── .github/                # PR and issue templates
```

## Who owns what

| Member | Role | Files |
|---|---|---|
| 1 | Team Leader / Integration | `main.py`, `config.py`, `README.md` |
| 2 | Dataset Lead | `data/raw/`, `preprocessing/build_dataset.py` |
| 3 | Preprocessing | `preprocessing/clean.py` |
| 4 | ML Model | `training/train.py`, `training/compare_models.py`, `nlp/classifier.py`, `models/` |
| 5 | Evaluation + Extractor | `training/evaluate.py`, `nlp/extractor.py` |
| 6 | Speech | `speech/transcriber.py` |
| 7 | Linux Commands | `commands/executor.py`, `scripts/` |
| 8 | Testing + Documentation | `tests/`, `docs/` |

## Adding your dataset file

1. Copy `data/raw/_template.csv` to `data/raw/<yourname>.csv`.
2. Write 8 sentences for every intent + 15 `UNKNOWN` sentences, in your own words.
3. Columns are `text,intent` (intent names must match exactly).

## Git rules

1. Never push directly to `main`.
2. One task = one issue = one branch = one Pull Request.
3. Branch names: `data/<name>`, `ml/<topic>`, `speech/<topic>`, `cmd/<topic>`, `docs/<topic>`, `fix/<topic>`.
4. Your buddy reviews your PR, then Member 1 merges it.
5. Only edit your own files. Need a change in someone else's file? Open an issue.
6. Commit messages: `add training script`, `fix folder name bug`.

```bash
git pull origin main
git checkout -b ml/train-model
git add .
git commit -m "add training script"
git push origin ml/train-model     # then open a Pull Request on GitHub
```
