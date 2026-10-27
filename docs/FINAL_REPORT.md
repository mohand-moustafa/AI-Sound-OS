# Local completion report

## Scope and environment

This report covers the isolated local worktree used for validation. No commit,
push, pull request, remote, or shared checkout was changed.

- Platform tested: Windows, Python 3.13
- Virtual environment: `.venv`
- Target platform: Ubuntu Linux with a microphone
- Intent dataset: `data/raw/raw_intents_data.csv`
- Processed dataset: 802 unique rows, split into 641 training and 161 test rows

## Fixes completed

1. Unified text normalization through `preprocessing.clean.cleaned_text_column`.
   Training, comparison, evaluation, and live inference now use the same
   `clean_text` behavior.
2. Regenerated `data/processed/train.csv` and `test.csv` with the canonical
   `text_clean` column.
3. Trained the real TF-IDF + Logistic Regression model in the project virtual
   environment.
4. Added model metadata, including preprocessing name, labels, training row
   count, seed, and confidence threshold.
5. Tuned the runtime confidence threshold to `0.35`, which prevents the normal
   demo commands from being rejected as `UNKNOWN` while retaining the
   classifier's low-confidence fallback.
6. Added a saved classification report and a preprocessing consistency test.
7. Added `--wake-word` integration to `main.py`, with a clear audio failure
   message and return to wake-word listening after each command.
8. Made executor home-directory behavior respect `HOME`, allowing isolated
   testing and preserving Linux behavior.
9. Added the missing `soundfile` dependency.
10. Added Ubuntu setup, demo, speech-data, troubleshooting, and limitation
    instructions in `docs/DEMO.md`.

## Real model results

The model was trained with:

```text
python -m training.train
```

Artifacts:

- `models/intent_model.joblib`
- `models/intent_model_metadata.json`

Held-out test evaluation:

| Metric | Result |
|---|---:|
| Accuracy | 98.1% |
| Macro precision | 98.9% |
| Macro recall | 98.9% |
| Macro F1 | 98.9% |
| Test rows | 161 |
| Misclassified rows | 3 |

The full report is in `docs/classification_report.txt`; the confusion matrix
is `docs/confusion_matrix.png`; the three errors are in `docs/errors.csv`.

Five-fold training-set comparison was also run:

| Model | Macro F1 mean |
|---|---:|
| Logistic Regression | 0.968 |
| Linear SVM | 0.983 |
| Naive Bayes | 0.944 |
| Random Forest | 0.967 |

The project plan specifies Logistic Regression as the deployable model because
it provides probabilities for confidence-based `UNKNOWN` handling. The
comparison outputs are in `docs/model_comparison.md` and
`models/model_comparison_results.csv`.

## Commands executed

| Command | Result |
|---|---|
| `python -m preprocessing.build_dataset` | Passed; 802 unique rows, 641/161 split |
| `python -m training.compare_models` | Passed; comparison report generated |
| `python -m training.train` | Passed; model and metadata generated |
| `python -m training.evaluate` | Passed; report, matrix, and errors generated |
| `.venv\Scripts\python.exe -m pytest -q` | Passed; 45 tests passed |
| `.venv\Scripts\python.exe -m compileall -q .` | Passed |
| Import check for all requirements | Passed |
| `python main.py --text --dry-run` | Passed for create folder, create file, open folder, and unknown command |
| `python -m speech.evaluate_wer --no-wake` | Not run successfully; recordings are absent |

The speech evaluator stopped with:

```text
FileNotFoundError: data/audio_test/transcripts.csv
```

No WER, latency, wake-word, or end-to-end spoken-command metric has been
invented or reported.

## Remaining Ubuntu validation

1. Install the packages in `docs/DEMO.md` on the target Ubuntu laptop.
2. Confirm the microphone is visible through `sounddevice`.
3. Run `python main.py --text --dry-run` and then `python main.py --text`.
4. Test Linux-specific actions such as `xdg-open`, `pkill`, application
   discovery, and `/proc/meminfo`.
5. Collect real clean and noisy speech recordings with
   `speech.record_test_set`.
6. Run `speech.evaluate_wer` and review `docs/speech_results.md`.
7. Test `python main.py --wake-word` with “hey sound”.
8. Execute the final demo commands and record end-to-end success/failure.

## Known limitations

- The current validation host is Windows, while command execution is designed
  for Ubuntu/Linux.
- No audio recordings are included, so speech quality is not validated.
- Whisper model download and microphone behavior require Ubuntu/network/audio
  validation.
- The fallback keyword classifier remains in the code for development recovery,
  but the generated intent model is now available and is loaded by default.
