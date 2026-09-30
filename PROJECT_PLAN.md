# Sound Operating System

**Team size:** 8 members

---

## 1. Project Overview

Sound Operating System is a project where you **talk to Linux** instead of typing.

You say: *"Open Chrome"* and Chrome opens.
You say: *"Create a folder called AI Project"* and the folder is created.

**How it works in short:**
1. You speak.
2. A speech model turns your voice into text.
3. **Our ML model** understands what you want.
4. Python runs the Linux command.

**Why Linux?** Because it is easy to control with Python (`os`, `subprocess`) and it is free.

**Where is the AI?** The main AI part is the **intent classifier** that we train ourselves.
The speech-to-text part (Whisper) is already trained, so we only use it.

---

## 2. Project Goal

Build a working demo where we speak simple commands and Linux does them.

We want:
- A dataset that **we** made.
- An ML model that **we** trained.
- Results and numbers that show it works.
- A live demo at the end.

**Our targets:**

| What | Target |
|---|---|
| Model accuracy on text | 90% or more |
| Spoken commands that work from start to end | 80% or more |

---

## 3. How the System Works

```text
Voice
  ↓
Speech-to-Text (Whisper)      →  "create a folder called AI Project"
  ↓
Clean the text
  ↓
Our ML Model                  →  intent = CREATE_FOLDER
  ↓
Find the details              →  name = "AI Project"
  ↓
Run the Linux action          →  make the folder
  ↓
Show the result
```

**Simple example:**

| Step | Result |
|---|---|
| You say | "Please open Firefox" |
| Speech-to-Text | `Please open Firefox.` |
| ML model | `OPEN_APPLICATION` |
| Details | app = firefox |
| Linux | Firefox opens |

**Do we need ChatGPT or Gemini?** No. Our small ML model is enough because we only have a few commands. We do not call any LLM in the project.

---

## 4. System Architecture

The project has 4 parts:

| Part | What it does | Who makes it |
|---|---|---|
| Speech | Voice → text | Whisper (already trained) |
| Understanding | Text → intent + details | **Our ML model** + simple rules |
| Actions | Intent → Linux command | Our Python code |
| Main program | Connects everything (`main.py`) | Us |

The main program runs in a loop:

```text
listen → understand → do the action → show result → repeat
```

We will also have a **text mode** (type the command instead of speaking). It helps for testing, and it is our backup if the microphone fails in the demo.

---

## 5. Technologies

| Purpose | Tool |
|---|---|
| Operating system | Ubuntu Linux |
| Language | Python |
| ML | scikit-learn |
| Data | pandas, NumPy |
| Speech-to-Text | Whisper (`faster-whisper`) |
| Microphone | `sounddevice` |
| Linux commands | `subprocess`, `os`, `pathlib` |
| Teamwork | Git and GitHub |

**requirements.txt**

```text
numpy
pandas
scikit-learn
joblib
faster-whisper
sounddevice
pytest
```

We do **not** use: LLM APIs, deep learning training, cloud, Docker, or any GUI framework.

---

## 6. Supported Commands / Intents

We support **8 commands** + one "I don't know" class:

| Intent | Example | Details we need | Linux action |
|---|---|---|---|
| `OPEN_APPLICATION` | "open chrome" | app name | run the app |
| `CLOSE_APPLICATION` | "close chrome" | app name | `pkill` |
| `CREATE_FOLDER` | "create a folder called AI" | folder name | `mkdir` |
| `CREATE_FILE` | "create a file named notes.txt" | file name | create empty file |
| `LIST_FILES` | "show files in downloads" | folder | list the files |
| `OPEN_FOLDER` | "open the downloads folder" | folder | `xdg-open` |
| `SYSTEM_INFO` | "show system information" | none | show OS, CPU, RAM |
| `SHOW_DATE_TIME` | "what time is it" | none | show date and time |
| `UNKNOWN` | "what is the weather" | none | do nothing, say "I did not understand" |

**Apps we support:** Chrome, Firefox, VS Code, Terminal, Calculator, Text Editor.
**Folders we support:** Home, Downloads, Documents, Desktop.

**Tricky pairs** (the model may mix these up, so we test them):
- `OPEN_APPLICATION` and `OPEN_FOLDER` ("open chrome" vs "open downloads")
- `CREATE_FOLDER` and `CREATE_FILE`

**Rule:** one sentence = one command.

---

## 7. Dataset Plan

We need about **600 to 700 sentences**:
- **60 to 70 for each intent**
- **100 or more for `UNKNOWN`**

**How we collect them:** every team member writes their own sentences in their own file.

Each member writes:
- **8 different sentences for every intent** (8 x 8 = 64)
- **15 sentences for `UNKNOWN`** (things we do not support)

So each member writes about 80 sentences. This takes about 1 hour.

**File format** (`data/raw/<name>.csv`):

```csv
text,intent
Open Chrome,OPEN_APPLICATION
Please launch Firefox,OPEN_APPLICATION
Make a new folder called Reports,CREATE_FOLDER
Show me the files in Downloads,LIST_FILES
What is the weather today,UNKNOWN
```

**Tips for writing good sentences:**
- Use different words: open, launch, start, run.
- Use polite and short forms: "please open chrome", "chrome".
- Use different app names and folder names.
- For `UNKNOWN`, also write sentences that are close but not supported ("open the door", "create a playlist").

**Labeling rules:**
- Each sentence has only one intent.
- If two members disagree, the Dataset Lead decides.

---

## 8. Data Preprocessing

**Step 1: Merge** all the member files into one file.

**Step 2: Clean** the data:
- Remove empty rows.
- Remove duplicate sentences.
- Make the text lowercase.
- Remove punctuation (`.`, `,`, `!`, `?`).

```python
import re

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

**Step 3: Split** the data:
- 80% for training
- 20% for testing

```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=42)
```

**Important:** we keep the original text too, because folder names like "AI Project" need their capital letters.

We do **not** remove words like "the" and "in", because they help the model tell "open the downloads folder" from "open chrome".

---

## 9. ML Model

**We choose: TF-IDF + Logistic Regression.**

- **TF-IDF** turns each sentence into numbers.
- **Logistic Regression** learns which numbers mean which intent.

**Why Logistic Regression?**
- It works well for short text.
- It is fast and easy to explain.
- It gives a **confidence score** (for example 0.95), which we use for `UNKNOWN`. If the confidence is too low, we say "I did not understand".

We will also try **Naive Bayes, SVM and Random Forest** and compare the results in a table.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])
```

**Finding the details (app name, folder name)** is done with simple rules, not ML:
- Look for known app names ("chrome", "firefox", "vs code").
- Look for the words after "called" or "named" to get the folder name.
- Look for folder words like "downloads" or "documents".

```python
def find_name(text):
    for word in ["called", "named"]:
        if word in text.lower():
            return text.lower().split(word)[1].strip(" .")
    return None
```

---

## 10. Model Training

**Steps:**
1. Load `train.csv`.
2. Build the pipeline (TF-IDF + Logistic Regression).
3. Train it with `model.fit(...)`.
4. Save it with `joblib.dump(model, "models/intent_model.joblib")`.

```python
import pandas as pd, joblib

train = pd.read_csv("data/processed/train.csv")
model.fit(train["text_clean"], train["intent"])
joblib.dump(model, "models/intent_model.joblib")
```

**Then compare the 4 models** with cross-validation (5 folds) on the training data only, and choose the best one.

**Rules:**
- Use `random_state=42` so results are the same every time.
- Do **not** look at the test set while training.
- Save the first working model as soon as possible (Week 3) so other members can use it.

---

## 11. Model Evaluation

We test the model on the 20% test data that it has never seen.

**Metrics we use:**

| Metric | Meaning |
|---|---|
| Accuracy | How many predictions are correct |
| Precision | When it says "X", how often it is right |
| Recall | Out of all real "X", how many it finds |
| F1-score | Mix of precision and recall |
| Confusion matrix | A table that shows which intents get mixed up |

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**After testing:**
1. Save the wrong predictions to a file.
2. Look at them and find the reason.
3. Add more sentences for the weak intents.
4. Train again.

**Confidence threshold:** try values like 0.4, 0.5, 0.6 and choose the one that gives the best results. Below the threshold the answer is `UNKNOWN`.

---

## 12. Speech-to-Text Integration

We use **Whisper** (`faster-whisper`, model `base.en`). It is already trained, so we do not train it.

```python
import sounddevice as sd
from faster_whisper import WhisperModel

model = WhisperModel("base.en", device="cpu", compute_type="int8")

def listen(seconds=4):
    input("Press Enter and speak...")
    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    segments, _ = model.transcribe(audio.flatten(), language="en")
    return " ".join(s.text for s in segments).strip()
```

**How it connects to our ML model:**

```python
text = listen()                       # "Open Chrome."
intent, confidence = classifier.predict(text)
```

**Tips:**
- Load the Whisper model only once at the start (loading is slow).
- Test with the microphone we will use in the demo.
- If accuracy is low, try the bigger model `small.en`.
- Keep the text mode as a backup.

---

## 13. Linux Command Integration

Each intent has one Python function.

| Intent | Python code |
|---|---|
| `OPEN_APPLICATION` | `subprocess.Popen(["google-chrome"])` |
| `CLOSE_APPLICATION` | `subprocess.run(["pkill", "-x", "chrome"])` |
| `CREATE_FOLDER` | `Path.home() / name` then `.mkdir()` |
| `CREATE_FILE` | `Path.home() / name` then `.touch()` |
| `LIST_FILES` | `os.listdir(path)` |
| `OPEN_FOLDER` | `subprocess.Popen(["xdg-open", path])` |
| `SYSTEM_INFO` | `platform.platform()`, `os.cpu_count()` |
| `SHOW_DATE_TIME` | `datetime.now()` |

**Example:**

```python
import subprocess, shutil

def open_app(app):
    programs = {"chrome": "google-chrome", "firefox": "firefox", "vscode": "code"}
    program = programs.get(app)
    if program and shutil.which(program):
        subprocess.Popen([program])
        return "Opening " + app
    return "App not found"
```

**Safety rules (very important):**
- Never use `shell=True`.
- Only run apps from our allowed list.
- Only create files and folders inside the home folder.
- Do not allow `/` or `..` in folder names.
- Do not support "close terminal" (it would close our own program).

---

## 14. Project Folder Structure

```text
sound-operating-system/
│
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── main.py                 # runs the whole system
│
├── data/
│   ├── raw/                # one CSV file per member
│   └── processed/          # train.csv and test.csv
│
├── preprocessing/
│   ├── clean.py            # clean_text()
│   └── build_dataset.py    # merge, clean, split
│
├── training/
│   ├── train.py            # train the model
│   ├── compare_models.py   # compare 4 models
│   └── evaluate.py         # accuracy, F1, confusion matrix
│
├── models/
│   └── intent_model.joblib # the saved model
│
├── nlp/
│   ├── classifier.py       # loads the model and predicts the intent
│   └── extractor.py        # finds app name, folder name
│
├── speech/
│   └── transcriber.py      # microphone + Whisper
│
├── commands/
│   └── executor.py         # Linux actions
│
├── tests/                  # test files
└── docs/                   # report, slides, results
```

---

## 15. Development Steps

The team follows these 11 steps.

### Step 1: Prepare the Linux environment
- **What we do:** Install Ubuntu (or a VM). Create the GitHub repo and the folders. Choose one laptop for the demo and test its microphone.
- **Who:** Member 1 (Member 7 helps)
- **Output:** A repo that everyone can clone and run.
- **Before the next step:** Every member can run `python main.py` on their computer.

### Step 2: Define the commands
- **What we do:** Confirm the 8 intents and the apps/folders we support (Section 6).
- **Who:** Member 2
- **Output:** The final list of intents, written in the README.
- **Before the next step:** The whole team agrees on the list.

### Step 3: Create the dataset
- **What we do:** Every member writes their own CSV file (8 sentences per intent + 15 `UNKNOWN`).
- **Who:** Member 2 (everyone helps)
- **Output:** About 600 to 700 labeled sentences.
- **Before the next step:** Every intent has at least 60 sentences and `UNKNOWN` has at least 100.

### Step 4: Preprocess the dataset
- **What we do:** Merge the files, remove duplicates, clean the text, and split into train and test.
- **Who:** Member 3
- **Output:** `train.csv` and `test.csv`.
- **Before the next step:** The same sentence is not in both files.

### Step 5: Train the model
- **What we do:** Train Logistic Regression, compare it with 3 other models, and save the best one.
- **Who:** Member 4
- **Output:** `models/intent_model.joblib` and a comparison table.
- **Before the next step:** The saved model loads and predicts correctly.

### Step 6: Evaluate the model
- **What we do:** Calculate accuracy, precision, recall, F1 and the confusion matrix. Look at the mistakes, add more data, and train again.
- **Who:** Member 5 (Member 4 retrains)
- **Output:** Results report and confusion matrix picture.
- **Before the next step:** Accuracy is 90% or more.

### Step 7: Add Speech-to-Text
- **What we do:** Record from the microphone and transcribe with Whisper. Test different team voices.
- **Who:** Member 6
- **Output:** `speech/transcriber.py`
- **Before the next step:** Most clear commands are transcribed correctly.

### Step 8: Connect intents to Linux commands
- **What we do:** Write one function for each intent and the safety rules. Also write the extractor for app names and folder names.
- **Who:** Member 7 (executor) and Member 5 (extractor)
- **Output:** `commands/executor.py` and `nlp/extractor.py`
- **Before the next step:** Each function works when we call it directly.

### Step 9: Integrate everything
- **What we do:** Put all parts together in `main.py`. Test first in text mode, then with voice.
- **Who:** Member 1 (everyone fixes bugs in their own part)
- **Output:** A full working system.
- **Before the next step:** All demo commands work.

### Step 10: Test the whole system
- **What we do:** Test many spoken commands, unknown commands, and wrong inputs. Write down every bug.
- **Who:** Member 8 (everyone fixes their bugs)
- **Output:** Test report.
- **Before the next step:** At least 80% of spoken commands work and there are no big bugs.

### Step 11: Prepare the final demo
- **What we do:** Write the demo script, make slides, write the report, and practice 3 times.
- **Who:** Member 8 (everyone presents a part)
- **Output:** Slides, report, and a backup video.
- **Done when:** We can run the whole demo without problems.

### Weekly plan

| Week | Work |
|---|---|
| 1 | Steps 1 and 2 |
| 2 | Step 3 (dataset), start Step 7 (speech) and Step 8 (Linux) |
| 3 | Steps 4 and 5 |
| 4 | Step 6, finish Steps 7 and 8 |
| 5 | Step 9 (integration) |
| 6 | Steps 10 and 11 (testing and demo) |

Steps 3, 7 and 8 happen **at the same time**, so nobody waits for the others.

---

## 16. Team Roles and Responsibilities

| Member | Role | Main work | Files |
|---|---|---|---|
| **1** | Team Leader / Integration | Sets up the repo, connects all parts, manages tasks | `main.py`, `README.md` |
| **2** | Dataset Lead | Defines intents, collects and merges the data | `data/raw/`, `build_dataset.py` |
| **3** | Preprocessing | Cleans the text, splits the data | `preprocessing/clean.py` |
| **4** | ML Model | Trains and compares the models, saves the model | `training/train.py`, `compare_models.py`, `nlp/classifier.py` |
| **5** | Evaluation + Extractor | Measures the model, finds the mistakes, writes the extractor | `training/evaluate.py`, `nlp/extractor.py` |
| **6** | Speech | Microphone and Whisper | `speech/transcriber.py` |
| **7** | Linux Commands | Writes the Linux actions and safety rules | `commands/executor.py` |
| **8** | Testing + Documentation | Tests everything, writes the report and slides, prepares the demo | `tests/`, `docs/` |

**Everyone also:**
- Writes their own dataset CSV file (Week 2).
- Speaks test commands for the speech test (Weeks 4 and 6).
- Writes one part of the final report about their own work.

**Buddy system:** each pair checks each other's work.

| Pair |
|---|
| Member 1 and Member 8 |
| Member 2 and Member 3 |
| Member 4 and Member 5 |
| Member 6 and Member 7 |

**Why nobody is blocked:** on Day 1 Member 1 creates simple fake versions (stubs) of every function. So Member 6 can build speech and Member 7 can build the Linux actions without waiting for the ML model.

**The functions everyone must follow:**

```python
transcribe / listen()              -> text
clean_text(text)                   -> clean text
classifier.predict(text)           -> (intent, confidence)
extract(intent, text)              -> details (like app or folder name)
execute(intent, details)           -> result message
```

---

## 17. Git/GitHub Workflow

**Rules:**
1. Never push directly to `main`.
2. For every task, create a new branch (example: `ml/train-model`).
3. When done, open a **Pull Request**.
4. Your **buddy** checks it, then Member 1 merges it.
5. Only edit your own files. If you need a change in someone else's file, ask them.
6. Write clear commit messages (example: `add training script`).

**Basic commands:**

```bash
git pull origin main
git checkout -b ml/train-model
git add .
git commit -m "add training script"
git push origin ml/train-model
```

**Other rules:**
- Each member has their own dataset file, so there are no merge conflicts.
- Commit the small model file `intent_model.joblib` so everyone can run the project.
- Use GitHub Issues for tasks and bugs.
- Meet once every week (30 minutes).

---

## 18. Testing Plan

| What to test | How | Who |
|---|---|---|
| Text cleaning | A few test sentences | Member 3 |
| ML model | Test set + confusion matrix | Member 5 |
| App/folder name finder | 50 sentences with known answers | Member 5 |
| Linux actions | Test with a temporary folder | Member 7 |
| Speech | Everyone records commands and we check the text | Member 6 |
| Full system (text) | All 8 intents | Member 8 |
| **Full system (voice)** | About 50 spoken commands from different members | Member 8 |
| Unknown / wrong inputs | Silence, noise, unsupported commands | Member 8 |

**For every spoken test we write down:**
what was said, what Whisper heard, the predicted intent, and whether the action worked.
This shows **which part** failed (speech, ML, or Linux).

**A stage is finished when:** its tests pass, its goal is reached, and its Pull Request is merged.

---

## 19. Final Integration

**Order:**
1. Connect the real ML model and try it in **text mode**.
2. Connect the extractor and the Linux actions.
3. Test all 8 intents by typing.
4. Add Whisper for voice.
5. Test on the demo laptop after a fresh `git clone`.

**main.py (simple idea):**

```python
while True:
    text = listen()                              # or input() in text mode
    intent, confidence = classifier.predict(text)
    details = extract(intent, text)
    result = execute(intent, details)
    print("Heard:", text)
    print("Intent:", intent, confidence)
    print("Result:", result)
```

Printing the heard text, the intent and the confidence is useful for the demo because it **shows that our model is working**.

**After Week 5:** no new features, only bug fixes.

---

## 20. Final Demonstration

We speak these commands:

| # | We say | What should happen |
|---|---|---|
| 1 | "Open Chrome." | Chrome opens |
| 2 | "Open VS Code." | VS Code opens |
| 3 | "Create a folder called AI." | The folder `AI` is created |
| 4 | "Show the files in Downloads." | The files are listed |
| 5 | "Open the Downloads folder." | The folder opens |
| 6 | "Show system information." | System info is shown |
| 7 | "Close Chrome." | Chrome closes |
| 8 | "What is the weather today?" | "I did not understand" (nothing happens) |
| 9 | "Could you fire up Firefox for me?" | Firefox opens (a sentence the model never saw) |

**What we show on the screen:** for each command, the heard text, the intent, and the confidence.

**Also show:** the dataset size, the model comparison table, and the confusion matrix.

**Backup plan:** if the microphone does not work, use text mode. If the laptop fails, play the backup video.
