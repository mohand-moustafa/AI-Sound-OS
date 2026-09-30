"""Intent classifier wrapper. Owner: Member 4.

Loads models/intent_model.joblib and predicts the intent of a sentence.

Until the real model is trained (Week 3), a TEMPORARY keyword stub is used
so that everybody else can already run the whole pipeline.
"""
from pathlib import Path

import config
from preprocessing.clean import clean_text


def _keyword_stub(text):
    """TEMPORARY fake model. It is replaced by the real ML model in Week 3."""
    t = clean_text(text)
    words = t.split()

    def has(*keys):
        return any(k in words for k in keys)

    if has("create", "make", "new") and has("folder", "directory"):
        return "CREATE_FOLDER", 0.90
    if has("create", "make", "new") and has("file"):
        return "CREATE_FILE", 0.90
    if has("close", "quit", "kill", "exit") and not has("folder"):
        return "CLOSE_APPLICATION", 0.90
    if has("open", "launch", "start", "run") and has("folder", "directory"):
        return "OPEN_FOLDER", 0.90
    if has("open", "launch", "start", "run"):
        return "OPEN_APPLICATION", 0.90
    if has("show", "list", "display") and has("file", "files"):
        return "LIST_FILES", 0.90
    if "system" in words:
        return "SYSTEM_INFO", 0.90
    if has("time", "date"):
        return "SHOW_DATE_TIME", 0.90
    return "UNKNOWN", 0.90


class IntentClassifier:
    def __init__(self, model_path=config.MODEL_PATH, threshold=config.THRESHOLD):
        self.threshold = threshold
        self.model = None
        if Path(model_path).exists():
            import joblib
            self.model = joblib.load(model_path)
        else:
            print("[warning] No trained model found. Using the temporary keyword stub.\n")

    def predict(self, text):
        """Return (intent, confidence). Low confidence -> UNKNOWN."""
        if self.model is None:
            return _keyword_stub(text)

        probs = self.model.predict_proba([clean_text(text)])[0]
        best = probs.argmax()
        intent = str(self.model.classes_[best])
        confidence = float(probs[best])
        if confidence < self.threshold:
            return "UNKNOWN", confidence
        return intent, confidence
