"""Train the intent classifier (TF-IDF + Logistic Regression). Owner: Member 4.

Run from the project root:
    python -m training.train
"""
import json

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from preprocessing.clean import cleaned_text_column


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def main():
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    model = build_model()
    model.fit(cleaned_text_column(train), train["intent"])
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    metadata = {
        "model": "TF-IDF + Logistic Regression",
        "preprocessing": "preprocessing.clean.clean_text",
        "training_data": str(config.PROCESSED_DIR / "train.csv"),
        "training_rows": len(train),
        "labels": sorted(train["intent"].unique().tolist()),
        "random_state": 42,
        "confidence_threshold": config.THRESHOLD,
        "scikit_learn_version": sklearn.__version__,
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="utf-8")
    print("Model saved to", config.MODEL_PATH)
    print("Metadata saved to", config.MODEL_METADATA_PATH)


if __name__ == "__main__":
    main()
