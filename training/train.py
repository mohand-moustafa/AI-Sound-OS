"""Train the intent classifier (TF-IDF + Logistic Regression). Owner: Member 4.

Run from the project root:
    python -m training.train
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main():
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    model = build_model()
    model.fit(train["text_clean"], train["intent"])
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print("Model saved to", config.MODEL_PATH)


if __name__ == "__main__":
    main()
