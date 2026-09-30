"""Compare 4 models with 5-fold cross-validation on the TRAIN data only.
Owner: Member 4.

Run from the project root:
    python -m training.compare_models
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}


def main():
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, clf in MODELS.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", clf)])
        scores = cross_val_score(pipe, train["text_clean"], train["intent"], cv=cv, scoring="f1_macro")
        rows.append((name, scores.mean(), scores.std()))
        print(f"{name:22s} F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    config.DOCS_DIR.mkdir(exist_ok=True)
    with open(config.DOCS_DIR / "model_comparison.md", "w") as f:
        f.write("| Model | Macro F1 (mean) | Std |\n|---|---|---|\n")
        for name, mean, std in rows:
            f.write(f"| {name} | {mean:.3f} | {std:.3f} |\n")
    print("Saved docs/model_comparison.md")


if __name__ == "__main__":
    main()
