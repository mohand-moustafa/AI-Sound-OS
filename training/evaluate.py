"""Evaluate the saved model on the test set. Owner: Member 5.

Prints the classification report, saves the confusion matrix picture
and the list of wrong predictions.

Run from the project root:
    python -m training.evaluate
"""
import joblib
import matplotlib
matplotlib.use("Agg")   # draw pictures without a screen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import config


def main():
    model = joblib.load(config.MODEL_PATH)
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    y_true = test["intent"]
    y_pred = model.predict(test["text_clean"])

    print(classification_report(y_true, y_pred, digits=3))

    config.DOCS_DIR.mkdir(exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(config.DOCS_DIR / "confusion_matrix.png")

    errors = test[y_true != y_pred].copy()
    errors["predicted"] = y_pred[(y_true != y_pred).values]
    errors.to_csv(config.DOCS_DIR / "errors.csv", index=False)
    print(f"Saved docs/confusion_matrix.png and docs/errors.csv ({len(errors)} mistakes)")


if __name__ == "__main__":
    main()
