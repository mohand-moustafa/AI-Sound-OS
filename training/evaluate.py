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
from preprocessing.clean import cleaned_text_column


def main():
    model = joblib.load(config.MODEL_PATH)
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    y_true = test["intent"]
    y_pred = model.predict(cleaned_text_column(test))

    report = classification_report(y_true, y_pred, digits=3)
    print(report)

    config.DOCS_DIR.mkdir(exist_ok=True)
    (config.DOCS_DIR / "classification_report.txt").write_text(
        report + "\n", encoding="utf-8"
    )

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(config.DOCS_DIR / "confusion_matrix.png")

    errors = test[y_true != y_pred].copy()
    errors["predicted"] = y_pred[(y_true != y_pred).values]
    errors.to_csv(config.DOCS_DIR / "errors.csv", index=False)
    print("Saved docs/classification_report.txt, docs/confusion_matrix.png, "
          f"and docs/errors.csv ({len(errors)} mistakes)")


if __name__ == "__main__":
    main()
