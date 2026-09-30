"""Merge all member files, clean, remove duplicates, split into train/test.
Owner: Member 3 (Member 2 helps).

Run from the project root:
    python -m preprocessing.build_dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from preprocessing.clean import clean_text


def main():
    # Files that start with "_" (like _template.csv) are ignored.
    files = sorted(f for f in config.RAW_DATA_DIR.glob("*.csv") if not f.name.startswith("_"))
    if not files:
        raise SystemExit("No CSV files found in data/raw/. Every member must add their file first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip().str.upper()
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"] != ""]

    # Same sentence with two different intents = labeling mistake. Show it to Member 2.
    conflicts = df.groupby("text_clean")["intent"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts) > 0:
        print("WARNING: these sentences have more than one intent:")
        for sentence in conflicts.index:
            print("  -", sentence, "->", sorted(df[df["text_clean"] == sentence]["intent"].unique()))

    df = df.drop_duplicates(subset="text_clean")

    print("\nSentences per intent:")
    counts = df["intent"].value_counts()
    print(counts.to_string())
    for intent, n in counts.items():
        if n < 60:
            print(f"WARNING: {intent} has only {n} sentences (target: 60 or more)")

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["intent"], random_state=42
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    print(f"\nSaved train.csv ({len(train)} rows) and test.csv ({len(test)} rows)")


if __name__ == "__main__":
    main()
