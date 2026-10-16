import pandas as pd

file_path = r"data/raw/raw_intents_data.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 1. Show all exact duplicate rows
duplicates = df[df.duplicated(keep=False)].sort_values(
    by=["text", "intent"]
)

print("\n=== Exact Duplicate Rows ===")
print(duplicates.to_string(index=False))

# 2. Check whether the same text has different intents
intent_counts = df.groupby("text")["intent"].nunique()

conflicting_texts = intent_counts[intent_counts > 1].index

print("\n=== Texts With Conflicting Intents ===")
print(
    df[df["text"].isin(conflicting_texts)]
    .sort_values("text")
    .to_string(index=False)
)

print("\nNumber of conflicting texts:", len(conflicting_texts))

# 3. Check unique intents
print("\n=== Intent Distribution ===")
print(df["intent"].value_counts())
