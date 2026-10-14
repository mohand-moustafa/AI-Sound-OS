
import pandas as pd
from pathlib import Path

# 1. Define file paths
input_path = Path("data/raw/raw_intents_data.csv")
output_path = Path("data/processed/intents_clean.csv")

# 2. Load raw dataset
df = pd.read_csv(input_path)

print("Before cleaning:", df.shape)

# 3. Remove leading and trailing whitespace
df["text"] = df["text"].str.strip()
df["intent"] = df["intent"].str.strip()

# 4. Remove rows with empty or missing values
df = df.dropna(subset=["text", "intent"])
df = df[
    df["text"].ne("") &
    df["intent"].ne("")
]

# 5. Remove exact duplicate examples
df = df.drop_duplicates(
    subset=["text", "intent"],
    keep="first"
)

# 6. Reset row index
df = df.reset_index(drop=True)

# 7. Create output folder and save
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")

# 8. Print cleaning report
print("\nAfter cleaning:", df.shape)
print("Rows removed:", 837 - len(df))
print("Remaining duplicates:", df.duplicated(
    subset=["text", "intent"]
).sum())

print("\nExamples per intent:")
print(df["intent"].value_counts())

print("\nSaved cleaned dataset to:", output_path)
