
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Load cleaned dataset
input_path = Path("data/processed/intents_clean.csv")
df = pd.read_csv(input_path)

# 2. Separate features and target
X = df["text"]
y = df["intent"]

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Rebuild train and test DataFrames
train_df = pd.DataFrame({
    "text": X_train,
    "intent": y_train
})

test_df = pd.DataFrame({
    "text": X_test,
    "intent": y_test
})

# 5. Save the split datasets
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

# 6. Print results
print("Original dataset:", len(df))
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining intent distribution:")
print(train_df["intent"].value_counts())

print("\nTesting intent distribution:")
print(test_df["intent"].value_counts())

print("\nTrain and test files saved successfully!")
