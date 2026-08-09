import pandas as pd
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

INPUT_FILE = "data/training/final_dataset.csv"
OUTPUT_FILE = "data/training/final_dataset_prepared.csv"

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("=" * 60)
print(f"Loaded {len(df):,} rows")
print("=" * 60)

# ==========================================
# CREATE TRAINING TEXT
# ==========================================

df["positive_text"] = (
    "Title: "
    + df["title"].fillna("").astype(str)
    + "\n\nContent:\n"
    + df["positive"].fillna("").astype(str)
)

# Remove rows with missing question or content
df = df.dropna(subset=["question", "positive_text"])

# Remove duplicate question-content pairs
df = df.drop_duplicates(subset=["question", "positive_text"])

# ==========================================
# SAVE
# ==========================================

Path("data/training").mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nPrepared dataset saved to:\n{OUTPUT_FILE}")
print(f"Final rows: {len(df):,}")