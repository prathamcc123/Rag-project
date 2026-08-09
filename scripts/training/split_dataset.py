import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = "data/training/final_dataset_prepared.csv"
OUTPUT_DIR = "data/splits"

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("=" * 60)
print("Dataset Statistics")
print("=" * 60)
print(f"Total Question-Passage Pairs : {len(df):,}")
print(f"Unique Documents            : {df['document_id'].nunique():,}")

# ==========================================================
# DOCUMENT-LEVEL SPLIT
# ==========================================================

doc_ids = df["document_id"].unique()

np.random.seed(RANDOM_SEED)
np.random.shuffle(doc_ids)

n_docs = len(doc_ids)

train_end = int(TRAIN_RATIO * n_docs)
valid_end = int((TRAIN_RATIO + VALID_RATIO) * n_docs)

train_docs = set(doc_ids[:train_end])
valid_docs = set(doc_ids[train_end:valid_end])
test_docs = set(doc_ids[valid_end:])

train_df = df[df["document_id"].isin(train_docs)].reset_index(drop=True)
valid_df = df[df["document_id"].isin(valid_docs)].reset_index(drop=True)
test_df = df[df["document_id"].isin(test_docs)].reset_index(drop=True)

# ==========================================================
# VERIFY
# ==========================================================

assert len(train_docs & valid_docs) == 0
assert len(train_docs & test_docs) == 0
assert len(valid_docs & test_docs) == 0

assert len(train_df) + len(valid_df) + len(test_df) == len(df)

print("\n✅ No document overlap detected.")
print("✅ All rows accounted for.")

# ==========================================================
# SAVE
# ==========================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
valid_df.to_csv(f"{OUTPUT_DIR}/valid.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("Split Statistics")
print("=" * 60)

print(f"Train : {len(train_df):,}")
print(f"Valid : {len(valid_df):,}")
print(f"Test  : {len(test_df):,}")

print("\nSaved:")
print(f"{OUTPUT_DIR}/train.csv")
print(f"{OUTPUT_DIR}/valid.csv")
print(f"{OUTPUT_DIR}/test.csv")