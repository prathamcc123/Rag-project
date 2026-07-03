import json
import random
import re
from pathlib import Path
from collections import defaultdict

# ==========================================================
# Domain-Aware MuRIL Training Pair Generator
#
# Creates positive sentence pairs from agriculture chunks
# for fine-tuning Google's MuRIL sentence encoder.
#
# Dataset:
# Vikaspedia Agriculture (Hindi)
# ==========================================================

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

INPUT_DIR = Path("data/chunks/hindi")
OUTPUT_DIR = Path("data/training")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALID_FILE = OUTPUT_DIR / "valid.jsonl"
STATS_FILE = OUTPUT_DIR / "statistics.json"

VALID_SPLIT = 0.05
MIN_WORDS = 20

random.seed(42)

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def clean_text(text):
    """
    Normalize whitespace and remove Vikaspedia boilerplate.
    """

    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    patterns = [
        r"\|\s*Vikaspedia\s*-\s*Agriculture",
        r"\|\s*Vikaspedia",
        r"Vikaspedia\s*-\s*Agriculture",
        r"Vikaspedia"
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    return text.strip(" |-")


def word_count(text):
    return len(text.split())


def get_article_id(chunk_id):
    """
    article_0123_chunk_4

    →

    article_0123
    """

    try:
        return "_".join(chunk_id.split("_")[:2])

    except Exception:
        return None


def get_chunk_number(chunk_id):
    """
    article_0123_chunk_4

    →

    4
    """

    try:
        return int(chunk_id.split("_")[-1])

    except Exception:
        return 0


# ----------------------------------------------------------
# Load Chunk Files
# ----------------------------------------------------------

print("=" * 60)
print("Loading Chunk Files...")
print("=" * 60)

chunk_files = sorted(INPUT_DIR.glob("*.json"))

print(f"Chunk Files Found : {len(chunk_files)}")

articles = defaultdict(list)

skipped = 0

for file in chunk_files:

    try:

        with open(file, "r", encoding="utf-8") as f:
            chunk = json.load(f)

        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            skipped += 1
            continue

        article_id = get_article_id(chunk_id)

        if article_id is None:
            skipped += 1
            continue

        title = clean_text(chunk.get("title", ""))

        content = clean_text(chunk.get("content", ""))

        if not content:
            skipped += 1
            continue

        if word_count(content) < MIN_WORDS:
            skipped += 1
            continue

        chunk["title"] = title
        chunk["content"] = content

        articles[article_id].append(chunk)

    except Exception as e:

        print(f"Error reading {file.name}")

        skipped += 1

print(f"Articles Loaded : {len(articles)}")
print(f"Skipped Chunks  : {skipped}")

# ----------------------------------------------------------
# Sort Chunks
# ----------------------------------------------------------

for article_id in articles:

    articles[article_id].sort(
        key=lambda x: get_chunk_number(x["chunk_id"])
    )

print("Chunks sorted successfully.")

# ----------------------------------------------------------
# Generate Positive Pairs
# ----------------------------------------------------------

pairs = []
seen = set()

print("\nGenerating Training Pairs...")

for article_id, chunks in articles.items():

    # ------------------------------------------
    # Title → Chunk
    # ------------------------------------------

    for chunk in chunks:

        title = chunk["title"]
        content = chunk["content"]

        if title and word_count(title) >= 2:

            key = (title, content)

            if key not in seen:

                seen.add(key)

                pairs.append({

                    "sentence1": title,
                    "sentence2": content,

                    "pair_type": "title_chunk",

                    "source": article_id,

                    "chunk_id": chunk["chunk_id"],

                    "language": chunk.get(
                        "language",
                        "Hindi"
                    ),

                    "url": chunk.get("url", "")

                })

    # ------------------------------------------
    # Adjacent Chunk → Chunk
    # ------------------------------------------

    for i in range(len(chunks) - 1):

        chunk1 = chunks[i]["content"]
        chunk2 = chunks[i + 1]["content"]

        key = (chunk1, chunk2)

        if key in seen:
            continue

        seen.add(key)

        pairs.append({

            "sentence1": chunk1,
            "sentence2": chunk2,

            "pair_type": "chunk_chunk",

            "source": article_id,

            "chunk_id_1": chunks[i]["chunk_id"],
            "chunk_id_2": chunks[i + 1]["chunk_id"],

            "language": chunks[i].get(
                "language",
                "Hindi"
            ),

            "url": chunks[i].get("url", "")

        })

print(f"Total Training Pairs : {len(pairs)}")
# ----------------------------------------------------------
# Shuffle Dataset
# ----------------------------------------------------------

random.shuffle(pairs)

split_index = int(len(pairs) * (1 - VALID_SPLIT))

train_pairs = pairs[:split_index]
valid_pairs = pairs[split_index:]

print(f"Training Pairs   : {len(train_pairs)}")
print(f"Validation Pairs : {len(valid_pairs)}")

# ----------------------------------------------------------
# Save train.jsonl
# ----------------------------------------------------------

with open(TRAIN_FILE, "w", encoding="utf-8") as f:

    for pair in train_pairs:
        f.write(
            json.dumps(
                pair,
                ensure_ascii=False
            ) + "\n"
        )

# ----------------------------------------------------------
# Save valid.jsonl
# ----------------------------------------------------------

with open(VALID_FILE, "w", encoding="utf-8") as f:

    for pair in valid_pairs:
        f.write(
            json.dumps(
                pair,
                ensure_ascii=False
            ) + "\n"
        )

print("\nTraining files saved successfully.")

# ----------------------------------------------------------
# Dataset Statistics
# ----------------------------------------------------------

title_chunk_pairs = sum(
    1 for p in pairs
    if p["pair_type"] == "title_chunk"
)

chunk_chunk_pairs = sum(
    1 for p in pairs
    if p["pair_type"] == "chunk_chunk"
)

avg_chunks = round(
    len(chunk_files) / len(articles),
    2
) if len(articles) else 0

stats = {

    "dataset": "Vikaspedia Agriculture",

    "language": "Hindi",

    "encoder": "google/muril-base-cased",

    "chunk_files": len(chunk_files),

    "articles": len(articles),

    "skipped_chunks": skipped,

    "average_chunks_per_article": avg_chunks,

    "title_chunk_pairs": title_chunk_pairs,

    "chunk_chunk_pairs": chunk_chunk_pairs,

    "total_pairs": len(pairs),

    "training_pairs": len(train_pairs),

    "validation_pairs": len(valid_pairs)

}

with open(
    STATS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stats,
        f,
        ensure_ascii=False,
        indent=4
    )

# ----------------------------------------------------------
# Final Report
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(json.dumps(
    stats,
    indent=4,
    ensure_ascii=False
))

print("\nGenerated Files")
print("-" * 60)

print(f"Train File      : {TRAIN_FILE}")
print(f"Validation File : {VALID_FILE}")
print(f"Statistics File : {STATS_FILE}")

print("=" * 60)
print("Dataset generation completed successfully.")
print("=" * 60)