import json
import gc
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

# ============================================================
# Configuration
# ============================================================

import argparse

parser = argparse.ArgumentParser(
    description="Generate embeddings using a specified SentenceTransformer model."
)

parser.add_argument(
    "model_path",
    type=str,
    help="Path to the SentenceTransformer model"
)

args = parser.parse_args()

MODEL_PATH = args.model_path

CHUNK_DIR = Path("data/chunks/hindi")

OUTPUT_DIR = Path("data/embeddings")

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

FINAL_EMBEDDING_FILE = OUTPUT_DIR / "embeddings.npy"

FINAL_METADATA_FILE = OUTPUT_DIR / "metadata.json"

BATCH_SIZE = 4

SAVE_EVERY = 500

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# Create folders
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Banner
# ============================================================

print("=" * 65)
print("Fine-Tuned MuRIL Embedding Generator")
print("=" * 65)

print(f"Device            : {DEVICE}")
print(f"Model             : {MODEL_PATH}")
print(f"Chunk Directory   : {CHUNK_DIR}")
print(f"Output Directory  : {OUTPUT_DIR}")
print(f"Batch Size        : {BATCH_SIZE}")
print(f"Checkpoint Every  : {SAVE_EVERY} chunks")
print("=" * 65)

# ============================================================
# Load Fine-Tuned Model
# ============================================================

print("\nLoading Fine-Tuned MuRIL model...\n")

model = SentenceTransformer(
    MODEL_PATH,
    device=DEVICE
)

print("✓ Model Loaded Successfully")

print(f"Embedding Dimension : {model.get_embedding_dimension()}")
# ============================================================
# Read Chunk Files
# ============================================================

chunk_files = sorted(CHUNK_DIR.glob("*.json"))

if len(chunk_files) == 0:
    raise FileNotFoundError(
        f"No chunk files found inside {CHUNK_DIR}"
    )

print(f"\nTotal Chunk Files Found : {len(chunk_files)}")

# ============================================================
# Resume Support
# ============================================================

existing_metadata = []

existing_embeddings = []

processed_files = set()

metadata_checkpoint = CHECKPOINT_DIR / "metadata_checkpoint.json"

if metadata_checkpoint.exists():

    print("\nCheckpoint found.")

    with open(metadata_checkpoint, "r", encoding="utf-8") as f:

        existing_metadata = json.load(f)

    processed_files = {
        item["chunk_id"]
        for item in existing_metadata
    }

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob("embeddings_part_*.npy")
    )

    for emb_file in checkpoint_files:

        existing_embeddings.append(
            np.load(emb_file)
        )

    print(f"Resuming from {len(processed_files)} processed chunks.")

else:

    print("\nNo previous checkpoint found.")
    print("Starting from scratch.")

print()
print("=" * 65)
print("Ready to generate embeddings...")
print("=" * 65)
# ============================================================
# Read Remaining Chunks
# ============================================================

texts = []
metadata = []
current_batch_chunk_ids = []

print("\nScanning chunk files...\n")

for file in tqdm(chunk_files, desc="Reading Chunks"):

    try:

        with open(file, "r", encoding="utf-8") as f:
            chunk = json.load(f)

        chunk_id = chunk.get("chunk_id")

        if chunk_id in processed_files:
            continue

        text = chunk.get("content", "").strip()

        if len(text) == 0:
            continue

        texts.append(text)

        metadata.append(
            {
                "chunk_id": chunk_id,
                "title": chunk.get("title"),
                "summary": chunk.get("summary"),
                "language": chunk.get("language"),
                "url": chunk.get("url"),
                "content": text,
            }
        )

        current_batch_chunk_ids.append(chunk_id)

    except Exception as e:

        print(f"\nSkipping {file.name}")
        print(e)

print("\n==============================================")
print(f"New Chunks To Process : {len(texts)}")
print("==============================================")

# Nothing left?

if len(texts) == 0:

    print("\nEverything already embedded.")
    sys.exit(0)

# ============================================================
# Generate Embeddings
# ============================================================

print("\nGenerating embeddings...\n")

total = len(texts)

start = 0

checkpoint_index = len(existing_embeddings)

while start < total:

    end = min(start + SAVE_EVERY, total)

    batch_texts = texts[start:end]

    batch_metadata = metadata[start:end]

    print(
        f"\nProcessing Chunks {start + 1} -> {end}"
    )

    with torch.no_grad():

        batch_embeddings = model.encode(

            batch_texts,

            batch_size=BATCH_SIZE,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True,

        )

    checkpoint_file = (
        CHECKPOINT_DIR
        / f"embeddings_part_{checkpoint_index:04d}.npy"
    )

    np.save(checkpoint_file, batch_embeddings)

    existing_embeddings.append(batch_embeddings)

    existing_metadata.extend(batch_metadata)

    with open(
        metadata_checkpoint,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            existing_metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\n✓ Checkpoint Saved -> {checkpoint_file.name}"
    )

    print(
        f"Processed : {len(existing_metadata)} chunks"
    )

    checkpoint_index += 1

    start = end

    # ----------------------------------------------------
    # Free Memory
    # ----------------------------------------------------

    del batch_embeddings
    del batch_texts
    del batch_metadata

    gc.collect()

    if DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

print("\n==============================================")
print("Embedding Generation Finished")
print("==============================================")
# ============================================================
# Merge All Checkpoints
# ============================================================

print("\n")
print("=" * 65)
print("Merging checkpoint files...")
print("=" * 65)

all_embeddings = []

checkpoint_files = sorted(
    CHECKPOINT_DIR.glob("embeddings_part_*.npy")
)

if len(checkpoint_files) == 0:
    raise RuntimeError("No checkpoint files found.")

for file in tqdm(checkpoint_files, desc="Loading Checkpoints"):

    emb = np.load(file)

    all_embeddings.append(emb)

final_embeddings = np.vstack(all_embeddings)

print("\nFinal Embedding Shape :", final_embeddings.shape)

# ============================================================
# Save Final Embeddings
# ============================================================

print("\nSaving final embeddings...")

np.save(
    FINAL_EMBEDDING_FILE,
    final_embeddings,
)

print(f"✓ Saved -> {FINAL_EMBEDDING_FILE}")

# ============================================================
# Save Final Metadata
# ============================================================

print("\nSaving metadata...")

with open(
    FINAL_METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        existing_metadata,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"✓ Saved -> {FINAL_METADATA_FILE}")

# ============================================================
# Verify
# ============================================================

print("\nVerifying output...")

loaded_embeddings = np.load(FINAL_EMBEDDING_FILE)

assert len(existing_metadata) == loaded_embeddings.shape[0]

print("✓ Verification Successful")

# ============================================================
# Cleanup
# ============================================================

print("\nCleaning temporary memory...")

del all_embeddings
del final_embeddings
del loaded_embeddings

gc.collect()

if DEVICE == "mps":
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 65)
print("EMBEDDING GENERATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Device                : {DEVICE}")
print(f"Model                 : {MODEL_PATH}")
print(f"Embedding Dimension   : {model.get_embedding_dimension()}")
print(f"Total Chunks          : {len(existing_metadata)}")
print(f"Output Embeddings     : {FINAL_EMBEDDING_FILE}")
print(f"Output Metadata       : {FINAL_METADATA_FILE}")
print(f"Checkpoint Directory  : {CHECKPOINT_DIR}")

print("=" * 65)

print("\nProject is now ready for FAISS Index Generation.\n")