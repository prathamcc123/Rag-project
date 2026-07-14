import gc
import json
from pathlib import Path

import faiss
import numpy as np

# ============================================================
# Configuration
# ============================================================

EMBEDDING_FILE = Path("data/embeddings/embeddings.npy")

METADATA_FILE = Path("data/embeddings/metadata.json")

OUTPUT_DIR = Path("vector_db")

INDEX_FILE = OUTPUT_DIR / "faiss.index"

OUTPUT_METADATA = OUTPUT_DIR / "metadata.json"

# ============================================================
# Create Output Directory
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Banner
# ============================================================

print("=" * 65)
print("FAISS Index Builder")
print("=" * 65)

print(f"Embedding File : {EMBEDDING_FILE}")
print(f"Metadata File  : {METADATA_FILE}")
print(f"Output Folder  : {OUTPUT_DIR}")

print("=" * 65)

# ============================================================
# Check Files
# ============================================================

if not EMBEDDING_FILE.exists():
    raise FileNotFoundError(
        f"Embedding file not found: {EMBEDDING_FILE}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )

# ============================================================
# Load Embeddings
# ============================================================

print("\nLoading embeddings...\n")

embeddings = np.load(
    EMBEDDING_FILE,
    allow_pickle=False
)

print("✓ Embeddings Loaded")

print(f"Shape : {embeddings.shape}")

dimension = embeddings.shape[1]

print(f"Embedding Dimension : {dimension}")

# ============================================================
# Load Metadata
# ============================================================

print("\nLoading metadata...\n")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print("✓ Metadata Loaded")

print(f"Metadata Entries : {len(metadata)}")

# ============================================================
# Verify
# ============================================================

if len(metadata) != embeddings.shape[0]:

    raise ValueError(
        "Metadata count does not match embedding count."
    )

print("\nVerification Passed.")

print("=" * 65)
print("Ready to build FAISS index...")
print("=" * 65)
# ============================================================
# Build FAISS Index
# ============================================================

print("\n")
print("=" * 65)
print("Creating FAISS Index...")
print("=" * 65)

# We use Inner Product because embeddings were normalized
# during generation. Inner Product = Cosine Similarity.

index = faiss.IndexFlatIP(dimension)

print("✓ FAISS Index Created")

print("\nAdding embeddings to FAISS...")

index.add(embeddings)

print("✓ Embeddings Added")

print(f"Total vectors in index : {index.ntotal}")

# ============================================================
# Save FAISS Index
# ============================================================

print("\nSaving FAISS index...")

faiss.write_index(
    index,
    str(INDEX_FILE)
)

print(f"✓ Saved -> {INDEX_FILE}")

# ============================================================
# Save Metadata
# ============================================================

print("\nSaving metadata...")

with open(
    OUTPUT_METADATA,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"✓ Saved -> {OUTPUT_METADATA}")

print("\n")
print("=" * 65)
print("FAISS Index Created Successfully")
print("=" * 65)
# ============================================================
# Verify Saved FAISS Index
# ============================================================

print("\n")
print("=" * 65)
print("Verifying Saved FAISS Index...")
print("=" * 65)

loaded_index = faiss.read_index(
    str(INDEX_FILE)
)

print("✓ FAISS Index Loaded Successfully")

if loaded_index.ntotal != len(metadata):
    raise RuntimeError(
        "Verification failed: Number of vectors does not match metadata."
    )

print("✓ Verification Passed")

total_vectors = loaded_index.ntotal

print(f"Vectors in Index : {total_vectors}")
# ============================================================
# Cleanup Memory
# ============================================================

print("\nCleaning memory...")

del embeddings
del metadata
del index
del loaded_index

gc.collect()

print("✓ Memory Cleaned")

# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 65)
print("FAISS INDEX GENERATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Embedding Dimension : {dimension}")
print(f"Total Vectors       : {total_vectors}")
print(f"Index File          : {INDEX_FILE}")
print(f"Metadata File       : {OUTPUT_METADATA}")

print("=" * 65)

print("\nProject is now ready for Retrieval Testing.\n")