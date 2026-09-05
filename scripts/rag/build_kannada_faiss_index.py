import gc
import json
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# Configuration
# ============================================================

EMBEDDING_FILE = Path(
    "data/embeddings/kannada/embeddings.npy"
)

METADATA_FILE = Path(
    "data/embeddings/kannada/metadata.json"
)

OUTPUT_DIR = Path(
    "vector_db/kannada"
)

INDEX_FILE = OUTPUT_DIR / "faiss.index"

OUTPUT_METADATA = OUTPUT_DIR / "metadata.json"


# ============================================================
# Create Output Directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Banner
# ============================================================

print("=" * 70)
print("KANNADA FAISS INDEX BUILDER")
print("=" * 70)

print(f"Embedding File : {EMBEDDING_FILE}")
print(f"Metadata File  : {METADATA_FILE}")
print(f"Output Folder  : {OUTPUT_DIR}")

print("=" * 70)


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

print("\nLoading Kannada embeddings...\n")

embeddings = np.load(
    EMBEDDING_FILE,
    allow_pickle=False
)

print("✓ Embeddings Loaded")

print(f"Shape : {embeddings.shape}")


# ============================================================
# Validate Embeddings
# ============================================================

if embeddings.ndim != 2:

    raise ValueError(
        "Embeddings must be a 2-dimensional array."
    )


dimension = embeddings.shape[1]

print(
    f"Embedding Dimension : {dimension}"
)


if dimension != 768:

    raise ValueError(
        f"Expected 768-dimensional embeddings, "
        f"got {dimension}"
    )


# FAISS works best with float32
embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ============================================================
# Load Metadata
# ============================================================

print("\nLoading Kannada metadata...\n")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print("✓ Metadata Loaded")

print(
    f"Metadata Entries : {len(metadata)}"
)


# ============================================================
# Verify
# ============================================================

if len(metadata) != embeddings.shape[0]:

    raise ValueError(
        "Metadata count does not match "
        "embedding count."
    )


print("\n✓ Metadata / Embedding count matches")


# ============================================================
# Verify Language
# ============================================================

languages = {
    item.get("language")
    for item in metadata
}

print(
    f"Languages Found : {languages}"
)


if languages != {"kn"}:

    raise ValueError(
        f"Expected only Kannada language 'kn', "
        f"found: {languages}"
    )


print("✓ All metadata entries are Kannada")


# ============================================================
# Create FAISS Index
# ============================================================

print("\n" + "=" * 70)
print("Creating Kannada FAISS Index...")
print("=" * 70)


# Embeddings were normalized during generation.
# Therefore Inner Product = Cosine Similarity.

index = faiss.IndexFlatIP(
    dimension
)


print("✓ FAISS Index Created")


# ============================================================
# Add Embeddings
# ============================================================

print("\nAdding Kannada embeddings...")

index.add(
    embeddings
)


print("✓ Embeddings Added")

print(
    f"Total vectors : {index.ntotal}"
)


# ============================================================
# Save Index
# ============================================================

print("\nSaving FAISS index...")

faiss.write_index(
    index,
    str(INDEX_FILE)
)


print(
    f"✓ Saved -> {INDEX_FILE}"
)


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


print(
    f"✓ Saved -> {OUTPUT_METADATA}"
)


# ============================================================
# Verify Saved Index
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED INDEX")
print("=" * 70)


loaded_index = faiss.read_index(
    str(INDEX_FILE)
)


print("✓ FAISS index loaded successfully")

print(
    f"Vectors in index : {loaded_index.ntotal}"
)


if loaded_index.ntotal != len(metadata):

    raise RuntimeError(
        "Verification failed: "
        "FAISS vector count does not match metadata."
    )


print("✓ Vector count matches metadata")


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 70)
print("KANNADA FAISS INDEX COMPLETE")
print("=" * 70)

print(
    f"Embedding Dimension : {dimension}"
)

print(
    f"Total Vectors       : {loaded_index.ntotal}"
)

print(
    f"Index File          : {INDEX_FILE}"
)

print(
    f"Metadata File       : {OUTPUT_METADATA}"
)

print("=" * 70)


# ============================================================
# Cleanup
# ============================================================

del embeddings
del metadata
del index
del loaded_index

gc.collect()

print("\n✓ Memory cleaned")
