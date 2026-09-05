import json
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

EMBEDDING_FILE = Path(
    "data/embeddings/kannada/embeddings.npy"
)

METADATA_FILE = Path(
    "data/embeddings/kannada/metadata.json"
)


# ============================================================
# BANNER
# ============================================================

print("=" * 70)
print("KANNADA EMBEDDING VALIDATION")
print("=" * 70)


# ============================================================
# CHECK FILES
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
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(EMBEDDING_FILE)

print(f"Embedding shape : {embeddings.shape}")
print(f"Data type       : {embeddings.dtype}")


# ============================================================
# BASIC SHAPE VALIDATION
# ============================================================

if embeddings.ndim != 2:
    raise ValueError(
        f"Expected 2D embeddings, got {embeddings.ndim}D"
    )

num_embeddings, embedding_dimension = embeddings.shape

if embedding_dimension != 768:
    raise ValueError(
        f"Expected embedding dimension 768, "
        f"got {embedding_dimension}"
    )

print("✓ Embedding dimension is 768")


# ============================================================
# CHECK NaN / INFINITY
# ============================================================

print("\nChecking for invalid values...")

nan_count = np.isnan(embeddings).sum()
inf_count = np.isinf(embeddings).sum()

print(f"NaN values      : {nan_count}")
print(f"Infinite values : {inf_count}")

if nan_count > 0 or inf_count > 0:
    raise ValueError(
        "Invalid NaN or infinite values found."
    )

print("✓ No NaN or infinite values")


# ============================================================
# CHECK NORMS
# ============================================================

print("\nChecking embedding normalization...")

norms = np.linalg.norm(
    embeddings,
    axis=1
)

print(f"Minimum norm : {norms.min():.6f}")
print(f"Maximum norm : {norms.max():.6f}")
print(f"Average norm : {norms.mean():.6f}")

normalized_count = np.sum(
    np.isclose(norms, 1.0, atol=1e-4)
)

print(
    f"Normalized vectors : "
    f"{normalized_count}/{len(norms)}"
)

if normalized_count != len(norms):
    print(
        "WARNING: Not all embeddings appear "
        "to be normalized."
    )
else:
    print("✓ All embeddings are normalized")


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(f"Metadata records : {len(metadata)}")


# ============================================================
# CHECK COUNT MATCH
# ============================================================

if len(metadata) != num_embeddings:

    raise ValueError(
        f"Count mismatch: "
        f"{num_embeddings} embeddings vs "
        f"{len(metadata)} metadata records"
    )

print("✓ Embedding count matches metadata count")


# ============================================================
# CHECK METADATA STRUCTURE
# ============================================================

print("\nChecking metadata structure...")

required_fields = {
    "chunk_id",
    "title",
    "summary",
    "language",
    "url",
    "content",
}


missing_records = 0
empty_content = 0
chunk_ids = []


for index, item in enumerate(metadata):

    missing = required_fields - item.keys()

    if missing:

        print(
            f"WARNING: Record {index} missing: "
            f"{sorted(missing)}"
        )

        missing_records += 1

    chunk_id = item.get("chunk_id")

    if chunk_id:
        chunk_ids.append(chunk_id)

    content = item.get("content", "")

    if not content.strip():
        empty_content += 1


print(f"Missing-field records : {missing_records}")
print(f"Empty-content records : {empty_content}")


# ============================================================
# CHECK DUPLICATE CHUNK IDS
# ============================================================

print("\nChecking duplicate chunk IDs...")

unique_chunk_ids = set(chunk_ids)

duplicate_count = (
    len(chunk_ids) - len(unique_chunk_ids)
)

print(f"Unique chunk IDs : {len(unique_chunk_ids)}")
print(f"Duplicate IDs    : {duplicate_count}")

if duplicate_count == 0:
    print("✓ No duplicate chunk IDs")
else:
    print("WARNING: Duplicate chunk IDs found")


# ============================================================
# CHECK KANNADA LANGUAGE
# ============================================================

print("\nChecking language metadata...")

language_counts = {}

for item in metadata:

    language = item.get(
        "language",
        "UNKNOWN"
    )

    language_counts[language] = (
        language_counts.get(language, 0) + 1
    )


for language, count in language_counts.items():

    print(
        f"{language}: {count}"
    )


# ============================================================
# SAMPLE RECORD
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE METADATA")
print("=" * 70)

if metadata:

    sample = metadata[0]

    print(
        f"Chunk ID : {sample.get('chunk_id')}"
    )

    print(
        f"Title   : {sample.get('title')}"
    )

    print(
        f"Language: {sample.get('language')}"
    )

    print(
        f"URL     : {sample.get('url')}"
    )

    content = sample.get(
        "content",
        ""
    )

    print("\nContent preview:")
    print(content[:500])


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    f"Embeddings : {num_embeddings}"
)

print(
    f"Dimension  : {embedding_dimension}"
)

print(
    f"Metadata   : {len(metadata)}"
)

print(
    f"NaN        : {nan_count}"
)

print(
    f"Infinity   : {inf_count}"
)

print(
    f"Duplicate IDs : {duplicate_count}"
)

print("=" * 70)

if (
    num_embeddings == len(metadata)
    and nan_count == 0
    and inf_count == 0
    and duplicate_count == 0
    and empty_content == 0
):

    print("✓ KANNADA EMBEDDINGS PASSED VALIDATION")

else:

    print(
        "⚠ KANNADA EMBEDDINGS NEED FURTHER REVIEW"
    )

print("=" * 70)
