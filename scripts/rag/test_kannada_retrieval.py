import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/final_model")

INDEX_PATH = Path("vector_db/kannada/faiss.index")

METADATA_PATH = Path("vector_db/kannada/metadata.json")

TOP_K = 5


# ============================================================
# BANNER
# ============================================================

print("=" * 70)
print("KANNADA FAISS RETRIEVAL TEST")
print("=" * 70)

print(f"Model    : {MODEL_PATH}")
print(f"FAISS    : {INDEX_PATH}")
print(f"Metadata : {METADATA_PATH}")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

for name, path in {
    "Model": MODEL_PATH,
    "FAISS index": INDEX_PATH,
    "Metadata": METADATA_PATH,
}.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SentenceTransformer model...")

model = SentenceTransformer(
    str(MODEL_PATH)
)

print("✓ Model loaded")

print(
    f"Embedding dimension : "
    f"{model.get_sentence_embedding_dimension()}"
)


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading Kannada FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

print("✓ FAISS index loaded")

print(
    f"Vectors in index : {index.ntotal}"
)


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading Kannada metadata...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print("✓ Metadata loaded")

print(
    f"Metadata records : {len(metadata)}"
)


# ============================================================
# VERIFY
# ============================================================

if index.ntotal != len(metadata):

    raise RuntimeError(
        "FAISS vector count does not match metadata count."
    )

print("✓ FAISS / metadata count matches")


# ============================================================
# TEST QUERIES
# ============================================================

queries = [
    "ಭತ್ತದ ಬೆಳೆಯನ್ನು ಹೇಗೆ ಬೆಳೆಯಬೇಕು?",
    "ರೈತರಿಗೆ ಕೃಷಿ ಸಾಲದ ಮಾಹಿತಿ",
    "ಕೀಟಗಳಿಂದ ಬೆಳೆಗಳನ್ನು ಹೇಗೆ ರಕ್ಷಿಸಬಹುದು?",
]


# ============================================================
# RUN RETRIEVAL
# ============================================================

for query_number, query in enumerate(queries, start=1):

    print("\n")
    print("=" * 70)
    print(f"QUERY {query_number}")
    print("=" * 70)

    print(f"\nKannada Query:")
    print(query)

    # --------------------------------------------------------
    # Generate Kannada query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.expand_dims(
        query_embedding,
        axis=0
    )

    print(
        f"\nQuery embedding shape : "
        f"{query_embedding.shape}"
    )

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    print("\nTop Results:")
    print("-" * 70)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        item = metadata[idx]

        print(f"\n[{rank}] Similarity Score : {score:.4f}")

        print(
            f"Title : "
            f"{item.get('title', 'Untitled')}"
        )

        print(
            f"Language : "
            f"{item.get('language', 'N/A')}"
        )

        print(
            f"Chunk ID : "
            f"{item.get('chunk_id', 'N/A')}"
        )

        print("\nContent Preview:")

        content = item.get(
            "content",
            ""
        )

        print(
            content[:700]
            .replace("\n", " ")
        )

        print("-" * 70)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("KANNADA RETRIEVAL TEST COMPLETE")
print("=" * 70)

print("✓ Query embedding generated")
print("✓ FAISS search completed")
print("✓ Kannada metadata retrieved")
print("✓ Retrieval pipeline test completed")

print("=" * 70)
