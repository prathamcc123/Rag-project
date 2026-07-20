import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/muril_agriculture"
INDEX_PATH = "vector_db/faiss.index"
METADATA_PATH = "vector_db/metadata.json"

TOP_K = 5
SEARCH_K = 15

# ============================================================
# Utility Functions
# ============================================================

def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def verify_required_files():
    """Verify that all required files exist."""

    required_files = {
        "Model": MODEL_PATH,
        "FAISS Index": INDEX_PATH,
        "Metadata": METADATA_PATH,
    }

    for name, path in required_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")


# ============================================================
# Load Model
# ============================================================

def load_model():
    """Load the fine-tuned sentence transformer."""

    print("\nLoading Fine-Tuned MuRIL model...\n")

    model = SentenceTransformer(MODEL_PATH)

    print("✓ Model Loaded")
    print(f"Embedding Dimension : {model.get_embedding_dimension()}")

    return model


# ============================================================
# Load FAISS Index
# ============================================================

def load_faiss_index():
    """Load FAISS vector index."""

    print("\nLoading FAISS index...\n")

    index = faiss.read_index(INDEX_PATH)

    print("✓ FAISS Index Loaded")
    print(f"Total Vectors : {index.ntotal}")

    return index


# ============================================================
# Load Metadata
# ============================================================

def load_metadata():
    """Load metadata."""

    print("\nLoading metadata...\n")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("✓ Metadata Loaded")
    print(f"Metadata Entries : {len(metadata)}")

    return metadata


# ============================================================
# Verify Index
# ============================================================

def verify_index(index, metadata):
    """Verify FAISS index and metadata size."""

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Metadata count ({len(metadata)}) "
            f"does not match FAISS index ({index.ntotal})"
        )

    print("\n✓ Verification Successful")


# ============================================================
# Initialize Retrieval Resources
# ============================================================

print_banner("RAG Retrieval System")

verify_required_files()

model = load_model()

index = load_faiss_index()

metadata = load_metadata()

verify_index(index, metadata)

print("\nSystem Ready for Retrieval.")

print_banner("Semantic Retrieval")


# ============================================================
# Generate Query Embedding
# ============================================================

def generate_query_embedding(model, query):
    """Generate normalized query embedding."""

    print("\nGenerating query embedding...\n")

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding = np.expand_dims(embedding, axis=0)

    print("✓ Query embedding generated")

    return embedding


# ============================================================
# Search Index
# ============================================================

def search_index(index, query_embedding):
    """Search FAISS index."""

    print("\nSearching FAISS index...\n")

    scores, indices = index.search(
        query_embedding,
        SEARCH_K,
    )

    print("✓ Retrieval completed")

    return scores, indices
# ============================================================
# Process Retrieved Results
# ============================================================

def process_results(
    scores,
    indices,
    metadata,
    verbose=True,
    save_context=True,
):
    """
    Process retrieved chunks.

    Parameters
    ----------
    scores : FAISS similarity scores
    indices : FAISS retrieved indices
    metadata : metadata list
    verbose : print retrieved chunks (default=True)
    save_context : save context.txt (default=True)

    Returns
    -------
    retrieved_chunks : list
    context : str
    """

    if verbose:
        print_banner(f"Top {TOP_K} Unique Retrieved Chunks")

    seen_articles = set()

    retrieved_chunks = []

    context_parts = []

    rank = 1

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = metadata[idx]

        article_id = chunk["chunk_id"].split("_chunk_")[0]

        # Skip duplicate articles
        if article_id in seen_articles:
            continue

        seen_articles.add(article_id)

        retrieved_chunks.append(chunk)

        # --------------------------------------------------------
        # Display Results
        # --------------------------------------------------------

        if verbose:

            print(f"\nResult {rank}")
            print("-" * 65)

            print(f"Similarity Score : {score:.4f}")
            print(f"Chunk ID         : {chunk['chunk_id']}")
            print(f"Title            : {chunk['title']}")
            print(f"Summary          : {chunk['summary']}")
            print(f"Language         : {chunk['language']}")
            print(f"URL              : {chunk['url']}")

            print("\nContent:")

            preview = chunk["content"][:800]

            print(preview)

            if len(chunk["content"]) > 800:
                print("...")

        # --------------------------------------------------------
        # Build Context
        # --------------------------------------------------------

        context_parts.append(
            f"""
================ Document {rank} ================

Title:
{chunk['title']}

Summary:
{chunk['summary']}

Language:
{chunk['language']}

URL:
{chunk['url']}

Content:
{chunk['content']}
"""
        )

        rank += 1

        if rank > TOP_K:
            break

    # --------------------------------------------------------
    # Merge Context
    # --------------------------------------------------------

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Save Context (Optional)
    # --------------------------------------------------------

    if save_context:

        with open("context.txt", "w", encoding="utf-8") as f:
            f.write(context)

        if verbose:
            print("\n✓ Context saved to context.txt")

    return retrieved_chunks, context
# ============================================================
# Retrieve Query (Reusable Function)
# ============================================================

def retrieve_query(
    query,
    verbose=True,
    save_context=True,
):
    """
    Retrieve relevant chunks for a given query.

    Parameters
    ----------
    query : str
        User question.
    verbose : bool
        Display retrieved chunks.
    save_context : bool
        Save context.txt.

    Returns
    -------
    retrieved_chunks : list
    context : str
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_embedding = generate_query_embedding(
        model,
        query,
    )

    scores, indices = search_index(
        index,
        query_embedding,
    )

    retrieved_chunks, context = process_results(
        scores,
        indices,
        metadata,
        verbose=verbose,
        save_context=save_context,
    )

    if verbose:

        print_banner("Retrieval Summary")

        print(
            f"Unique Documents Retrieved : "
            f"{len(retrieved_chunks)}"
        )

        print(
            f"Context Length             : "
            f"{len(context):,} characters"
        )

        print("\nRetrieval Completed Successfully")

    return retrieved_chunks, context


# ============================================================
# Main Function
# ============================================================

def main():
    """
    Interactive retrieval mode.
    """

    query = input("\nEnter your question: ").strip()

    retrieved_chunks, context = retrieve_query(
        query=query,
        verbose=True,
        save_context=True,
    )

    return query, retrieved_chunks, context


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()