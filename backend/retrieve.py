import json
from pathlib import Path
from typing import List, Tuple, Dict
from deep_translator import GoogleTranslator
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import (
    MODEL_PATH,
    INDEX_PATH,
    METADATA_PATH,
    TOP_K,
    SEARCH_K,
)


# ============================================================
# Retriever Class
# ============================================================

class Retriever:
    """
    Semantic Retriever using
    - Fine-tuned SentenceTransformer
    - FAISS Index
    - Metadata JSON

    This class loads everything only once when the
    Streamlit app starts.
    """

    def __init__(self):

        self.model = None
        self.index = None
        self.metadata = None

        self._initialize()

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize(self):

        self._verify_files()

        self.model = self._load_model()

        self.index = self._load_index()

        self.metadata = self._load_metadata()

        self._verify_index()

    # ========================================================
    # File Verification
    # ========================================================

    def _verify_files(self):

        required_files = {
            "Model": MODEL_PATH,
            "FAISS Index": INDEX_PATH,
            "Metadata": METADATA_PATH,
        }

        missing = []

        for name, path in required_files.items():

            if not Path(path).exists():
                missing.append(f"{name}: {path}")

        if missing:

            raise FileNotFoundError(
                "\n".join(missing)
            )

    # ========================================================
    # Load Sentence Transformer
    # ========================================================

    def _load_model(self):

        print("Loading embedding model...")

        model = SentenceTransformer(str(MODEL_PATH))

        print("✓ Embedding model loaded")

        return model

    # ========================================================
    # Load FAISS
    # ========================================================

    def _load_index(self):

        print("Loading FAISS index...")

        index = faiss.read_index(str(INDEX_PATH))

        print(f"✓ Loaded {index.ntotal} vectors")

        return index

    # ========================================================
    # Load Metadata
    # ========================================================

    def _load_metadata(self):

        print("Loading metadata...")

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"✓ Loaded {len(metadata)} metadata entries")

        return metadata

    # ========================================================
    # Verify Index
    # ========================================================

    def _verify_index(self):

        if self.index.ntotal != len(self.metadata):

            raise RuntimeError(
                "Metadata count does not match FAISS index."
            )

    # ========================================================
    # Query Embedding
    # ========================================================
       # ========================================================
    # Query Embedding
    # ========================================================

    def _embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        translated_query = query

        if GoogleTranslator is not None:

            try:

                translated = GoogleTranslator(
                    source="auto",
                    target="hi"
                ).translate(query)

                if translated and translated.strip():
                    translated_query = translated

                print(f"Original Query   : {query}")
                print(f"Translated Query : {translated_query}")

            except Exception as e:

                print(f"Translation Error: {e}")

        else:

            print(
                "Translation skipped: deep_translator package not installed."
            )

        embedding = self.model.encode(
            translated_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.expand_dims(
            embedding,
            axis=0,
        )
    
    # ========================================================
    # Search
    # ========================================================

    def _search(
        self,
        embedding: np.ndarray,
    ):

        scores, indices = self.index.search(
            embedding,
            SEARCH_K,
        )

        return scores, indices
        # ========================================================
    # Process Retrieved Results
    # ========================================================

    def _process_results(
        self,
        scores,
        indices,
    ) -> Tuple[List[Dict], str]:
        """
        Process FAISS results, remove duplicate articles,
        and build the final context.
        """

        retrieved_chunks = []
        context_parts = []
        seen_articles = set()

        rank = 1

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            chunk = self.metadata[idx]

            # --------------------------------------------------
            # Detect duplicate articles
            # --------------------------------------------------

            if "chunk_id" in chunk:

                article_id = chunk["chunk_id"].split("_chunk_")[0]

            elif "url" in chunk:

                article_id = chunk["url"]

            else:

                article_id = chunk.get(
                    "title",
                    f"doc_{idx}",
                )

            if article_id in seen_articles:
                continue

            seen_articles.add(article_id)

            retrieved_chunks.append(
                {
                    "score": float(score),
                    **chunk,
                }
            )

            # --------------------------------------------------
            # Build Context
            # --------------------------------------------------

            title = chunk.get("title", "Untitled")
            summary = chunk.get("summary", "")
            language = chunk.get("language", "")
            url = chunk.get("url", "")
            content = chunk.get("content", "")

            context_parts.append(
                f"""
================ Document {rank} ================

Title:
{title}

Summary:
{summary}

Language:
{language}

URL:
{url}

Content:
{content}
"""
            )

            rank += 1

            if len(retrieved_chunks) >= TOP_K:
                break

        context = "\n\n".join(context_parts)

        return retrieved_chunks, context

    # ========================================================
    # Public Retrieval Function
    # ========================================================

    def retrieve(
        self,
        query: str,
    ) -> Tuple[List[Dict], str]:
        """
        Retrieve relevant chunks for a user query.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self._embed_query(query)

        scores, indices = self._search(embedding)

        retrieved_chunks, context = self._process_results(
            scores,
            indices,
        )

        return retrieved_chunks, context
    # ========================================================
# Retriever Singleton
# ========================================================

_retriever = None


def get_retriever() -> Retriever:
    """
    Returns a singleton Retriever instance.

    The model, FAISS index and metadata are loaded only once,
    which avoids repeated initialization in Streamlit.
    """

    global _retriever

    if _retriever is None:
        _retriever = Retriever()

    return _retriever


# ========================================================
# Backward Compatible Function
# ========================================================

def retrieve_query(
    query: str,
):
    """
    Backward compatible wrapper.

    Returns
    -------
    retrieved_chunks : List[Dict]
    context : str
    """

    retriever = get_retriever()

    return retriever.retrieve(query)


# ========================================================
# Utility Function
# ========================================================

def health_check() -> dict:
    """
    Returns information about the retrieval system.

    Useful for debugging and About page.
    """

    retriever = get_retriever()

    return {
        "model_loaded": retriever.model is not None,
        "index_loaded": retriever.index is not None,
        "metadata_loaded": retriever.metadata is not None,
        "embedding_dimension": retriever.model.get_sentence_embedding_dimension(),
        "total_vectors": retriever.index.ntotal,
        "metadata_entries": len(retriever.metadata),
        "top_k": TOP_K,
        "search_k": SEARCH_K,
    }


# ========================================================
# Interactive Testing
# ========================================================

def main():

    print("=" * 60)
    print("AgriSahayak AI Retrieval Test")
    print("=" * 60)

    while True:

        query = input("\nEnter your question (or 'exit'): ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:

            retrieved_chunks, context = retrieve_query(query)

            print("\nRetrieved Documents")
            print("-" * 60)

            for i, chunk in enumerate(retrieved_chunks, start=1):

                print(f"\n{i}. {chunk.get('title', 'Untitled')}")
                print(f"Language : {chunk.get('language', '-')}")
                print(f"Score    : {chunk.get('score', 0):.4f}")
                print(f"Summary  : {chunk.get('summary', '')}")

            print("\n")
            print("=" * 60)
            print("Context Length :", len(context), "characters")
            print("=" * 60)

        except Exception as e:

            print(f"\nError: {e}")


# ========================================================
# Entry Point
# ========================================================

if __name__ == "__main__":
    main()