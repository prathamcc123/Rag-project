import json
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import numpy as np
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

from backend.config import (
    MODEL_PATH,
    INDEX_PATH,
    METADATA_PATH,
    TOP_K,
    SEARCH_K,
)


# ============================================================
# Kannada Vector Database Configuration
# ============================================================

KANNADA_VECTOR_DB_PATH = (
    Path(INDEX_PATH).parent / "kannada"
)

KANNADA_INDEX_PATH = (
    KANNADA_VECTOR_DB_PATH / "faiss.index"
)

KANNADA_METADATA_PATH = (
    KANNADA_VECTOR_DB_PATH / "metadata.json"
)


# ============================================================
# Retriever Class
# ============================================================

class Retriever:
    """
    Language-aware semantic retriever for AgriSahayak AI.

    Hindi / Existing Path:
        Query
          ↓
        Translate to Hindi
          ↓
        Embedding Model
          ↓
        Hindi FAISS
          ↓
        Hindi Metadata

    Kannada Path:
        Query
          ↓
        No Translation
          ↓
        Embedding Model
          ↓
        Kannada FAISS
          ↓
        Kannada Metadata
    """

    def __init__(self):

        # Embedding model
        self.model = None

        # Hindi resources
        self.index = None
        self.metadata = None

        # Kannada resources
        self.kannada_index = None
        self.kannada_metadata = None

        self._initialize()

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize(self):

        self._verify_files()

        # Load embedding model
        self.model = self._load_model()

        # Load Hindi resources
        self.index = self._load_index(
            INDEX_PATH,
            "Hindi FAISS Index",
        )

        self.metadata = self._load_metadata(
            METADATA_PATH,
            "Hindi Metadata",
        )

        # Load Kannada resources
        self.kannada_index = self._load_index(
            KANNADA_INDEX_PATH,
            "Kannada FAISS Index",
        )

        self.kannada_metadata = self._load_metadata(
            KANNADA_METADATA_PATH,
            "Kannada Metadata",
        )

        # Verify both databases
        self._verify_index(
            self.index,
            self.metadata,
            "Hindi",
        )

        self._verify_index(
            self.kannada_index,
            self.kannada_metadata,
            "Kannada",
        )

    # ========================================================
    # Verify Required Files
    # ========================================================

    def _verify_files(self):

        required_files = {
            "Embedding Model": MODEL_PATH,

            "Hindi FAISS Index": INDEX_PATH,
            "Hindi Metadata": METADATA_PATH,

            "Kannada FAISS Index": KANNADA_INDEX_PATH,
            "Kannada Metadata": KANNADA_METADATA_PATH,
        }

        missing = []

        for name, path in required_files.items():

            if not Path(path).exists():

                missing.append(
                    f"{name}: {path}"
                )

        if missing:

            raise FileNotFoundError(
                "Missing required files:\n"
                + "\n".join(missing)
            )

    # ========================================================
    # Load Embedding Model
    # ========================================================

    def _load_model(self):

        print("Loading embedding model...")

        model = SentenceTransformer(
            str(MODEL_PATH)
        )

        print("✓ Embedding model loaded")

        print(
            "Embedding dimension:",
            model.get_sentence_embedding_dimension()
        )

        return model

    # ========================================================
    # Load FAISS Index
    # ========================================================

    def _load_index(
        self,
        path,
        name: str,
    ):

        print(
            f"Loading {name}..."
        )

        index = faiss.read_index(
            str(path)
        )

        print(
            f"✓ {name} loaded"
        )

        print(
            f"  Vectors: {index.ntotal}"
        )

        print(
            f"  Dimension: {index.d}"
        )

        return index

    # ========================================================
    # Load Metadata
    # ========================================================

    def _load_metadata(
        self,
        path,
        name: str,
    ):

        print(
            f"Loading {name}..."
        )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(f)

        print(
            f"✓ {name} loaded"
        )

        print(
            f"  Records: {len(metadata)}"
        )

        return metadata

    # ========================================================
    # Verify FAISS and Metadata
    # ========================================================

    def _verify_index(
        self,
        index,
        metadata,
        language: str,
    ):

        if index.ntotal != len(metadata):

            raise RuntimeError(
                f"{language} FAISS index and "
                f"metadata count do not match.\n"
                f"FAISS vectors: {index.ntotal}\n"
                f"Metadata records: {len(metadata)}"
            )

        print(
            f"✓ {language} index verification passed"
        )

    # ========================================================
    # Kannada Language Detection
    # ========================================================

    @staticmethod
    def _is_kannada_query(
        query: str,
    ) -> bool:
        """
        Detect Kannada-script text.

        Kannada Unicode range:
            U+0C80 - U+0CFF

        This is intentionally used instead of translating
        every query to Hindi because Kannada queries must
        be searched directly against the Kannada index.
        """

        for char in query:

            if "\u0C80" <= char <= "\u0CFF":

                return True

        return False

    # ========================================================
    # Detect Query Language
    # ========================================================

    def _detect_language(
        self,
        query: str,
    ) -> str:
        """
        Detect the retrieval language.

        Returns:
            kn -> Kannada
            hi -> Existing Hindi retrieval path

        Existing Hindi behavior is preserved for all
        non-Kannada queries.
        """

        if self._is_kannada_query(query):

            return "kn"

        return "hi"

    # ========================================================
    # Generate Query Embedding
    # ========================================================

    def _embed_query(
        self,
        query: str,
        language: str,
    ) -> np.ndarray:

        query_for_embedding = query

        # ----------------------------------------------------
        # Kannada
        # ----------------------------------------------------

        if language == "kn":

            print(
                f"Original Query   : {query}"
            )

            print(
                "Detected Language: Kannada"
            )

            print(
                "Translation      : Skipped"
            )

        # ----------------------------------------------------
        # Hindi / Existing Path
        # ----------------------------------------------------

        else:

            print(
                f"Original Query   : {query}"
            )

            print(
                "Detected Language: Hindi/Existing Path"
            )

            try:

                translated_query = (
                    GoogleTranslator(
                        source="auto",
                        target="hi",
                    ).translate(query)
                )

                if (
                    translated_query
                    and translated_query.strip()
                ):

                    query_for_embedding = (
                        translated_query
                    )

                print(
                    f"Translated Query : "
                    f"{query_for_embedding}"
                )

            except Exception as e:

                print(
                    f"Translation Error: {e}"
                )

                print(
                    "Using original query "
                    "for embedding."
                )

        # ----------------------------------------------------
        # Generate Embedding
        # ----------------------------------------------------

        embedding = self.model.encode(
            query_for_embedding,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Make sure embedding is float32
        embedding = embedding.astype(
            np.float32
        )

        return np.expand_dims(
            embedding,
            axis=0,
        )

    # ========================================================
    # Search FAISS
    # ========================================================

    def _search(
        self,
        embedding: np.ndarray,
        language: str,
    ):

        if language == "kn":

            index = self.kannada_index

        else:

            index = self.index

        scores, indices = index.search(
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
        language: str,
    ) -> Tuple[List[Dict], str]:

        # Select correct metadata
        if language == "kn":

            metadata = self.kannada_metadata

        else:

            metadata = self.metadata

        retrieved_chunks = []

        context_parts = []

        # Avoid returning multiple chunks from the same
        # article when possible.
        seen_articles = set()

        rank = 1

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            # FAISS may return -1 if no result exists
            if idx == -1:

                continue

            # Safety check
            if idx >= len(metadata):

                continue

            chunk = metadata[idx]

            # ------------------------------------------------
            # Article ID
            # ------------------------------------------------

            if "chunk_id" in chunk:

                article_id = (
                    chunk["chunk_id"]
                    .split("_chunk_")[0]
                )

            elif "url" in chunk:

                article_id = chunk["url"]

            else:

                article_id = chunk.get(
                    "title",
                    f"doc_{idx}",
                )

            # ------------------------------------------------
            # Deduplicate Article
            # ------------------------------------------------

            if article_id in seen_articles:

                continue

            seen_articles.add(
                article_id
            )

            # ------------------------------------------------
            # Store Retrieved Chunk
            # ------------------------------------------------

            retrieved_chunks.append(
                {
                    "score": float(score),
                    **chunk,
                }
            )

            # ------------------------------------------------
            # Extract Metadata
            # ------------------------------------------------

            title = chunk.get(
                "title",
                "Untitled",
            )

            summary = chunk.get(
                "summary",
                "",
            )

            chunk_language = chunk.get(
                "language",
                language,
            )

            url = chunk.get(
                "url",
                "",
            )

            content = chunk.get(
                "content",
                "",
            )

            # ------------------------------------------------
            # Build RAG Context
            # ------------------------------------------------

            context_parts.append(
                f"""
================ Document {rank} ================

Title:
{title}

Summary:
{summary}

Language:
{chunk_language}

URL:
{url}

Content:
{content}
"""
            )

            rank += 1

            if len(
                retrieved_chunks
            ) >= TOP_K:

                break

        context = "\n\n".join(
            context_parts
        )

        return (
            retrieved_chunks,
            context,
        )

    # ========================================================
    # Main Retrieval Method
    # ========================================================

    def retrieve(
        self,
        query: str,
    ) -> Tuple[List[Dict], str]:
        """
        Retrieve relevant documents for a user query.

        Kannada queries use the Kannada vector database.

        All other queries preserve the existing Hindi
        retrieval behavior.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # ----------------------------------------------------
        # Detect language
        # ----------------------------------------------------

        language = self._detect_language(
            query
        )

        print(
            f"\nRetrieval Language: {language}"
        )

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        embedding = self._embed_query(
            query=query,
            language=language,
        )

        # ----------------------------------------------------
        # Search correct vector database
        # ----------------------------------------------------

        scores, indices = self._search(
            embedding=embedding,
            language=language,
        )

        # ----------------------------------------------------
        # Process results
        # ----------------------------------------------------

        retrieved_chunks, context = (
            self._process_results(
                scores=scores,
                indices=indices,
                language=language,
            )
        )

        return (
            retrieved_chunks,
            context,
        )


# ============================================================
# Retriever Singleton
# ============================================================

_retriever = None


def get_retriever() -> Retriever:

    global _retriever

    if _retriever is None:

        _retriever = Retriever()

    return _retriever


# ============================================================
# Public API
# ============================================================

def retrieve_query(
    query: str,
):
    """
    Public retrieval function used by RAGPipeline.
    """

    retriever = get_retriever()

    return retriever.retrieve(
        query
    )


# ============================================================
# Health Check
# ============================================================

def health_check() -> dict:
    """
    Return retriever health information.
    """

    retriever = get_retriever()

    return {
        "model_loaded":
            retriever.model is not None,

        "hindi_index_loaded":
            retriever.index is not None,

        "hindi_metadata_loaded":
            retriever.metadata is not None,

        "kannada_index_loaded":
            retriever.kannada_index is not None,

        "kannada_metadata_loaded":
            retriever.kannada_metadata is not None,

        "hindi_vectors":
            retriever.index.ntotal,

        "kannada_vectors":
            retriever.kannada_index.ntotal,

        "embedding_dimension":
            retriever.model
            .get_sentence_embedding_dimension(),
    }


# ============================================================
# Interactive Testing
# ============================================================

def main():

    print("=" * 60)
    print("AgriSahayak AI - Multilingual Retriever Test")
    print("=" * 60)

    print(
        f"\nHindi Index   : {INDEX_PATH}"
    )

    print(
        f"Kannada Index : {KANNADA_INDEX_PATH}"
    )

    print(
        f"Model         : {MODEL_PATH}"
    )

    while True:

        question = input(
            "\nAsk your agriculture question "
            "('exit' to quit): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nGoodbye!"
            )

            break

        try:

            results, context = (
                retrieve_query(
                    question
                )
            )

            print("\n")
            print("=" * 60)
            print("RETRIEVED SOURCES")
            print("=" * 60)

            if not results:

                print(
                    "\nNo relevant documents found."
                )

                continue

            for i, source in enumerate(
                results,
                start=1,
            ):

                print(
                    f"\n{i}. "
                    f"{source.get('title', 'Untitled')}"
                )

                print(
                    f"Language : "
                    f"{source.get('language', '-')}"
                )

                print(
                    f"URL      : "
                    f"{source.get('url', '-')}"
                )

                print(
                    f"Score    : "
                    f"{source.get('score', 0.0):.4f}"
                )

            print("\n")
            print("=" * 60)
            print("CONTEXT GENERATED")
            print("=" * 60)

            print(
                f"Context characters: "
                f"{len(context)}"
            )

        except Exception as e:

            print(
                f"\nError: {e}"
            )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()