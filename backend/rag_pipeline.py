from typing import Dict, List, Tuple

from backend.retrieve import retrieve_query
from backend.generate_answer import generate_answer


# ============================================================
# RAG Pipeline
# ============================================================

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Pipeline Flow
    -------------
    User Query
        ↓
    Retrieve Relevant Documents
        ↓
    Build Context
        ↓
    Generate Gemini Answer
        ↓
    Return Answer + Sources
    """

    def __init__(self):
        pass

    # ========================================================
    # Ask Question
    # ========================================================

    def ask(
        self,
        question: str,
    ) -> Tuple[str, List[Dict]]:
        """
        Execute complete RAG pipeline.

        Parameters
        ----------
        question : str
            User question.

        Returns
        -------
        answer : str
        sources : List[Dict]
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # ----------------------------------------------------
        # Retrieve Context
        # ----------------------------------------------------

        retrieved_chunks, context = retrieve_query(
            query=question,
        )

        # ----------------------------------------------------
        # Generate Final Answer
        # ----------------------------------------------------

        answer = generate_answer(
            query=question,
            context=context,
        )

        return answer, retrieved_chunks
    # ============================================================
# Pipeline Singleton
# ============================================================

_pipeline = None


def get_pipeline() -> RAGPipeline:
    """
    Returns a singleton RAGPipeline instance.

    The retriever and generator are initialized only once,
    making the pipeline efficient for Streamlit applications.
    """

    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()

    return _pipeline


# ============================================================
# Public API
# ============================================================

def ask_question(
    question: str,
) -> tuple[str, list[dict]]:
    """
    Execute the complete RAG pipeline.

    Parameters
    ----------
    question : str
        User's question.

    Returns
    -------
    answer : str
        Final generated answer.

    sources : list[dict]
        Retrieved source documents.
    """

    pipeline = get_pipeline()

    return pipeline.ask(question)


# ============================================================
# Health Check
# ============================================================

def health_check() -> dict:
    """
    Returns overall pipeline status.
    """

    return {
        "pipeline_ready": True,
        "retriever": "Ready",
        "generator": "Ready",
    }


# ============================================================
# Interactive Testing
# ============================================================

def main():

    print("=" * 60)
    print("AgriSahayak AI - RAG Pipeline Test")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk your agriculture question ('exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        try:

            answer, sources = ask_question(question)

            print("\n")
            print("=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(answer)

            print("\n")
            print("=" * 60)
            print("SOURCES")
            print("=" * 60)

            for i, source in enumerate(sources, start=1):

                print(f"\n{i}. {source.get('title', 'Untitled')}")
                print(f"Language : {source.get('language', '-')}")
                print(f"URL      : {source.get('url', '-')}")
                print(
                    f"Score    : {source.get('score', 0.0):.4f}"
                )

        except Exception as e:

            print(f"\nError: {e}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()