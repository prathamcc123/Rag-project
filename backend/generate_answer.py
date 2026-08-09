import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

from backend.config import GEMINI_MODEL
from backend.prompt import build_prompt


# ============================================================
# Gemini Generator
# ============================================================

class GeminiGenerator:
    """
    Wrapper around Gemini API.

    Loads the Gemini client once and exposes a simple
    generate() method for answer generation.
    """

    def __init__(self):

        self.client = None

        self._initialize()

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize(self):

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:

            raise EnvironmentError(
                "GEMINI_API_KEY not found in .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

    # ========================================================
    # Generate Response
    # ========================================================

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not context.strip():

            return (
                "I could not find this information "
                "in the knowledge base."
            )

        prompt = build_prompt(
            query=query,
            context=context,
        )
                # ----------------------------------------------------
        # Build Prompt
        # ----------------------------------------------------

        prompt = build_prompt(
            query=query,
            context=context,
        )

        try:

            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if response is None:

                return (
                    "I could not generate a response "
                    "at this time."
                )

            answer = getattr(response, "text", None)

            if answer is None:

                return (
                    "I could not generate a response "
                    "at this time."
                )

            answer = answer.strip()

            if not answer:

                return (
                    "I could not generate a response "
                    "at this time."
                )

            return answer

        except Exception as e:

            raise RuntimeError(
                f"Gemini API Error: {e}"
            ) from e
        # ========================================================
# Singleton Generator
# ========================================================

_generator: Optional[GeminiGenerator] = None


def get_generator() -> GeminiGenerator:
    """
    Returns a singleton GeminiGenerator instance.

    The Gemini client is initialized only once,
    making it efficient for Streamlit applications.
    """

    global _generator

    if _generator is None:
        _generator = GeminiGenerator()

    return _generator


# ========================================================
# Public API
# ========================================================

def generate_answer(
    query: str,
    context: str,
) -> str:
    """
    Generate an answer using Gemini.

    Parameters
    ----------
    query : str
        User's question.
    context : str
        Retrieved RAG context.

    Returns
    -------
    str
        Final generated answer.
    """

    generator = get_generator()

    return generator.generate(
        query=query,
        context=context,
    )


# ========================================================
# Health Check
# ========================================================

def health_check() -> dict:
    """
    Returns Gemini service status.
    Useful for debugging and About page.
    """

    generator = get_generator()

    return {
        "client_initialized": generator.client is not None,
        "model": GEMINI_MODEL,
    }


# ========================================================
# Interactive Testing
# ========================================================

def main():

    print("=" * 60)
    print("AgriSahayak AI - Gemini Generator Test")
    print("=" * 60)

    query = input("\nEnter your question: ").strip()

    if not query:

        print("Question cannot be empty.")
        return

    context = input(
        "\nPaste retrieved context:\n\n"
    ).strip()

    if not context:

        print("Context cannot be empty.")
        return

    try:

        answer = generate_answer(
            query=query,
            context=context,
        )

        print("\n")
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

    except Exception as e:

        print(f"\nError: {e}")


# ========================================================
# Entry Point
# ========================================================

if __name__ == "__main__":
    main()