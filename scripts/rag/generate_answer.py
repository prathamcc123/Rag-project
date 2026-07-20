import os

from dotenv import load_dotenv
from google import genai

# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# ============================================================
# Initialize Gemini Client
# ============================================================

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# ============================================================
# Banner
# ============================================================

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

print_banner("Agriculture RAG Answer Generator")

print("✓ Gemini Client Initialized")
print(f"✓ Model : {MODEL_NAME}")
# ============================================================
# Import Retrieval Pipeline
# ============================================================

from retrieve import main as retrieve_pipeline

print("✓ Retrieval Pipeline Loaded")
# ============================================================
# Retrieve Context
# ============================================================

print_banner("Retrieving Knowledge")

query, retrieved_chunks, context = retrieve_pipeline()

print("\n✓ Context Retrieved Successfully")

print(f"Retrieved Documents : {len(retrieved_chunks)}")
print(f"Context Size        : {len(context):,} characters")
# ============================================================
# Prompt Builder
# ============================================================

prompt = f"""
You are an expert agricultural assistant.

Answer ONLY using the supplied context.

If the answer is not present in the context,
reply exactly:

I could not find this information in the knowledge base.

Do not hallucinate.
Do not invent facts.

=========================
CONTEXT
=========================

{context}

=========================
QUESTION
=========================

{query}

=========================
ANSWER
=========================
"""
# ============================================================
# Generate Answer
# ============================================================

print_banner("Generating Answer")

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt,
)

print("\n✓ Answer Generated\n")

print("=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(response.text)