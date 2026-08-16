import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/cleaned/kannada_final")
OUTPUT_DIR = Path("data/chunks/kannada")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHUNK SETTINGS
# ============================================================

TARGET_WORDS = 500
MAX_WORDS = 650
OVERLAP_WORDS = 80

MIN_CHUNK_WORDS = 50


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic text normalization while preserving Kannada Unicode.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Split Kannada/English mixed text into sentences.

    Handles:
    .  ।  !  ?  newline
    """

    # Normalize danda
    text = text.replace("॥", "।")

    # Sentence boundary:
    # Kannada danda
    # English punctuation
    # newline
    parts = re.split(
        r"(?<=[.!?।])\s+|\n+",
        text
    )

    sentences = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        sentences.append(part)

    return sentences


# ============================================================
# WORD COUNT
# ============================================================

def word_count(text):
    """
    Unicode-safe whitespace based word count.
    """

    return len(text.split())


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def create_chunks(text):
    """
    Create sentence-aware chunks.

    The algorithm:
    1. Split article into sentences.
    2. Add sentences until target size is reached.
    3. Never exceed MAX_WORDS unless one sentence itself is larger.
    4. Carry the last OVERLAP_WORDS into the next chunk.
    """

    sentences = split_sentences(text)

    chunks = []

    current_sentences = []
    current_words = 0

    for sentence in sentences:

        sentence_words = word_count(sentence)

        # ----------------------------------------------------
        # Very large individual sentence
        # ----------------------------------------------------

        if sentence_words > MAX_WORDS:

            # Save current chunk first
            if current_sentences:

                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []
                current_words = 0

            # Split oversized sentence by words
            words = sentence.split()

            start = 0

            while start < len(words):

                end = min(
                    start + MAX_WORDS,
                    len(words)
                )

                chunk = " ".join(words[start:end])

                chunks.append(chunk)

                start = end - OVERLAP_WORDS

                if start < 0:
                    start = 0

            continue

        # ----------------------------------------------------
        # Normal sentence
        # ----------------------------------------------------

        if (
            current_words + sentence_words
            <= TARGET_WORDS
        ):

            current_sentences.append(sentence)
            current_words += sentence_words

        else:

            # Save current chunk
            if current_sentences:

                chunks.append(
                    " ".join(current_sentences)
                )

            # ------------------------------------------------
            # Create overlap from previous sentences
            # ------------------------------------------------

            overlap = []
            overlap_count = 0

            for previous in reversed(current_sentences):

                previous_words = word_count(previous)

                if (
                    overlap_count
                    + previous_words
                    > OVERLAP_WORDS
                ):
                    break

                overlap.insert(0, previous)

                overlap_count += previous_words

            current_sentences = overlap + [sentence]

            current_words = (
                overlap_count
                + sentence_words
            )

    # --------------------------------------------------------
    # Final chunk
    # --------------------------------------------------------

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    # --------------------------------------------------------
    # Remove extremely tiny chunks
    # --------------------------------------------------------

    final_chunks = []

    for chunk in chunks:

        chunk = clean_text(chunk)

        if word_count(chunk) >= MIN_CHUNK_WORDS:

            final_chunks.append(chunk)

    return final_chunks


# ============================================================
# PROCESS ONE ARTICLE
# ============================================================

def process_article(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        article = json.load(f)

    content = clean_text(
        article.get("content", "")
    )

    if not content:

        return []

    chunks = create_chunks(content)

    results = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        results.append({

            "chunk_id": (
                f"{file_path.stem}_chunk_{index:04d}"
            ),

            "source": "Vikaspedia",

            "language": "kn",

            "title": article.get(
                "title",
                ""
            ),

            "summary": article.get(
                "summary",
                ""
            ),

            "url": article.get(
                "url",
                ""
            ),

            "chunk_index": index,

            "total_chunks": len(chunks),

            "content": chunk,

            "word_count": word_count(chunk)
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print("=" * 70)
    print("KANNADA SEMANTIC CHUNKING")
    print("=" * 70)

    print(f"Input articles : {len(files)}")
    print(f"Target words   : {TARGET_WORDS}")
    print(f"Maximum words  : {MAX_WORDS}")
    print(f"Overlap words  : {OVERLAP_WORDS}")
    print()

    total_articles = 0
    total_chunks = 0

    for index, file_path in enumerate(
        files,
        start=1
    ):

        try:

            chunks = process_article(
                file_path
            )

            if not chunks:

                print(
                    f"[{index}/{len(files)}] "
                    f"No content: {file_path.name}"
                )

                continue

            # ------------------------------------------------
            # Save one JSON file per article
            # ------------------------------------------------

            output_file = (
                OUTPUT_DIR
                / f"{file_path.stem}.json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    chunks,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            total_articles += 1
            total_chunks += len(chunks)

            print(
                f"[{index}/{len(files)}] "
                f"{len(chunks):3d} chunks | "
                f"{file_path.name[:50]}"
            )

        except Exception as e:

            print(
                f"[{index}/{len(files)}] ERROR | "
                f"{file_path.name}"
            )

            print(
                f"       {e}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    print(
        f"Articles processed : {total_articles}"
    )

    print(
        f"Total chunks      : {total_chunks}"
    )

    if total_articles:

        print(
            f"Average chunks/article : "
            f"{total_chunks / total_articles:.2f}"
        )

    print(
        f"Output            : {OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()