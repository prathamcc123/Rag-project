import json
from pathlib import Path

# ==========================================
# Paths
# ==========================================

INPUT_DIR = Path("data/agriculture_only/hindi")
OUTPUT_DIR = Path("data/chunks/hindi")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.json"))

print("=" * 60)
print(f"Total Articles : {len(files)}")
print("=" * 60)

# ==========================================
# Chunk Parameters
# ==========================================

CHUNK_SIZE = 500      # words
OVERLAP = 50          # words

total_chunks = 0

# ==========================================
# Process Articles
# ==========================================

for file in files:

    try:

        with open(file, "r", encoding="utf-8") as f:
            article = json.load(f)

        title = article.get("title", "")
        summary = article.get("summary", "")
        url = article.get("url", "")
        language = article.get("language", "Hindi")
        content = article.get("content", "")

        words = content.split()

        start = 0
        chunk_number = 1

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunk = {
                "chunk_id": f"{file.stem}_chunk_{chunk_number}",
                "title": title,
                "summary": summary,
                "url": url,
                "language": language,
                "content": chunk_text
            }

            output_file = OUTPUT_DIR / f"{file.stem}_chunk_{chunk_number}.json"

            with open(output_file, "w", encoding="utf-8") as out:
                json.dump(
                    chunk,
                    out,
                    ensure_ascii=False,
                    indent=4
                )

            total_chunks += 1
            chunk_number += 1

            start += (CHUNK_SIZE - OVERLAP)

        if total_chunks % 500 == 0:
            print(f"{total_chunks} chunks created...")

    except Exception as e:
        print(file.name, e)

print("\n" + "=" * 60)
print("CHUNKING FINISHED")
print("=" * 60)
print(f"Total Chunks : {total_chunks}")