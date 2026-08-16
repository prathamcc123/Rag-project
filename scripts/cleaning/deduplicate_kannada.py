import json
import hashlib
from pathlib import Path

INPUT_DIR = Path("data/scraped/kannada")
OUTPUT_DIR = Path("data/cleaned/kannada")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

seen_hashes = set()

total = 0
unique = 0
duplicates = 0

for file in INPUT_DIR.glob("*.json"):

    total += 1

    try:
        with open(file, "r", encoding="utf-8") as f:
            article = json.load(f)

        content = article.get("content", "").strip()

        if not content:
            continue

        content_hash = hashlib.md5(
            content.encode("utf-8")
        ).hexdigest()

        if content_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(content_hash)

        output_file = OUTPUT_DIR / file.name

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                article,
                f,
                ensure_ascii=False,
                indent=4
            )

        unique += 1

    except Exception as e:
        print(f"Error in {file.name}: {e}")

print("\n" + "=" * 50)
print("DEDUPLICATION COMPLETE")
print("=" * 50)
print(f"Total Articles : {total}")
print(f"Unique Articles: {unique}")
print(f"Duplicates     : {duplicates}")
print(f"Output Folder  : {OUTPUT_DIR}")