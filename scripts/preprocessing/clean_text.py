import json
import re
from pathlib import Path

# ==========================================
# Paths
# ==========================================

INPUT_DIR = Path("data/scraped/hindi")
OUTPUT_DIR = Path("data/cleaned/hindi")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.json"))

print("=" * 60)
print(f"Total Files : {len(files)}")
print("=" * 60)

cleaned = 0
skipped = 0

# ==========================================
# Cleaning Loop
# ==========================================

for file in files:

    try:
        with open(file, "r", encoding="utf-8") as f:
            article = json.load(f)

        title = article.get("title", "").strip()
        summary = article.get("summary", "").strip()
        content = article.get("content", "").strip()

        # --------------------------------------
        # Basic Cleaning
        # --------------------------------------

        # Remove multiple spaces/tabs
        content = re.sub(r"[ \t]+", " ", content)

        # Remove multiple blank lines
        content = re.sub(r"\n\s*\n+", "\n\n", content)

        # Strip whitespace from every line
        content = "\n".join(
            line.strip()
            for line in content.splitlines()
        )

        # Remove empty lines
        content = "\n".join(
            line
            for line in content.splitlines()
            if line.strip()
        )

        # Remove invisible Unicode characters
        content = re.sub(
            r"[\u200b-\u200f\u202a-\u202e]",
            "",
            content
        )

        content = content.strip()

        # --------------------------------------
        # Skip Empty / Tiny Articles
        # --------------------------------------

        if len(content) < 100:
            skipped += 1
            print(f"{file.name} -> Skipped (Too Small)")
            continue

        # --------------------------------------
        # Save Cleaned JSON
        # --------------------------------------

        cleaned_article = {
            "title": title,
            "summary": summary,
            "language": article.get("language", "Hindi"),
            "url": article.get("url", ""),
            "content": content,
            "characters": len(content)
        }

        output_file = OUTPUT_DIR / file.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                cleaned_article,
                f,
                ensure_ascii=False,
                indent=4
            )

        cleaned += 1

        if cleaned % 50 == 0:
            print(f"{cleaned} files cleaned...")

    except Exception as e:
        skipped += 1
        print(f"{file.name} -> {e}")

# ==========================================
# Summary
# ==========================================

print("\n" + "=" * 60)
print("CLEANING FINISHED")
print("=" * 60)
print("Cleaned :", cleaned)
print("Skipped :", skipped)