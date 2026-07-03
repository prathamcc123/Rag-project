"""
clean_text.py

Purpose:
Clean extracted PDF text while preserving useful
Hindi and English agricultural content.
"""

import re
from pathlib import Path

INPUT_DIR = Path("data/extracted_text")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs
    text = text.replace("\t", " ")

    # Remove page markers created during extraction
    text = re.sub(r"===== PAGE \d+ =====", "", text)

    # Remove multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove trailing spaces
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_file(file_path: Path):

    print(f"Cleaning: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_text(text)

    output_path = OUTPUT_DIR / file_path.name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Saved -> {output_path.name}")


def main():

    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    print(f"\nFound {len(txt_files)} text files.\n")

    if not txt_files:
        print("No text files found.")
        return

    for file in txt_files:
        process_file(file)

    print("\n✅ Cleaning completed successfully.")


if __name__ == "__main__":
    main()