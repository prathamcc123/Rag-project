"""
semantic_chunker.py

Version 2

Creates semantic chunks with metadata.

Author: Major Project
"""

import json
import re
from pathlib import Path

INPUT_FOLDER = Path("data/processed")
OUTPUT_FOLDER = Path("data/chunks")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Detect Language
# --------------------------------------------------

def detect_language(filename):

    filename = filename.lower()

    if "english" in filename:
        return "English"

    elif "kannada" in filename:
        return "Kannada"

    return "Hindi"


# --------------------------------------------------
# Detect Category
# --------------------------------------------------

def detect_category(text):

    text = text.lower()

    categories = {

        "Crop Cultivation": [
            "धान", "rice",
            "गेहूं", "wheat",
            "खेती", "crop",
            "maize", "मक्का",
            "cotton", "कपास"
        ],

        "Disease Management": [
            "रोग",
            "disease",
            "blast",
            "blight",
            "fungicide",
            "bacteria"
        ],

        "Pest Management": [
            "कीट",
            "pest",
            "worm",
            "larva",
            "insect"
        ],

        "Fertilizer": [
            "fertilizer",
            "urea",
            "dap",
            "npk",
            "खाद",
            "उर्वरक"
        ],

        "Government Scheme": [
            "योजना",
            "scheme",
            "subsidy",
            "सरकार",
            "pm-kisan"
        ]

    }

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword.lower() in text:
                return category

    return "General Agriculture"


# --------------------------------------------------
# Extract Keywords
# --------------------------------------------------

def extract_keywords(text):

    words = re.findall(r"\w+", text)

    keywords = []

    for word in words:

        if len(word) > 4:

            if word not in keywords:
                keywords.append(word)

    return keywords[:10]


# --------------------------------------------------
# Detect Article Title
# --------------------------------------------------

def detect_title(chunk):

    lines = chunk.split("\n")

    for line in lines:

        line = line.strip()

        if len(line) > 5:
            return line

    return "Untitled"


# --------------------------------------------------
# Split Into Semantic Chunks
# --------------------------------------------------

def split_into_chunks(text):

    paragraphs = []

    current = ""

    for line in text.splitlines():

        line = line.strip()

        if line == "":

            if current:

                paragraphs.append(current.strip())
                current = ""

            continue

        current += line + "\n"

    if current:
        paragraphs.append(current.strip())

    return paragraphs


# --------------------------------------------------
# Process One File
# --------------------------------------------------

def process_file(file_path):

    print(f"\nProcessing : {file_path.name}")

    language = detect_language(file_path.name)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_into_chunks(text)

    output = []

    for idx, chunk in enumerate(chunks, start=1):

        output.append({

            "chunk_id": f"{file_path.stem}_{idx:03}",

            "source": file_path.stem,

            "language": language,

            "page": -1,

            "article_title": detect_title(chunk),

            "category": detect_category(chunk),

            "keywords": extract_keywords(chunk),

            "content": chunk

        })

    output_file = OUTPUT_FOLDER / f"{file_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Created {len(output)} semantic chunks")



# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    files = sorted(INPUT_FOLDER.glob("*.txt"))

    files = [
        file
        for file in files
        if "copy" not in file.name.lower()
    ]

    print(f"\nFound {len(files)} processed files.")

    if len(files) == 0:

        print("No processed files found.")
        return

    for file in files:
        process_file(file)

    print("\n✅ Semantic Chunking V2 Completed Successfully.")


if __name__ == "__main__":
    main()