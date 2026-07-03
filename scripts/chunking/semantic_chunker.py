import json
import re
from pathlib import Path

INPUT_FOLDER = Path("data/processed")
OUTPUT_FOLDER = Path("data/chunks")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def detect_language(filename):
    name = filename.lower()

    if "english" in name:
        return "English"

    elif "kannada" in name:
        return "Kannada"

    return "Hindi"


def split_into_chunks(text):

    paragraphs = []

    current = ""

    for line in text.splitlines():

        line = line.strip()

        if not line:
            if current:
                paragraphs.append(current.strip())
                current = ""
            continue

        current += " " + line

    if current:
        paragraphs.append(current.strip())

    return paragraphs
def detect_category(text):
    text = text.lower()

    categories = {
        "Crop Cultivation": [
            "धान", "rice", "गेहूं", "wheat",
            "खेती", "crop"
        ],

        "Disease Management": [
            "रोग", "disease",
            "blast", "blight"
        ],

        "Pest Management": [
            "कीट", "pest",
            "worm", "insect"
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
            "सरकार"
        ]
    }

    for category, words in categories.items():
        for word in words:
            if word.lower() in text:
                return category

    return "General Agriculture"
def extract_keywords(text):

    words = text.split()

    keywords = []

    for word in words:

        word = word.strip(",.()[]")

        if len(word) > 4 and word not in keywords:
            keywords.append(word)

    return keywords[:10]
def detect_title(chunk):

    lines = chunk.split("\n")

    for line in lines:

        line = line.strip()

        if len(line) > 5:
            return line

    return "Untitled"


def process_file(file_path):

    print(f"Processing {file_path.name}")

    language = detect_language(file_path.name)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_into_chunks(text)

    output = []

    for idx, chunk in enumerate(chunks, start=1):

        output.append({
            "id": idx,
            "source": file_path.stem,
            "language": language,
            "content": chunk
        })

    output_file = OUTPUT_FOLDER / f"{file_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Created {len(output)} chunks")
    print()


def main():

    files = sorted(INPUT_FOLDER.glob("*.txt"))

    print(f"\nFound {len(files)} processed files.\n")

    for file in files:
        process_file(file)

    print("✅ Semantic Chunking Completed")


if __name__ == "__main__":
    main()