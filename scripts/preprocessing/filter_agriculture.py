import json
from pathlib import Path

# ==========================================
# Paths
# ==========================================

INPUT_DIR = Path("data/cleaned/hindi")
OUTPUT_DIR = Path("data/agriculture_only/hindi")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.json"))

print("=" * 60)
print(f"Total Cleaned Articles : {len(files)}")
print("=" * 60)

# ==========================================
# Keywords
# ==========================================

KEEP_KEYWORDS = [
    "गेहूं", "धान", "चावल", "मक्का", "बाजरा", "ज्वार",
    "चना", "अरहर", "मूंग", "उड़द", "सोयाबीन",
    "सरसों", "कपास", "गन्ना",
    "आलू", "प्याज", "टमाटर", "बैंगन", "भिंडी",
    "फल", "सब्जी",
    "दलहन", "तिलहन",
    "फसल", "खेती", "कृषि",
    "बीज", "बुवाई", "कटाई",
    "मिट्टी", "भूमि",
    "उर्वरक", "खाद",
    "सिंचाई",
    "जैविक",
    "कीट",
    "रोग",
    "खरपतवार"
]

REMOVE_KEYWORDS = [
    "मत्स्य", "मछली", "मछलीपालन",
    "पशुपालन", "दुग्ध", "डेयरी",
    "गाय", "भैंस", "बकरी", "भेड़",
    "सूअर", "मुर्गी", "पोल्ट्री",
    "बीमा", "ऋण", "बैंक",
    "मंत्रालय", "संगठन",
    "डायरेक्टरी",
    "कृषि विज्ञान केंद्र",
    "केवीके",
    "संस्थान"
]

# ==========================================
# Counters
# ==========================================

kept = 0
removed = 0

# ==========================================
# Filtering
# ==========================================

for file in files:

    try:
        with open(file, "r", encoding="utf-8") as f:
            article = json.load(f)

        text = " ".join([
            article.get("title", ""),
            article.get("summary", ""),
            article.get("content", "")
        ]).lower()

        # Remove unwanted topics
        if any(word.lower() in text for word in REMOVE_KEYWORDS):
            removed += 1
            continue

        # Keep only agriculture topics
        if any(word.lower() in text for word in KEEP_KEYWORDS):

            output_file = OUTPUT_DIR / file.name

            with open(output_file, "w", encoding="utf-8") as out:
                json.dump(
                    article,
                    out,
                    ensure_ascii=False,
                    indent=4
                )

            kept += 1

            if kept % 50 == 0:
                print(f"{kept} agriculture articles saved...")

        else:
            removed += 1

    except Exception as e:
        removed += 1
        print(f"{file.name} -> {e}")

# ==========================================
# Summary
# ==========================================

print("\n" + "=" * 60)
print("FILTERING FINISHED")
print("=" * 60)
print("Agriculture Articles :", kept)
print("Removed Articles     :", removed)