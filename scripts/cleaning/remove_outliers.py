import json
import shutil
from pathlib import Path

SOURCE_DIR = Path("data/cleaned/kannada")
TARGET_DIR = Path("data/cleaned/kannada_final")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORDS = 10000

kept = 0
removed = 0

for file in SOURCE_DIR.glob("*.json"):

    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    wc = len(data["content"].split())

    if wc > MAX_WORDS:
        removed += 1
        print(f"REMOVED ({wc} words): {file.name}")
        continue

    shutil.copy(file, TARGET_DIR / file.name)
    kept += 1

print("\n" + "="*50)
print("OUTLIER REMOVAL COMPLETE")
print("="*50)
print("Kept:", kept)
print("Removed:", removed)
print("Output:", TARGET_DIR)