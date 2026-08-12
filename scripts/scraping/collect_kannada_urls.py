# scripts/scraping/collect_kannada_urls.py

import requests
import json
import re
from collections import deque
from pathlib import Path

BASE = "https://kn.vikaspedia.in/viewcontent"
ROOT = f"{BASE}/agriculture"

visited = set()
article_urls = set()

queue = deque([ROOT])

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

while queue:
    url = queue.popleft()

    if url in visited:
        continue

    visited.add(url)

    print(f"Visiting: {url}")

    try:
        html = session.get(url, timeout=30).text

        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html,
            re.S
        )

        if not match:
            print("No NEXT_DATA")
            continue

        data = json.loads(match.group(1))

        page_data = (
            data.get("props", {})
                .get("pageProps", {})
                .get("ssrPageData", {})
        )

        items = page_data.get("contentList", [])

        print(f"Found {len(items)} items")

        for item in items:

            context_path = item.get("context_path")

            if not context_path:
                continue

            full_url = BASE + context_path

            item_type = item.get("context_type", "")

            if item_type == "folder":
                queue.append(full_url)

            elif item_type == "document":
                article_urls.add(full_url)

    except Exception as e:
        print("ERROR:", e)

# Save URLs
output = Path("data/urls")
output.mkdir(parents=True, exist_ok=True)

outfile = output / "kannada_urls.txt"

with open(outfile, "w", encoding="utf-8") as f:
    for url in sorted(article_urls):
        f.write(url + "\n")

print("\n====================")
print("CRAWL COMPLETE")
print("====================")
print("Folders Visited:", len(visited))
print("Article URLs:", len(article_urls))
print("Saved To:", outfile)