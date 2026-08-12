"""
scrape_vikaspedia.py

Production-ready Vikaspedia scraper
Major Project (RAG)

Features:
- Resume support
- Skip existing files
- Retry (3 attempts)
- Random delay (2-5 sec)
- Failed URL logging
- Safe extraction from __NEXT_DATA__
"""

import json
import random
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# =====================================================
# PATHS
# =====================================================

URL_FILE = Path("data/urls/kannada_urls.txt")

OUTPUT_DIR = Path("data/scraped/kannada")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAILED_FILE = Path("data/urls/failed_urls.txt")

# =====================================================
# HEADERS
# =====================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

# =====================================================
# LOAD URLS
# =====================================================

with open(URL_FILE, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

# Remove duplicate URLs
urls = list(dict.fromkeys(urls))

TOTAL = len(urls)

print("=" * 70)
print(f"Total URLs : {TOTAL}")
print("=" * 70)

saved = 0
skipped = 0
failed = 0

# =====================================================
# SCRAPER
# =====================================================

for index, url in enumerate(urls, start=1):

    filename = OUTPUT_DIR / f"article_{index:04d}.json"

    # -------------------------------------------------
    # Resume support
    # -------------------------------------------------

    if filename.exists():
        skipped += 1
        print(f"[{index}/{TOTAL}] Skipped")
        continue

    success = False

    # -------------------------------------------------
    # Retry up to 3 times
    # -------------------------------------------------

    for attempt in range(3):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=60
            )

            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # -----------------------------------------
            # TITLE
            # -----------------------------------------

            title = ""

            if soup.title:
                title = soup.title.get_text(strip=True)

            # -----------------------------------------
            # SUMMARY
            # -----------------------------------------

            summary = ""

            meta = soup.find("meta", attrs={"name": "description"})

            if meta:
                summary = meta.get("content", "")

            # -----------------------------------------
            # CONTENT
            # -----------------------------------------

            content = ""

            next_data = soup.find("script", id="__NEXT_DATA__")

            if next_data and next_data.string:

                try:

                    data = json.loads(next_data.string)

                    page_props = (
                        data.get("props", {})
                            .get("pageProps", {})
                    )

                    ssr = page_props.get("ssrPageContent")

                    if ssr and ssr.get("content"):

                        html = ssr["content"]

                        article = BeautifulSoup(html, "lxml")

                        content = article.get_text(
                            separator="\n",
                            strip=True
                        )

                except Exception:
                    pass

            # -----------------------------------------
            # Fallback
            # -----------------------------------------

            if len(content.strip()) < 100:

                content = soup.get_text(
                    separator="\n",
                    strip=True
                )

            # -----------------------------------------
            # Save JSON
            # -----------------------------------------
            if len(content.strip()) < 300:

               print(f"[{index}/{TOTAL}] Content too small, skipped")

               continue

            article = {
                "title": title,
                "summary": summary,
                "language": "Kannada",
                "source": "Vikaspedia",
                "url": url,
                "content": content
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    article,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            saved += 1
            success = True

            print(f"[{index}/{TOTAL}] Saved")

            # -----------------------------------------
            # Random polite delay
            # -----------------------------------------

            sleep_time = random.uniform(2, 5)

            print(f"Sleeping {sleep_time:.1f} sec...")

            time.sleep(sleep_time)

            break

        except Exception as e:

            print(f"[{index}/{TOTAL}] Attempt {attempt+1}/3 failed")

            print(e)

            if attempt < 2:

                wait = random.randint(8, 15)

                print(f"Retrying in {wait} sec...")

                time.sleep(wait)

    # -------------------------------------------------
    # Permanent Failure
    # -------------------------------------------------

    if not success:

        failed += 1

        print(f"[{index}/{TOTAL}] FAILED")

        with open(
            FAILED_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(url + "\n")

        # Extra delay after failure

        time.sleep(20)

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 70)
print("SCRAPING FINISHED")
print("=" * 70)

print(f"Saved   : {saved}")
print(f"Skipped : {skipped}")
print(f"Failed  : {failed}")