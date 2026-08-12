import requests
import json
from bs4 import BeautifulSoup

url = "https://kn.vikaspedia.in/viewcontent/agriculture"

html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")

data = json.loads(
    soup.find("script", id="__NEXT_DATA__").string
)

page_props = data["props"]["pageProps"]

items = page_props["ssrPageData"]["contentList"]

print("FOUND:", len(items))

for item in items[:5]:
    print(item["title"])
    print(item["context_path"])
    print()