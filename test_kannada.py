import requests

url = "https://kn.vikaspedia.in/agriculture"

r = requests.get(url)

with open("page.html", "w", encoding="utf-8") as f:
    f.write(r.text)

print("Saved page.html")