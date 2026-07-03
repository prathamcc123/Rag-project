import json

file_path = "data/english/crops/rice.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Crop Name:", data["crop_name"])
print("Scientific Name:", data["scientific_name"])
print("Growing Season:", data["growing_season"])
print("Major Diseases:", data["major_diseases"])