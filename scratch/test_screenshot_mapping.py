import json
from pathlib import Path

# Load master mapping and plantdoc mapping
master_mapping_file = Path("d:/KrishiMitra2/ai_models/disease_detection/outputs/classification/class_mapping.json")
plantdoc_mapping_file = Path("d:/KrishiMitra2/ai_models/disease_detection/outputs/classification/plantdoc_mapping.json")

with open(master_mapping_file, "r") as f:
    master_map = json.load(f)

with open(plantdoc_mapping_file, "r") as f:
    pd_map = json.load(f)

# Also test case & space variations
for k, v in list(pd_map.items()):
    pd_map[k.replace(" ", "_")] = v
    pd_map[k.replace("_", " ")] = v
    pd_map[k.lower()] = v
    pd_map[k.replace(" ", "_").lower()] = v
    pd_map[k.replace("_", " ").lower()] = v

# Folders from screenshot
folders_in_screenshot = [
    "Apple leaf",
    "Apple rust leaf",
    "Apple Scab Leaf",
    "Bell_pepper leaf",
    "Bell_pepper leaf spot",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "grape leaf",
    "grape leaf black rot",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf"
]

print(f"{'Folder Name in Screenshot':<35} | {'Mapped Master Class Name':<50} | {'Status'}")
print("-" * 95)

all_matched = True
for folder in folders_in_screenshot:
    mapped = pd_map.get(folder, pd_map.get(folder.replace("_", " "), pd_map.get(folder.replace(" ", "_"), None)))
    if mapped in master_map:
        print(f"{folder:<35} | {mapped:<50} | ✅ MATCHED (ID {master_map[mapped]})")
    else:
        print(f"{folder:<35} | {str(mapped):<50} | ❌ MISMATCH")
        all_matched = False

print("-" * 95)
if all_matched:
    print("ALL 26 FOLDERS IN SCREENSHOT ARE 100% MATCHED TO MASTER CLASSES!")
