"""
KrishiMitra - Real Farm Crop Disease Image Downloader

Downloads 10+ real farm images for each of the 14 supported crop species
into D:\\KrishiMitra2\\ai_models\\disease_detection\\datasets\\raw\\google_images\\
"""

import os
import sys
from pathlib import Path

# Install icrawler if missing
try:
    from icrawler.builtin import BingImageCrawler
except ImportError:
    print("Installing icrawler package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "icrawler"])
    from icrawler.builtin import BingImageCrawler

# 14 Crop Species with specific search queries for real farm disease photos
CROPS_QUERIES = {
    "apple": "Apple leaf disease scab black rot photo farm",
    "blueberry": "Blueberry leaf photo farm field",
    "cherry": "Cherry leaf powdery mildew photo farm",
    "corn": "Corn leaf blight rust disease photo farm",
    "grape": "Grape leaf black rot measles disease photo farm",
    "orange": "Citrus greening orange leaf disease photo farm",
    "peach": "Peach leaf bacterial spot disease photo farm",
    "pepper_bell": "Bell pepper leaf bacterial spot disease photo farm",
    "potato": "Potato leaf late blight early blight photo farm",
    "raspberry": "Raspberry leaf disease photo farm",
    "soybean": "Soybean leaf disease photo farm",
    "squash": "Squash leaf powdery mildew disease photo farm",
    "strawberry": "Strawberry leaf scorch disease photo farm",
    "tomato": "Tomato leaf late blight early blight disease photo farm"
}

# Explicit Target Directory: D:\KrishiMitra2\ai_models\disease_detection\datasets\raw\google_images
base_dir = Path("D:/KrishiMitra2/ai_models/disease_detection/datasets/raw/google_images")
base_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"Starting Real Farm Image Download into:\n{base_dir.resolve()}")
print("=" * 60)

summary_stats = {}

for crop_name, query in CROPS_QUERIES.items():
    crop_dir = base_dir / crop_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Downloading 12 images for crop '{crop_name}'...")
    print(f"   Query: '{query}'")
    
    try:
        crawler = BingImageCrawler(
            storage={"root_dir": str(crop_dir)},
            log_level=30 # Suppress verbose logs
        )
        crawler.crawl(keyword=query, max_num=12)
        
        # Count downloaded images
        downloaded = [f for f in crop_dir.glob("*") if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
        summary_stats[crop_name] = len(downloaded)
        print(f"   ✅ Saved {len(downloaded)} images in: {crop_dir}")
    except Exception as e:
        print(f"   ❌ Error downloading for {crop_name}: {e}")
        summary_stats[crop_name] = len(list(crop_dir.glob("*")))

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY ACROSS 14 CROPS:")
print("=" * 60)
total_imgs = 0
for crop, count in summary_stats.items():
    print(f"  • {crop:<15}: {count:>2} images")
    total_imgs += count
print("-" * 60)
print(f"TOTAL IMAGES DOWNLOADED: {total_imgs} images across 14 crop folders.")
print(f"Target Directory: {base_dir.resolve()}")
print("=" * 60)
