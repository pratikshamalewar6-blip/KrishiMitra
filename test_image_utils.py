from pathlib import Path

from common.file_utils import FileUtils
from common.image_utils import ImageUtils


print("=" * 60)
print("Testing ImageUtils")
print("=" * 60)

# Find one sample image
images = FileUtils.list_images("datasets")

if not images:
    print("No images found!")
    exit()

sample = images[0]

print(f"Sample Image : {sample}")

print("\n1. Verify Image")
print(ImageUtils.verify_image(sample))

print("\n2. Load Image")

image = ImageUtils.load_image(sample)

print(image is not None)

print("\n3. Metadata")

metadata = ImageUtils.get_metadata(sample)

for key, value in metadata.items():
    print(f"{key:<15}: {value}")

print("\n4. SHA256")

print(ImageUtils.compute_sha256(sample)[:32], "...")

print("\n5. Convert RGB")

rgb = ImageUtils.convert_rgb(image)

print(rgb.mode)

print("\n6. Resize")

resized = ImageUtils.resize(rgb, (224, 224))

print(resized.size)

print("\n7. Save")

FileUtils.ensure_directory("outputs/test")

ImageUtils.save_image(
    resized,
    "outputs/test/resized_sample.jpg",
)

print("\nImage saved successfully!")

print("=" * 60)
print("ImageUtils Test Completed Successfully")
print("=" * 60)

# We'll make this more robust later by changing:

# image.format

# to something like:

# image.format if image.format else Path(path).suffix.replace(".", "").upper()