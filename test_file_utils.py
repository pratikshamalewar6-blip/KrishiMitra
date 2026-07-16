from common.file_utils import FileUtils

print()

print("=" * 60)
print("Testing FileUtils")
print("=" * 60)

FileUtils.ensure_directory("outputs/test")

print(FileUtils.directory_exists("outputs"))

print(FileUtils.file_exists("requirements.txt"))

print(FileUtils.count_files("."))

images = FileUtils.list_images("datasets")

print(f"Images Found : {len(images)}")

print("=" * 60)
print("FileUtils Test Completed Successfully")
print("=" * 60)