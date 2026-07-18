from scripts.dataset_split import DatasetSplitGenerator

print("=" * 60)
print("Testing Dataset Split Generator")
print("=" * 60)

DatasetSplitGenerator().run()

print("=" * 60)
print("Dataset Split Test Completed")
print("=" * 60)

# from scripts.dataset_split import DatasetSplitGenerator

# print("=" * 60)
# print("Testing Record Collection")
# print("=" * 60)

# generator = DatasetSplitGenerator()

# dataset_paths = generator.get_dataset_paths()

# for name, path in dataset_paths.items():

#     records = generator.collect_records(
#         name,
#         path,
#     )

#     print()
#     print(name)
#     print("Total Records:", len(records))

#     print(records[0])