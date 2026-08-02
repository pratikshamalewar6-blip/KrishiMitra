# from datasets import load_dataset

# dataset = load_dataset(
#     "mohanty/PlantVillage",
#     "default"
# )

# print(dataset)

from datasets import load_dataset

dataset = load_dataset("mohanty/PlantVillage", "default")

print(dataset["train"][0])