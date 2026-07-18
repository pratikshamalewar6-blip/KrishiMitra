from common.config import ConfigManager

config = ConfigManager()

print("=" * 60)

print(config.get("project.name"))

print(config.get("project.version"))

print("=" * 60)

print(config.get("paths.outputs"))

print(config.get("paths.datasets.raw"))

print(config.get("paths.datasets.plantvillage"))

print("=" * 60)

print(config.get("training.batch_size"))

print(config.get("training.learning_rate"))

print("=" * 60)

# print(config.get("model.classification.architecture"))

# print(config.get("model.classification.input_size"))

config.get("classification.architecture")
config.get("classification.input_size")

print("=" * 60)

print(config.get("logging.level"))

print("=" * 60)

print(config.to_dict())