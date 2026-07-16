from common.config import ConfigManager

config = ConfigManager("configs/paths.yaml")

print(config.get("project.name"))
print(config.get("paths.datasets.plantvillage"))