



















"""
KrishiMitra - Configuration Manager

Loads and merges all YAML configuration files into a single
configuration namespace.

Author: Pratiksha Malewar
"""

from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    """
    Central configuration manager.

    Automatically loads every YAML file inside configs/
    and merges them into one configuration dictionary.
    """

    def __init__(self, config_dir: str = "configs") -> None:
        target_path = Path(config_dir)
        if not target_path.is_absolute():
            # Resolve relative to package directory (ai_models/disease_detection)
            pkg_root = Path(__file__).resolve().parent.parent
            target_path = pkg_root / config_dir

        self.config_dir = target_path

        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Configuration directory not found: {self.config_dir}"
            )

        self.config: dict[str, Any] = {}

        self._load_configs()

    # -----------------------------------------------------

    def _load_configs(self) -> None:
        """
        Load every YAML file and merge into one config.
        """

        for yaml_file in sorted(self.config_dir.glob("*.yaml")):

            with open(yaml_file, "r", encoding="utf-8") as file:

                data = yaml.safe_load(file) or {}

            self._deep_merge(self.config, data)

    # -----------------------------------------------------

    def _deep_merge(
        self,
        destination: dict,
        source: dict
    ) -> None:
        """
        Recursively merge dictionaries.
        """

        for key, value in source.items():

            if (
                key in destination
                and isinstance(destination[key], dict)
                and isinstance(value, dict)
            ):

                self._deep_merge(destination[key], value)

            else:

                destination[key] = value

    # -----------------------------------------------------

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Retrieve value using dot notation.

        Example

        config.get("paths.outputs")

        config.get("training.batch_size")
        """

        value: Any = self.config

        for part in key.split("."):

            if isinstance(value, dict):

                value = value.get(part)

            else:

                return default

            if value is None:

                return default

        return value

    # -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return complete configuration.
        """

        return self.config
# """
# KrishiMitra - Configuration Manager

# Loads all YAML configuration files and provides a single
# configuration interface for the entire project.

# Author: Pratiksha Malewar
# """

# from pathlib import Path
# from typing import Any, Dict

# import yaml


# class ConfigManager:
#     """
#     Central configuration manager.

#     Automatically loads all YAML configuration files
#     inside the configs directory.
#     """

#     def __init__(self, config_dir: str = "configs"):

#         self.config_dir = Path(config_dir)

#         if not self.config_dir.exists():
#             raise FileNotFoundError(
#                 f"Configuration directory not found: {self.config_dir}"
#             )

#         self.config: Dict[str, Any] = {}

#         self._load_all_configs()

#     # ---------------------------------------------------------

#     def _load_all_configs(self) -> None:
#         """
#         Load every YAML file inside configs/.
#         """

#         for yaml_file in self.config_dir.glob("*.yaml"):

#             with open(yaml_file, "r", encoding="utf-8") as file:

#                 data = yaml.safe_load(file) or {}

#             self.config[yaml_file.stem] = data

#     # ---------------------------------------------------------

#     def get(self, key: str, default: Any = None) -> Any:
#         """
#         Retrieve value using dot notation.

#         Example:

#         config.get("paths.paths.datasets.raw")

#         config.get("training.training.batch_size")

#         config.get("model.model.num_classes")
#         """

#         keys = key.split(".")

#         value: Any = self.config

#         for k in keys:

#             if isinstance(value, dict):

#                 value = value.get(k)

#             else:

#                 return default

#             if value is None:

#                 return default

#         return value

#     # ---------------------------------------------------------

#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Return complete configuration.
#         """

#         return self.config