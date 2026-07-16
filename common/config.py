"""
KrishiMitra - Configuration Manager

Loads all YAML configuration files and provides a single
configuration interface for the entire project.

Author: Pratiksha Malewar
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """
    Central configuration manager.

    Automatically loads all YAML configuration files
    inside the configs directory.
    """

    def __init__(self, config_dir: str = "configs"):

        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Configuration directory not found: {self.config_dir}"
            )

        self.config: Dict[str, Any] = {}

        self._load_all_configs()

    # ---------------------------------------------------------

    def _load_all_configs(self) -> None:
        """
        Load every YAML file inside configs/.
        """

        for yaml_file in self.config_dir.glob("*.yaml"):

            with open(yaml_file, "r", encoding="utf-8") as file:

                data = yaml.safe_load(file) or {}

            self.config[yaml_file.stem] = data

    # ---------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value using dot notation.

        Example:

        config.get("paths.paths.datasets.raw")

        config.get("training.training.batch_size")

        config.get("model.model.num_classes")
        """

        keys = key.split(".")

        value: Any = self.config

        for k in keys:

            if isinstance(value, dict):

                value = value.get(k)

            else:

                return default

            if value is None:

                return default

        return value

    # ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return complete configuration.
        """

        return self.config