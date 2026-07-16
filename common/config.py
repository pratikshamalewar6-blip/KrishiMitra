"""
KrishiMitra - Configuration Manager

Loads YAML configuration files for the project.

Author: Pratiksha Malewar
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """
    Loads and provides access to YAML configuration files.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config: Dict[str, Any] = yaml.safe_load(file)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value using dot notation.

        Example:
            config.get("paths.datasets.raw")
        """

        keys = key.split(".")

        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the complete configuration dictionary.
        """
        return self.config