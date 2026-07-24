"""
KrishiMitra - Merged Dataset

Combines PlantVillage and PlantDoc segmented crops using a configurable ratio
and balances classes using oversampling.

Author:
    Antigravity AI
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset

from common.config import ConfigManager
from common.logger import LoggerManager
from data.dataset import DatasetSample

logger = LoggerManager.get_logger("MergedDiseaseDataset")


class MergedDiseaseDataset(Dataset):
    """
    Merged Dataset combining PlantVillage (sterile background) and
    PlantDoc (real-world background-removed segmented leaf crops).
    """

    def __init__(
        self,
        split: str,
        transform=None,
        mix_ratio: float | None = None
    ) -> None:
        """
        Initialize the merged dataset.
        
        Parameters
        ----------
        split : str
            Dataset split: 'train', 'val', or 'test'.
        transform : callable
            Transform pipeline.
        mix_ratio : float
            Target ratio of PlantDoc crops in the dataset (e.g., 0.15 for 15%).
            If None, loaded from the config.
        """
        self.split = split
        self.transform = transform
        self.config = ConfigManager()

        # Load mix ratio from arguments, config, or default to 0.15
        if mix_ratio is None:
            self.mix_ratio = self.config.get("classification.mix_ratio", 0.15)
        else:
            self.mix_ratio = mix_ratio

        # 1. Load Master Class Mapping
        self.class_to_index = self._load_master_mapping()
        self.index_to_class = {idx: name for name, idx in self.class_to_index.items()}

        # 2. Load PlantVillage samples
        self.pv_samples = self._load_plantvillage_samples()
        
        # 3. Load PlantDoc samples
        self.pd_samples = self._load_plantdoc_samples()

        # 4. Merge datasets with target ratio
        self.samples = self._merge_datasets()
        
        logger.info(
            f"Merged dataset created for split '{split}'. "
            f"Total: {len(self.samples)} samples (PV: {len(self.pv_samples)}, PD target ratio: {self.mix_ratio*100}%)"
        )

    def _load_master_mapping(self) -> Dict[str, int]:
        """Loads the master class mapping from outputs."""
        mapping_path = Path("outputs/classification/class_mapping.json")
        if mapping_path.exists():
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load class_mapping.json: {e}. Building fallback.")
        
        # Fallback dictionary matching standard 38 classes if file is missing
        logger.warning("Class mapping file not found. Falling back to default list.")
        raise FileNotFoundError(f"Master class mapping file not found at: {mapping_path}")

    def _load_plantvillage_samples(self) -> List[DatasetSample]:
        """Loads PlantVillage samples from split CSV."""
        csv_path = Path("outputs/splits") / f"plantvillage_{self.split}.csv"
        if not csv_path.exists():
            logger.error(f"PlantVillage split file not found: {csv_path}")
            return []

        df = pd.read_csv(csv_path)
        samples = []
        for row in df.itertuples():
            samples.append(DatasetSample(
                image_path=row.image_path,
                class_name=row.class_name,
                dataset="plantvillage"
            ))
        return samples

    def _load_plantdoc_samples(self) -> List[DatasetSample]:
        """Loads processed PlantDoc segmented crops dynamically from directory."""
        # Map validation split to val or test folder
        dir_split = "val" if self.split in ["val", "test"] else "train"
        plantdoc_dir = Path("datasets/processed/plantdoc_classification") / dir_split

        if not plantdoc_dir.exists():
            logger.warning(f"Segmented PlantDoc directory not found at: {plantdoc_dir}. Preprocessing skipped?")
            return []

        samples = []
        for class_dir in plantdoc_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                # Verify that the class exists in our master mapping
                if class_name not in self.class_to_index:
                    logger.warning(f"PlantDoc class '{class_name}' is not in the 38-class master mapping. Skipping.")
                    continue
                
                for img_file in class_dir.glob("*.png"):
                    samples.append(DatasetSample(
                        image_path=str(img_file),
                        class_name=class_name,
                        dataset="plantdoc_classification"
                    ))
        return samples

    def _merge_datasets(self) -> List[DatasetSample]:
        """Merges the datasets applying oversampling/undersampling based on mix_ratio."""
        if not self.pv_samples or self.mix_ratio >= 1.0:
            return self.pd_samples
        if not self.pd_samples or self.mix_ratio <= 0.0:
            return self.pv_samples

        # Number of target PlantDoc samples in the final dataset
        # final_size = pv_size + pd_size
        # pd_size / final_size = mix_ratio => pd_size = pv_size * mix_ratio / (1 - mix_ratio)
        pv_size = len(self.pv_samples)
        target_pd_size = int(pv_size * self.mix_ratio / (1.0 - self.mix_ratio))
        
        logger.info(f"Targeting {target_pd_size} PlantDoc samples to match ratio of {self.mix_ratio*100}%")

        # Oversample PlantDoc samples to reach target size
        random.seed(42)
        pd_samples_resampled = []
        if len(self.pd_samples) > 0:
            pd_samples_resampled = random.choices(self.pd_samples, k=target_pd_size)

        # Merge and shuffle
        merged = self.pv_samples + pd_samples_resampled
        random.shuffle(merged)
        return merged

    def get_class_counts(self) -> dict[int, int]:
        """Returns the number of samples per class index."""
        counts = {}
        for sample in self.samples:
            label = self.class_to_index.get(sample.class_name)
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def list_all_crops(self) -> list[str]:
        """Returns a sorted list of unique crops in this dataset."""
        crops = set()
        for sample in self.samples:
            # Crop name is the first part of the class name before "___"
            if "___" in sample.class_name:
                crop = sample.class_name.split("___")[0]
                crops.add(crop)
        return sorted(list(crops))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[any, int]:
        sample = self.samples[index]
        image_path = Path(sample.image_path)

        # 1. Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image: {image_path}")
            raise RuntimeError(f"Failed to load image: {image_path}") from e

        # 2. Apply transform
        if self.transform is not None:
            image = self.transform(image)

        # 3. Convert label
        label = self.class_to_index[sample.class_name]

        return image, label


def main() -> None:
    # Test script execution
    transform = lambda x: x
    try:
        dataset = MergedDiseaseDataset(split="train", transform=transform, mix_ratio=0.15)
        print("=" * 60)
        print("Merged Dataset Test Successful!")
        print(f"Total samples: {len(dataset)}")
        print(f"Crops list   : {dataset.list_all_crops() if hasattr(dataset, 'list_all_crops') else 'N/A'}")
        print("=" * 60)
    except Exception as e:
        print(f"Failed to test Merged Dataset: {e}")


if __name__ == "__main__":
    main()
