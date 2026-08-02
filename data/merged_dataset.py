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
        mapping_paths = [
            Path("outputs/classification/class_mapping.json"),
            Path("ai_models/disease_detection/outputs/classification/class_mapping.json")
        ]
        for mapping_path in mapping_paths:
            if mapping_path.exists():
                try:
                    with open(mapping_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load class_mapping.json from {mapping_path}: {e}")
        
        # Fallback dictionary matching standard 38 classes if file is missing
        logger.warning("Class mapping file not found in candidate paths.")
        raise FileNotFoundError("Master class mapping file (class_mapping.json) not found.")

    def _load_plantdoc_mapping(self) -> Dict[str, str]:
        """Loads PlantDoc folder to Master Class Name mapping."""
        mapping_paths = [
            Path("outputs/classification/plantdoc_mapping.json"),
            Path("ai_models/disease_detection/outputs/classification/plantdoc_mapping.json")
        ]
        plantdoc_map: Dict[str, str] = {}
        for mapping_path in mapping_paths:
            if mapping_path.exists():
                try:
                    with open(mapping_path, "r", encoding="utf-8") as f:
                        raw_map = json.load(f)
                    for k, v in raw_map.items():
                        plantdoc_map[k] = v
                        plantdoc_map[k.replace(" ", "_")] = v
                        plantdoc_map[k.replace("_", " ")] = v
                        plantdoc_map[k.lower()] = v
                        plantdoc_map[k.replace(" ", "_").lower()] = v
                        plantdoc_map[k.replace("_", " ").lower()] = v
                    break
                except Exception as e:
                    logger.warning(f"Failed to load plantdoc_mapping.json from {mapping_path}: {e}")
        
        # Fallback dictionary for 28 PlantDoc classes
        fallback = {
            "Cherry_leaf": "Cherry_(including_sour)___healthy",
            "Peach_leaf": "Peach___healthy",
            "Corn_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
            "Apple_rust_leaf": "Apple___Cedar_apple_rust",
            "Potato_leaf_late_blight": "Potato___Late_blight",
            "Strawberry_leaf": "Strawberry___healthy",
            "Corn_rust_leaf": "Corn_(maize)___Common_rust_",
            "Tomato_leaf_late_blight": "Tomato___Late_blight",
            "Tomato_mold_leaf": "Tomato___Leaf_Mold",
            "Potato_leaf_early_blight": "Potato___Early_blight",
            "Apple_leaf": "Apple___healthy",
            "Tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Blueberry_leaf": "Blueberry___healthy",
            "Tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
            "Raspberry_leaf": "Raspberry___healthy",
            "Tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
            "Squash_Powdery_mildew_leaf": "Squash___Powdery_mildew",
            "grape_leaf": "Grape___healthy",
            "Corn_Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Tomato_Early_blight_leaf": "Tomato___Early_blight",
            "Apple_Scab_Leaf": "Apple___Apple_scab",
            "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
            "Tomato_leaf": "Tomato___healthy",
            "Soyabean_leaf": "Soybean___healthy",
            "Bell_pepper_leaf_spot": "Pepper,_bell___Bacterial_spot",
            "Bell_pepper_leaf": "Pepper,_bell___healthy",
            "grape_leaf_black_rot": "Grape___Black_rot",
            "Potato_leaf": "Potato___healthy",
            "Tomato_two_spotted_spider_mites_leaf": "Tomato___Spider_mites Two-spotted_spider_mite"
        }
        for k, v in fallback.items():
            if k not in plantdoc_map:
                plantdoc_map[k] = v
        return plantdoc_map

    def _load_plantvillage_samples(self) -> List[DatasetSample]:
        """Loads PlantVillage samples from split CSV or falls back to raw directory."""
        csv_candidates = [
            Path("outputs/splits") / f"plantvillage_{self.split}.csv",
            Path("ai_models/disease_detection/outputs/splits") / f"plantvillage_{self.split}.csv"
        ]
        csv_path = next((c for c in csv_candidates if c.exists()), None)
        
        samples = []
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            for row in df.itertuples():
                raw_rel = str(row.image_path).replace("\\", "/")
                filename = Path(raw_rel).name
                candidate_paths = [
                    Path(raw_rel),
                    Path("ai_models/disease_detection") / raw_rel,
                    Path("datasets/raw/plantvillage/color") / row.class_name / filename,
                    Path("ai_models/disease_detection/datasets/raw/plantvillage/color") / row.class_name / filename,
                ]
                img_p = next((p for p in candidate_paths if p.exists()), None)
                if img_p is not None:
                    samples.append(DatasetSample(
                        image_path=str(img_p),
                        class_name=row.class_name,
                        dataset="plantvillage"
                    ))
        
        # Direct folder scan fallback if CSV missing or empty
        if not samples:
            pv_dirs = [
                Path("datasets/raw/plantvillage/color"),
                Path("ai_models/disease_detection/datasets/raw/plantvillage/color")
            ]
            pv_dir = next((d for d in pv_dirs if d.exists()), None)
            if pv_dir is not None:
                logger.info(f"PlantVillage CSV not found/empty. Scanning raw directory: {pv_dir}")
                for class_dir in pv_dir.iterdir():
                    if class_dir.is_dir() and class_dir.name in self.class_to_index:
                        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                            for img_file in class_dir.glob(ext):
                                samples.append(DatasetSample(
                                    image_path=str(img_file),
                                    class_name=class_dir.name,
                                    dataset="plantvillage"
                                ))
        return samples

    def _load_plantdoc_samples(self) -> List[DatasetSample]:
        """Loads PlantDoc segmented/raw crops dynamically, mapping class names to 38 master classes."""
        plantdoc_map = self._load_plantdoc_mapping()

        csv_candidates = [
            Path("outputs/splits") / f"augmented_plantdoc_{self.split}.csv",
            Path("ai_models/disease_detection/outputs/splits") / f"augmented_plantdoc_{self.split}.csv",
            Path("outputs/splits") / f"plantdoc_classification_{self.split}.csv",
            Path("ai_models/disease_detection/outputs/splits") / f"plantdoc_classification_{self.split}.csv"
        ]
        # csv_candidates = [
        #     Path("outputs/splits") / f"plantdoc_classification_{self.split}.csv",
        #     Path("ai_models/disease_detection/outputs/splits") / f"plantdoc_classification_{self.split}.csv"
        # ]
        csv_path = next((c for c in csv_candidates if c.exists()), None)

        samples = []
        if csv_path is not None:
            logger.info(f"Loading PlantDoc samples for split '{self.split}' from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            for row in df.itertuples():
                raw_rel = str(row.image_path).replace("\\", "/")
                filename = Path(raw_rel).name
                raw_class_name = row.class_name
                master_class = plantdoc_map.get(
                    raw_class_name,
                    plantdoc_map.get(raw_class_name.replace("_", " "), raw_class_name)
                )

                if master_class in self.class_to_index:
                    candidate_paths = [
                        Path(raw_rel),
                        Path("ai_models/disease_detection") / raw_rel,
                        Path("datasets/raw/augmented_plantdoc") / self.split / raw_class_name / filename,
                        Path("ai_models/disease_detection/datasets/raw/augmented_plantdoc") / self.split / raw_class_name / filename,
                        Path("datasets/raw/plantdoc_classification") / self.split / raw_class_name / filename,
                        Path("ai_models/disease_detection/datasets/raw/plantdoc_classification") / self.split / raw_class_name / filename,
                        Path("datasets/processed/plantdoc_classification") / self.split / raw_class_name / filename,
                        Path("ai_models/disease_detection/datasets/processed/plantdoc_classification") / self.split / raw_class_name / filename,
                        Path("datasets/raw/augmented_plantdoc") / raw_class_name / filename,
                        Path("ai_models/disease_detection/datasets/raw/augmented_plantdoc") / raw_class_name / filename,
                        Path("datasets/raw/plantdoc_classification") / raw_class_name / filename,
                        Path("ai_models/disease_detection/datasets/raw/plantdoc_classification") / raw_class_name / filename,
                        Path("datasets/processed/plantdoc_classification") / raw_class_name / filename,
                        Path("ai_models/disease_detection/datasets/processed/plantdoc_classification") / raw_class_name / filename,
                    ]
                    img_p = next((p for p in candidate_paths if p.exists()), None)
                    if img_p is not None:
                        samples.append(DatasetSample(
                            image_path=str(img_p),
                            class_name=master_class,
                            dataset="plantdoc_classification"
                        ))

        if not samples:
            dir_split = self.split
            candidate_dirs = [
                Path("datasets/raw/augmented_plantdoc") / dir_split,
                Path("ai_models/disease_detection/datasets/raw/augmented_plantdoc") / dir_split,
                Path("datasets/raw/plantdoc_classification") / dir_split,
                Path("ai_models/disease_detection/datasets/raw/plantdoc_classification") / dir_split,
                Path("datasets/processed/plantdoc_classification") / dir_split,
                Path("ai_models/disease_detection/datasets/processed/plantdoc_classification") / dir_split,
                # Fallback to val if test dir is absent in older plantdoc
                Path("datasets/raw/plantdoc_classification") / ("val" if self.split == "test" else self.split),
                Path("ai_models/disease_detection/datasets/raw/plantdoc_classification") / ("val" if self.split == "test" else self.split),
                Path("datasets/raw/augmented_plantdoc"),
                Path("ai_models/disease_detection/datasets/raw/augmented_plantdoc"),
                Path("datasets/raw/plantdoc_classification"),
                Path("ai_models/disease_detection/datasets/raw/plantdoc_classification"),
            ]

            scanned_paths = set()
            for cd in candidate_dirs:
                if cd.exists():
                    for class_dir in cd.iterdir():
                        if class_dir.is_dir() and class_dir.name not in ["train", "val", "test", "__pycache__"]:
                            raw_class_name = class_dir.name
                            master_class = plantdoc_map.get(
                                raw_class_name,
                                plantdoc_map.get(raw_class_name.replace("_", " "), raw_class_name)
                            )
                            if master_class not in self.class_to_index:
                                continue

                            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                                for img_file in class_dir.glob(ext):
                                    if str(img_file) not in scanned_paths:
                                        scanned_paths.add(str(img_file))
                                        samples.append(DatasetSample(
                                            image_path=str(img_file),
                                            class_name=master_class,
                                            dataset="plantdoc_classification"
                                        ))

        logger.info(f"Loaded {len(samples)} PlantDoc samples mapped to master 38 classes for split '{self.split}'.")
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
