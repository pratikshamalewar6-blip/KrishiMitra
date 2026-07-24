"""
KrishiMitra - Knowledge Base Manager

Exposes the query API, keyword searches, and prompt context builder.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from common.logger import LoggerManager
from knowledge_base.exceptions import (
    DatabaseNotFoundError,
    ValidationError,
    DiseaseNotFoundError,
    CropNotFoundError,
)
from knowledge_base.validators import DatabaseValidator

logger = LoggerManager.get_logger("KnowledgeBaseManager")


class KnowledgeBaseManager:
    """
    Manager class to load, validate, search, and retrieve crop disease entries.
    """

    def __init__(self, database_path: Path | str | None = None, schema_path: Path | str | None = None) -> None:
        """
        Initialize the KnowledgeBaseManager.
        If paths are not provided, defaults to package directories.
        """
        pkg_dir = Path(__file__).parent
        self.database_path = Path(database_path) if database_path else pkg_dir / "disease_database.json"
        self.schema_path = Path(schema_path) if schema_path else pkg_dir / "disease_schema.json"
        self.validator = DatabaseValidator(self.schema_path)
        self.db_cache: dict = {}
        
        # Auto-load database upon initialization
        self.load_database()

    def load_database(self) -> None:
        """
        Loads the database into cache memory and runs full validations.
        """
        logger.info(f"Loading database from: {self.database_path}")
        if not self.database_path.exists():
            logger.error(f"Database file not found at: {self.database_path}")
            raise DatabaseNotFoundError(f"Database file not found at: {self.database_path}")
            
        try:
            # Validate and retrieve validated database dictionary
            self.db_cache = self.validate_database()
            logger.info(f"Successfully loaded and cached {len(self.db_cache)} disease records.")
        except ValidationError as ve:
            logger.error(f"Database validation failed during load: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"Unexpected error loading database: {e}")
            raise ValidationError(f"Failed to load database: {e}")

    def validate_database(self) -> dict:
        """
        Runs schema and custom business validation checks on the database file.
        Returns the parsed dictionary if valid.
        """
        logger.info("Executing database validation checks...")
        return self.validator.validate(self.database_path)

    def get_disease_by_class(self, class_name: str) -> dict:
        """
        Retrieves a disease entry matching the exact class name (e.g. 'Tomato___Bacterial_spot').
        Raises DiseaseNotFoundError if not found.
        """
        # Strip and normalize spaces
        class_name = class_name.strip()
        if class_name not in self.db_cache:
            # Try matching with replaced underscores or casing
            normalized_keys = {k.lower().replace(" ", ""): k for k in self.db_cache.keys()}
            norm_query = class_name.lower().replace(" ", "")
            if norm_query in normalized_keys:
                matched_key = normalized_keys[norm_query]
                logger.info(f"Fuzzy-matched query '{class_name}' to key '{matched_key}'")
                return self.db_cache[matched_key]
            
            logger.error(f"Class '{class_name}' not found in database.")
            raise DiseaseNotFoundError(f"Disease class '{class_name}' was not found in the knowledge base.")
            
        return self.db_cache[class_name]

    def get_crop_information(self, crop_name: str) -> list[dict]:
        """
        Retrieves all database entries corresponding to a specific crop name (exact case-insensitive).
        Raises CropNotFoundError if no entries match.
        """
        crop_name = crop_name.strip().lower()
        results = [
            entry for entry in self.db_cache.values()
            if entry.get("crop_name", "").strip().lower() == crop_name
        ]
        
        if not results:
            logger.error(f"No records found for crop: {crop_name}")
            raise CropNotFoundError(f"Crop '{crop_name}' was not found in the knowledge base.")
            
        return results

    def search_by_crop(self, crop_name: str) -> list[dict]:
        """
        Searches for crop records matching a partial or full crop name (case-insensitive).
        Returns an empty list if no matches are found.
        """
        crop_query = crop_name.strip().lower()
        if not crop_query:
            return []
            
        results = [
            entry for entry in self.db_cache.values()
            if crop_query in entry.get("crop_name", "").strip().lower()
        ]
        logger.info(f"Search for crop '{crop_name}' returned {len(results)} matches.")
        return results

    def list_all_crops(self) -> list[str]:
        """
        Returns a sorted list of all unique crop names in the database.
        """
        crops = sorted(list(set(entry.get("crop_name") for entry in self.db_cache.values() if entry.get("crop_name"))))
        return crops

    def list_all_diseases(self) -> list[str]:
        """
        Returns a sorted list of all unique disease names (excluding 'Healthy').
        """
        diseases = set()
        for entry in self.db_cache.values():
            d_name = entry.get("disease_name")
            if d_name and d_name.lower() != "healthy":
                diseases.add(d_name)
        return sorted(list(diseases))

    def search_by_keyword(self, keyword: str) -> list[dict]:
        """
        Searches the database for matching diseases using a case-insensitive keyword search.
        Searches across crop name, disease name, scientific name, symptoms, and causes.
        """
        query = keyword.strip().lower()
        if not query:
            return []
            
        results = []
        for entry in self.db_cache.values():
            # Check fields
            in_crop = query in entry.get("crop_name", "").lower()
            in_disease = query in entry.get("disease_name", "").lower()
            in_scientific = query in entry.get("scientific_name", "").lower()
            
            # Check list fields
            in_symptoms = any(query in sym.lower() for sym in entry.get("symptoms", []))
            in_causes = any(query in cause.lower() for cause in entry.get("causes", []))
            
            if in_crop or in_disease or in_scientific or in_symptoms or in_causes:
                results.append(entry)
                
        logger.info(f"Keyword search for '{keyword}' returned {len(results)} matches.")
        return results

    def export_json(self, path: Path | str) -> None:
        """
        Exports the current database to a specified file path.
        """
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(self.db_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Database exported successfully to: {export_path}")
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            raise IOError(f"Failed to export database to {export_path}: {e}")


def build_prompt_context(disease_name: str, confidence: float, kb_data: dict) -> dict:
    """
    Constructs a clean, structured JSON context specifically designed for
    direct prompt injection in the downstream Gemini AI model (Phase 6).
    
    Accepts:
        disease_name: Predicted class or display name
        confidence: Confidence score from the classifier model (float, e.g., 0.9982)
        kb_data: Dictionary containing the mapped database record
    Returns:
        Structured context dictionary.
    """
    # Normalize confidence to percentage display
    conf_percentage = confidence * 100.0 if confidence <= 1.0 else confidence
    conf_percentage = round(conf_percentage, 2)

    context = {
        "crop": kb_data.get("crop_name"),
        "disease": kb_data.get("disease_name"),
        "confidence": conf_percentage,
        "symptoms": kb_data.get("symptoms", []),
        "causes": kb_data.get("causes", []),
        "organic_treatment": kb_data.get("organic_treatment", []),
        "chemical_treatment": kb_data.get("chemical_treatment", []),
        "fertilizer": kb_data.get("recommended_fertilizers", []),
        "prevention": kb_data.get("preventive_measures", []),
        "farmer_tips": kb_data.get("farmer_tips", []),
        "severity": kb_data.get("risk_level"),
        "weather": kb_data.get("weather_conditions_favoring_disease"),
        "references": kb_data.get("trusted_references", [])
    }
    return context
