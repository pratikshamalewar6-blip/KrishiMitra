"""
KrishiMitra - Knowledge Base Unit Tests

Verifies database loading, validation rules, keyword searches, fuzzy matching,
exceptions, and alignment with class_mapping.json.

Author:
    Antigravity AI
"""

from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from knowledge_base.exceptions import (
    DatabaseNotFoundError,
    ValidationError,
    DiseaseNotFoundError,
    CropNotFoundError,
)
from knowledge_base.knowledge_base_manager import KnowledgeBaseManager, build_prompt_context


class TestKnowledgeBase(unittest.TestCase):
    """
    Test suite for the crop disease knowledge base.
    """

    def setUp(self) -> None:
        """Set up manager and test paths."""
        self.manager = KnowledgeBaseManager()
        self.db_path = self.manager.database_path
        self.schema_path = self.manager.schema_path

    def test_database_and_schema_exist(self) -> None:
        """Verify database and schema files are present."""
        self.assertTrue(self.db_path.exists(), "disease_database.json is missing.")
        self.assertTrue(self.schema_path.exists(), "disease_schema.json is missing.")

    def test_database_loads_and_is_valid(self) -> None:
        """Verify the database loads successfully without schema errors."""
        self.assertGreater(len(self.manager.db_cache), 0)
        # Should not raise any validation exceptions on first load
        self.manager.load_database()

    def test_all_classes_fully_mapped(self) -> None:
        """
        Verify that all 38 classes defined in class_mapping.json
        have a corresponding entry in disease_database.json.
        """
        mapping_path = Path("outputs/classification/class_mapping.json")
        self.assertTrue(mapping_path.exists(), "class_mapping.json must exist for this test.")

        with open(mapping_path, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)

        # 38 classes expected
        self.assertEqual(len(class_mapping), 38, f"Expected 38 classes in mapping, found {len(class_mapping)}")

        for class_name in class_mapping.keys():
            # Every class should load from the manager without raising DiseaseNotFoundError
            try:
                data = self.manager.get_disease_by_class(class_name)
                self.assertIsNotNone(data)
                self.assertEqual(data["disease_id"], class_name)
            except DiseaseNotFoundError:
                self.fail(f"Class '{class_name}' from class_mapping.json is missing in disease_database.json!")

    def test_search_by_crop_name(self) -> None:
        """Verify case-insensitive crop searching."""
        # Exact match
        tomato_diseases = self.manager.get_crop_information("Tomato")
        self.assertGreater(len(tomato_diseases), 0)
        
        # All matched items should belong to Tomato
        for item in tomato_diseases:
            self.assertEqual(item["crop_name"].lower(), "tomato")

        # Case-insensitive partial search
        matches = self.manager.search_by_crop("tom")
        self.assertGreater(len(matches), 0)
        self.assertEqual(len(matches), len(tomato_diseases))

        # Non-existent crop raises exception
        with self.assertRaises(CropNotFoundError):
            self.manager.get_crop_information("Mango")

    def test_search_by_keyword(self) -> None:
        """Verify keyword search returns expected records."""
        # Search by scientific name
        results = self.manager.search_by_keyword("infestans")
        self.assertGreater(len(results), 0)
        
        # Phytophthora infestans should match Potato Late Blight and Tomato Late Blight
        disease_ids = [r["disease_id"] for r in results]
        self.assertIn("Potato___Late_blight", disease_ids)
        self.assertIn("Tomato___Late_blight", disease_ids)

        # Search by symptom keyword
        results_symptom = self.manager.search_by_keyword("velvety")
        self.assertGreater(len(results_symptom), 0)

        # Empty query
        self.assertEqual(len(self.manager.search_by_keyword("")), 0)

    def test_list_helpers(self) -> None:
        """Verify list helpers return unique sorted names."""
        crops = self.manager.list_all_crops()
        self.assertIn("Tomato", crops)
        self.assertIn("Apple", crops)
        self.assertEqual(crops, sorted(crops))

        diseases = self.manager.list_all_diseases()
        self.assertIn("Apple Scab", diseases)
        self.assertNotIn("Healthy", diseases, "Healthy classes should be filtered from list_all_diseases().")
        self.assertEqual(diseases, sorted(diseases))

    def test_prompt_context_builder(self) -> None:
        """Verify build_prompt_context structure."""
        kb_data = self.manager.get_disease_by_class("Tomato___Early_blight")
        context = build_prompt_context("Tomato___Early_blight", 0.9852, kb_data)

        # Required fields in prompt context
        required_keys = [
            "crop", "disease", "confidence", "symptoms", "causes",
            "organic_treatment", "chemical_treatment", "fertilizer",
            "prevention", "farmer_tips", "severity", "weather", "references"
        ]
        for key in required_keys:
            self.assertIn(key, context)

        self.assertEqual(context["crop"], "Tomato")
        self.assertEqual(context["disease"], "Early Blight")
        self.assertEqual(context["confidence"], 98.52)
        self.assertEqual(context["severity"], "Medium")

    def test_invalid_json_handling(self) -> None:
        """Verify the validator catches syntax errors in JSON database."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write("{ invalid json syntax }")
            temp_path = Path(temp_file.name)

        try:
            with self.assertRaises(ValidationError):
                self.manager.validator.validate(temp_path)
        finally:
            temp_path.unlink()

    def test_duplicate_keys_validation(self) -> None:
        """Verify validator raises ValidationError on duplicate keys in database."""
        raw_json_with_duplicate = """
        {
            "Apple___healthy": {
                "disease_id": "Apple___healthy",
                "crop_name": "Apple",
                "disease_name": "Healthy"
            },
            "Apple___healthy": {
                "disease_id": "Apple___healthy",
                "crop_name": "Apple",
                "disease_name": "Healthy"
            }
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(raw_json_with_duplicate)
            temp_path = Path(temp_file.name)

        try:
            with self.assertRaises(ValidationError) as ctx:
                self.manager.validator.validate(temp_path)
            self.assertIn("Duplicate key detected", str(ctx.exception))
        finally:
            temp_path.unlink()

    def test_mismatched_id_validation(self) -> None:
        """Verify validator checks that key matches disease_id."""
        bad_db_data = {
            "Apple___healthy": {
                "disease_id": "Mismatched_Key_Here",
                "crop_name": "Apple",
                "disease_name": "Healthy",
                "scientific_name": "None",
                "disease_category": "Healthy",
                "pathogen_type": "None",
                "symptoms": ["Leaf looks healthy"],
                "causes": ["Good care"],
                "spread_method": "None",
                "risk_level": "Low",
                "disease_severity_description": "None",
                "affected_plant_parts": ["None"],
                "weather_conditions_favoring_disease": "None",
                "growth_stage_affected": "All",
                "organic_treatment": ["None"],
                "chemical_treatment": ["None"],
                "recommended_fungicides": ["None"],
                "recommended_pesticides": ["None"],
                "recommended_fertilizers": ["None"],
                "recommended_bio_control_methods": ["None"],
                "preventive_measures": ["None"],
                "recommended_irrigation_advice": "None",
                "recovery_time": "None",
                "expected_yield_loss": "0%",
                "dos": ["None"],
                "donts": ["None"],
                "government_advisory": "None",
                "trusted_references": ["None"],
                "frequently_asked_questions": [{"question": "Q", "answer": "A"}],
                "farmer_tips": ["None"]
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as temp_file:
            json.dump(bad_db_data, temp_file)
            temp_path = Path(temp_file.name)

        try:
            with self.assertRaises(ValidationError) as ctx:
                self.manager.validator.validate(temp_path)
            self.assertIn("Mismatched ID", str(ctx.exception))
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
