"""
KrishiMitra - Prediction Pipeline Integration Tests

Verifies end-to-end model inference, background removal,
knowledge base mapping, and prompt context building.

Author:
    Antigravity AI
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pipeline.prediction_pipeline import PredictionPipeline


class TestPredictionPipeline(unittest.TestCase):
    def test_pipeline_inference(self):
        print("\nStarting Prediction Pipeline End-to-End Test...")
        
        # 1. Initialize Pipeline (resolves cuda/cpu automatically)
        pipeline = PredictionPipeline()
        
        # 2. Path to validation image
        img_path = Path("datasets/raw/plantdoc_detection/images/val/val_00000.jpg")
        self.assertTrue(img_path.exists(), f"Validation image not found: {img_path}")

        # 3. Execute inference
        report = pipeline.predict(img_path, ood_threshold=0.60, save_visuals=True)

        # 4. Asserts
        self.assertEqual(report["status"], "success")
        self.assertTrue("image_path" in report)
        self.assertTrue("leaves_found" in report)
        
        print(f"  Leaves detected in test image: {report['leaves_found']}")
        
        if report["leaves_found"] > 0:
            self.assertTrue(report["annotated_image_path"] is not None)
            self.assertTrue(Path(report["annotated_image_path"]).exists())
            
            # Check the first leaf result
            first_leaf = report["results"][0]
            self.assertTrue("box" in first_leaf)
            self.assertTrue("crop_path" in first_leaf)
            self.assertTrue("predicted_class" in first_leaf)
            self.assertTrue("classification_confidence" in first_leaf)
            
            # Check crop files generated
            self.assertTrue(Path(first_leaf["crop_path"]).exists())

            # Check prompt context
            ctx = first_leaf["gemini_prompt_context"]
            self.assertTrue("crop" in ctx)
            self.assertTrue("disease" in ctx)
            self.assertTrue("symptoms" in ctx)
            self.assertTrue("prevention" in ctx)
            self.assertTrue("severity" in ctx)
            
            print(f"  First Leaf Predicted Class : {first_leaf['predicted_class']}")
            print(f"  First Leaf Classification Confidence : {first_leaf['classification_confidence']*100:.2f}%")
            print(f"  Mapped Crop Family         : {ctx['crop']}")
            print(f"  Severity                   : {ctx['severity']}")
            
        print("Prediction pipeline integration test completed successfully!")


if __name__ == "__main__":
    unittest.main()
