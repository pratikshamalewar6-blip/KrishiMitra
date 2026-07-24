"""
KrishiMitra - AI End-to-End Prediction Pipeline

Coordinates YOLOv11 leaf detection, SAM2 leaf segmentation,
EfficientNet-B0 crop disease classification, and Knowledge Base retrieval.

Author:
    Antigravity AI
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from common.logger import LoggerManager
from common.file_utils import FileUtils
from detection.detector import LeafDetector
from segmentation.segmenter import LeafSegmenter
from segmentation.config import SegmentationConfig
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier
from data.transforms import get_test_transforms
from knowledge_base.knowledge_base_manager import KnowledgeBaseManager, build_prompt_context

logger = LoggerManager.get_logger("PredictionPipeline")


class PredictionPipeline:
    """
    End-to-End Prediction Pipeline for crop disease diagnosis.
    """

    def __init__(self, device: str | None = None) -> None:
        """
        Initializes and loads all models and managers.
        """
        import os
        # Set current working directory to AI models root to align relative config paths
        pkg_root = Path(__file__).resolve().parent.parent
        os.chdir(pkg_root)

        logger.info("Initializing end-to-end prediction pipeline...")

        # 1. Device configuration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Pipeline running on target device: {self.device}")

        # 2. Setup config files
        self.class_config = ClassificationConfig()
        self.class_config.DEVICE = self.device
        
        self.seg_config = SegmentationConfig()
        self.seg_config.DEVICE = self.device
        self.seg_config.OUTPUT_FORMAT = "PNG"  # Ensure transparent background

        # 3. Load Class Mappings
        self.mapping_file = self.class_config.OUTPUT_DIRECTORY / "class_mapping.json"
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Class mapping file not found at: {self.mapping_file}")
            
        with open(self.mapping_file, "r", encoding="utf-8") as f:
            self.class_mapping = json.load(f)
        self.index_to_class = {idx: name for name, idx in self.class_mapping.items()}

        # 4. Load Models
        self.detector = LeafDetector()
        self.segmenter = LeafSegmenter(self.seg_config)
        self.classifier = self._load_classifier()
        
        # 5. Load Knowledge Base Manager
        self.kb_manager = KnowledgeBaseManager()

        logger.info("End-to-End prediction pipeline fully loaded and ready.")

    def _load_classifier(self) -> DiseaseClassifier:
        """Loads DiseaseClassifier selecting the real-world weights if available."""
        classifier = DiseaseClassifier(self.class_config)
        
        # Determine weights file: realworld first, fallback to standard
        realworld_path = Path("saved_models/efficientnet_b0_realworld.pt")
        standard_path = self.class_config.MODEL_FILE

        if realworld_path.exists():
            weights_path = realworld_path
            logger.info(f"Selecting improved real-world model weights: {weights_path}")
        elif standard_path.exists():
            weights_path = standard_path
            logger.info(f"Selecting base model weights: {weights_path}")
        else:
            raise FileNotFoundError(
                f"No classifier weights found. Searched:\n - {realworld_path}\n - {standard_path}"
            )

        classifier.load_state_dict(torch.load(weights_path, map_location=self.device))
        classifier.to(self.device)
        classifier.eval()
        return classifier

    def _predict_disease_pil(
        self,
        img: Image.Image,
        ood_threshold: float = 0.60
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Runs classifier inference directly on PIL Image memory."""
        # Convert RGBA/PNG to RGB (yielding black background for transparent pixels, matching training setup)
        rgb_img = img.convert("RGB")
        
        transform = get_test_transforms()
        img_tensor = transform(rgb_img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

        # Get Top-5
        topk_probs, topk_indices = torch.topk(probabilities, k=min(5, len(probabilities)))
        
        top5_predictions = []
        for prob, idx in zip(topk_probs, topk_indices):
            prob_val = float(prob.item())
            class_name = self.index_to_class.get(int(idx.item()), f"Class_{idx.item()}")
            top5_predictions.append((class_name, prob_val))

        top1_class, top1_prob = top5_predictions[0]

        # Out-of-Distribution (OOD) check
        if top1_prob < ood_threshold:
            predicted_class = "Unknown Disease"
        else:
            predicted_class = top1_class

        return predicted_class, top1_prob, top5_predictions

    def _draw_overlay(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        label: str,
        confidence: float
    ) -> None:
        """Draws bounding box and label text overlay directly on input image."""
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, outline="green", width=4)
        
        text = f"{label} ({confidence*100:.1f}%)"
        x1, y1, _, _ = bbox
        
        # Draw small background text block
        draw.rectangle([x1, y1 - 20, x1 + len(text) * 7, y1], fill="green")
        draw.text((x1 + 5, y1 - 17), text, fill="white")

    def predict(
        self,
        image_path: str | Path,
        ood_threshold: float = 0.60,
        save_visuals: bool = True
    ) -> dict:
        """
        Executes end-to-end diagnosis pipeline on a single image.
        
        Parameters
        ----------
        image_path : str | Path
            Path to the input image file.
        ood_threshold : float
            Confidence threshold for Out-of-Distribution (OOD) rejection.
        save_visuals : bool
            Whether to save visual Crops and Annotated overlay plots.
            
        Returns
        -------
        dict
            Structured prediction payload including leaf counts, segments, classifications,
            and prompt context mapping for Gemini.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")

        # 1. Run YOLOv11 Leaf Detection
        detections = self.detector.detect(img_path)
        
        # Ensure output directory for this run
        pipeline_out_dir = Path("outputs/pipeline")
        run_out_dir = pipeline_out_dir / img_path.stem
        
        if save_visuals:
            FileUtils.ensure_directory(run_out_dir)

        # Load PIL image for segmenting and overlays
        pil_img = Image.open(img_path).convert("RGB")
        overlay_img = pil_img.copy()

        leaves_data = []

        # 2. Iterate and process each detected leaf
        for idx, det in enumerate(detections):
            det_box = (det.x1, det.y1, det.x2, det.y2)

            # Stage 2: SAM2 Segmentation (Transparent Crop)
            segmented_crop = self.segmenter.segment_leaf(pil_img, det_box)
            
            # Save segmented crop
            crop_path = None
            if save_visuals:
                crop_file = run_out_dir / f"leaf_{idx + 1}_segmented.png"
                segmented_crop.save(crop_file, "PNG")
                crop_path = str(crop_file)

            # Stage 3: EfficientNet-B0 Disease Classification
            pred_class, confidence, top5 = self._predict_disease_pil(segmented_crop, ood_threshold)

            # Stage 4: Knowledge Base prompt context lookup
            try:
                kb_data = self.kb_manager.get_disease_by_class(pred_class)
            except Exception:
                # Fallback if class not found or OOD "Unknown Disease"
                kb_data = {
                    "crop_name": "Unknown",
                    "disease_name": "Unknown",
                    "symptoms": ["No symptoms cataloged."],
                    "causes": ["Unknown cause."],
                    "organic_treatment": [],
                    "chemical_treatment": [],
                    "recommended_fertilizers": [],
                    "preventive_measures": [],
                    "farmer_tips": [],
                    "risk_level": "N/A",
                    "weather_conditions_favoring_disease": "N/A",
                    "trusted_references": []
                }
            prompt_context = build_prompt_context(pred_class, confidence, kb_data)

            # Draw visual overlay on master image
            label_display = pred_class.split("___")[-1].replace("_", " ") if "___" in pred_class else pred_class
            self._draw_overlay(overlay_img, det_box, label_display, confidence)

            leaves_data.append({
                "leaf_index": idx + 1,
                "box": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                "detection_confidence": float(det.confidence),
                "crop_path": crop_path,
                "predicted_class": pred_class,
                "classification_confidence": float(confidence),
                "top5_predictions": [(cls_name, float(prob)) for cls_name, prob in top5],
                "gemini_prompt_context": prompt_context
            })

        # Save visual overlay
        overlay_path = None
        if save_visuals and len(detections) > 0:
            overlay_file = pipeline_out_dir / f"{img_path.stem}_annotated.jpg"
            overlay_img.save(overlay_file, "JPEG")
            overlay_path = str(overlay_file)
            logger.info(f"Visual overlay saved to: {overlay_file}")

        # Assemble final prediction report structure
        report = {
            "status": "success",
            "image_path": str(img_path.resolve()),
            "leaves_found": len(detections),
            "annotated_image_path": overlay_path,
            "results": leaves_data
        }

        return report
