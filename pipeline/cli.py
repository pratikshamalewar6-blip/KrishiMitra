"""
KrishiMitra - AI End-to-End Prediction Pipeline CLI Tool

Executes joint leaf detection, segmentation, classification, and knowledge base retrieval
on target images and dumps structured prediction JSON reports.

Author:
    Antigravity AI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.logger import LoggerManager
from pipeline.prediction_pipeline import PredictionPipeline

logger = LoggerManager.get_logger("PredictionPipelineCLI")


def process_image(
    pipeline: PredictionPipeline,
    img_path: Path,
    threshold: float,
    save_visuals: bool
) -> dict | None:
    """Processes a single image file through the prediction pipeline and saves reports."""
    logger.info(f"Running pipeline on: {img_path}")
    try:
        report = pipeline.predict(img_path, ood_threshold=threshold, save_visuals=save_visuals)
        
        # Save JSON prediction payload
        out_report_file = Path("outputs/pipeline") / f"{img_path.stem}_prediction.json"
        out_report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Structured JSON prediction report saved to: {out_report_file}")
        return report
    except Exception as e:
        logger.error(f"Pipeline failed on image {img_path.name}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="KrishiMitra End-to-End Crop Disease Prediction CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="Path to a single raw plant image",
    )
    group.add_argument(
        "--dir",
        type=str,
        help="Path to a directory containing raw plant images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Out-of-Distribution (OOD) threshold (default: 0.60)",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip saving crop PNG images and annotated visual overlays",
    )
    args = parser.parse_args()

    # Determine save visuals flag
    save_visuals = not args.no_visuals

    # 1. Initialize Pipeline
    try:
        pipeline = PredictionPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize prediction pipeline: {e}")
        return

    # 2. Process Input
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"Specified image file not found: {img_path}")
            return
            
        report = process_image(pipeline, img_path, args.threshold, save_visuals)
        if report:
            # Print simplified console summary
            print("\n" + "=" * 60)
            print("END-TO-END PIPELINE DIAGNOSIS SUMMARY")
            print("=" * 60)
            print(f"Source Image    : {img_path.name}")
            print(f"Leaves Detected : {report['leaves_found']}")
            if report['annotated_image_path']:
                print(f"Overlay Plot    : {report['annotated_image_path']}")
            print("-" * 60)
            
            for idx, result in enumerate(report["results"]):
                label = result["predicted_class"]
                label_display = label.split("___")[-1].replace("_", " ") if "___" in label else label
                conf = result["classification_confidence"]
                print(f"Leaf #{idx + 1}:")
                print(f"  Coordinates   : {result['box']}")
                print(f"  Diagnosis     : {label_display}")
                print(f"  Confidence    : {conf * 100:.2f}%")
                
                # Knowledge Base details
                ctx = result["gemini_prompt_context"]
                if ctx and ctx.get("crop") != "Unknown":
                    print(f"  Crop Family   : {ctx.get('crop')}")
                    print(f"  Severity      : {ctx.get('severity')}")
                    if ctx.get("symptoms"):
                        print(f"  Symptoms (1)  : {ctx['symptoms'][0]}")
                print("-" * 60)
            print("=" * 60 + "\n")

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Specified directory not found or invalid: {dir_path}")
            return

        image_files = sorted([
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        logger.info(f"Found {len(image_files)} images in directory: {dir_path}")
        success_count = 0
        
        for img_file in image_files:
            report = process_image(pipeline, img_file, args.threshold, save_visuals)
            if report:
                success_count += 1

        logger.info(f"Processed {success_count}/{len(image_files)} images successfully.")


if __name__ == "__main__":
    main()
