import sys
import json
from pathlib import Path

# Add project root and disease_detection directories to sys.path
script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_dir))

def preflight_check():
    print("=" * 70)
    print("      KRISHIMITRA - REAL-WORLD FINE-TUNING PRE-FLIGHT CHECKLIST")
    print("=" * 70)

    # 1. Check Model Checkpoint Files
    disease_pt = script_dir / "saved_models" / "efficientnet_b0_disease.pt"
    realworld_pt = script_dir / "saved_models" / "efficientnet_b0_realworld.pt"
    
    print("\n1. Model Checkpoints:")
    if realworld_pt.exists():
        print(f"   ✓ Existing Real-World Checkpoint: {realworld_pt} ({realworld_pt.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print("   ℹ️ Real-World Checkpoint not found yet (will be created upon best accuracy).")

    if disease_pt.exists():
        print(f"   ✓ Stage 1 PlantVillage Base Checkpoint: {disease_pt} ({disease_pt.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"   ⚠️ Base Checkpoint missing at: {disease_pt}")

    # 2. Check Master Class Mapping
    mapping_file = script_dir / "outputs" / "classification" / "class_mapping.json"
    print("\n2. Class Mapping:")
    if mapping_file.exists():
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        print(f"   ✓ Master Class Mapping File: {mapping_file} ({len(mapping)} classes)")
    else:
        print(f"   ❌ ERROR: Master Class Mapping missing at {mapping_file}")

    # 3. Check Augmented PlantDoc Dataset Directory
    aug_dir = script_dir / "datasets" / "raw" / "augmented_plantdoc"
    print("\n3. Augmented PlantDoc Dataset:")
    if aug_dir.exists():
        print(f"   ✓ Directory Exists: {aug_dir}")
        for sub in ["train", "val", "test"]:
            sub_p = aug_dir / sub
            if sub_p.exists():
                classes_found = [d for d in sub_p.iterdir() if d.is_dir()]
                print(f"     - '{sub}' split: {len(classes_found)} class folders found")
            else:
                print(f"     - '{sub}' split: Not found at {sub_p}")
    else:
        print(f"   ℹ️ Directory not created yet at: {aug_dir}")

    # 4. Check Dataset Loader import
    print("\n4. Code Imports & Modules:")
    try:
        from data.merged_dataset import MergedDiseaseDataset
        from classification.config import ClassificationConfig
        from classification.model import DiseaseClassifier
        print("   ✓ MergedDiseaseDataset, ClassificationConfig & DiseaseClassifier imported cleanly.")
    except Exception as e:
        print(f"   ❌ Import Error: {e}")

    print("=" * 70)
    print("Pre-Flight Verification Complete!")
    print("=" * 70)

if __name__ == "__main__":
    preflight_check()
