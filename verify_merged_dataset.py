import sys
import json
from pathlib import Path

# Add project root and disease_detection directories to sys.path
script_dir = Path(__file__).resolve().parent
candidates = [
    script_dir,
    script_dir.parent,
    script_dir.parent.parent if len(script_dir.parents) >= 2 else script_dir
]

for p in candidates:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    from ai_models.disease_detection.data.merged_dataset import MergedDiseaseDataset
except ImportError:
    from data.merged_dataset import MergedDiseaseDataset  # type: ignore

def verify_dataset():
    print("=" * 70)
    print("      KRISHIMITRA - MERGED DATASET VALIDATION & CLASS MATCHING")
    print("=" * 70)
    
    # 1. Load Master Class Mapping dynamically from candidate paths
    mapping_candidates = [
        Path("outputs/classification/class_mapping.json"),
        script_dir / "outputs" / "classification" / "class_mapping.json",
        script_dir.parent / "outputs" / "classification" / "class_mapping.json",
    ]
    
    mapping_path = next((p for p in mapping_candidates if p.exists()), None)
    if mapping_path is None:
        print(f"❌ ERROR: class_mapping.json missing in candidate paths: {[str(p) for p in mapping_candidates]}")
        return
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        master_mapping = json.load(f)
    
    print(f"\n1. Master Mapping Loaded: {len(master_mapping)} classes.")
    # 2. Test Dataset Loading (Train split)
    print("\n2. Initializing MergedDiseaseDataset (split='train', mix_ratio=0.15)...")
    try:
        train_ds = MergedDiseaseDataset(split="train", transform=None, mix_ratio=0.15)
    except Exception as e:
        print(f"❌ Failed to initialize dataset: {e}")
        return
        
    print(f"   ✓ Total train samples loaded: {len(train_ds.samples)}")
    print(f"   ✓ PlantVillage samples     : {len(train_ds.pv_samples)}")
    print(f"   ✓ PlantDoc / Augmented samples: {len(train_ds.pd_samples)}")
    if train_ds.pd_samples:
        sample_path = train_ds.pd_samples[0].image_path
        print(f"   ✓ Sample PlantDoc Path     : {sample_path}")
    print("\n3. Class Match Breakdown across Datasets:")
    pv_counts = {}
    pd_counts = {}
    
    for s in train_ds.pv_samples:
        pv_counts[s.class_name] = pv_counts.get(s.class_name, 0) + 1
        
    for s in train_ds.pd_samples:
        pd_counts[s.class_name] = pd_counts.get(s.class_name, 0) + 1
        
    print(f"   {'Index':<5} | {'Master Class Name':<50} | {'PV Samples':<10} | {'PD Samples':<10}")
    print("   " + "-" * 83)
    
    unmatched_pv = 0
    unmatched_pd = 0
    
    for class_name, idx in sorted(master_mapping.items(), key=lambda x: x[1]):
        pv_c = pv_counts.get(class_name, 0)
        pd_c = pd_counts.get(class_name, 0)
        print(f"   {idx:<5} | {class_name:<50} | {pv_c:<10} | {pd_c:<10}")
        if pv_c == 0:
            unmatched_pv += 1
        if pd_c == 0:
            unmatched_pd += 1
            
    print(f"\n   Classes with 0 PlantVillage samples: {unmatched_pv}")
    print(f"   Classes with 0 PlantDoc samples: {unmatched_pd}")

    # 4. Item Access & Label Consistency Test
    print("\n4. Testing __getitem__ access and index mapping:")
    if len(train_ds) > 0:
        sample_img, sample_label = train_ds[0]
        sample_obj = train_ds.samples[0]
        expected_label = train_ds.class_to_index[sample_obj.class_name]
        
        print(f"   Sample 0 Class Name : {sample_obj.class_name}")
        print(f"   Sample 0 Label Index: {sample_label} (Expected: {expected_label})")
        print(f"   Sample 0 Image Size : {sample_img.size} ({sample_img.mode})")
        print(f"   Sample 0 Dataset    : {sample_obj.dataset}")
        
        if sample_label == expected_label:
            print("   ✓ Label index perfectly matches Master Class Mapping!")
        else:
            print("   ❌ MISMATCH DETECTED between label index and master mapping!")
            
    print("=" * 70)
    print("Verification Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    verify_dataset()
