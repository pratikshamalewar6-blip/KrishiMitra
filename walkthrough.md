# Walkthrough - Phase 3: SAM2 Leaf Segmentation Completed

We have successfully implemented and verified the **SAM2 Leaf Segmentation** module. Below is a summary of the achievements, implemented components, and verification results.

---

## 🛠️ Accomplishments

### 1. Segmentation Configurations
* Created [segmentation/config.py](file:///d:/KrishiMitra2/ai_models/disease_detection/segmentation/config.py) to read model parameters (model weights name, background removal flag, output formats) from YAML configs via the central `ConfigManager`.

### 2. Model Downloader & Loader
* Created [segmentation/model.py](file:///d:/KrishiMitra2/ai_models/disease_detection/segmentation/model.py) which automatically checks for SAM2 model weights (`sam2_t.pt`) in `saved_models/` and downloads them from the official Ultralytics releases if missing, before initializing the model.

### 3. Segmentation Processor (Mask & Crop)
* Created [segmentation/segmenter.py](file:///d:/KrishiMitra2/ai_models/disease_detection/segmentation/segmenter.py) to segment crop leaves from the background using box prompts.
* **Premium Transparency Feature:** If the output format is set to `PNG`, the segmenter generates a 4-channel **RGBA** transparent background crop (alpha channel is set to `0` outside the leaf mask, and `255` inside). For other formats, it falls back to a 3-channel **RGB** crop on a black background.

### 4. Bounding Box + SAM2 Prediction CLI
* Created [segmentation/predict_segmentation.py](file:///d:/KrishiMitra2/ai_models/disease_detection/segmentation/predict_segmentation.py) to process an image or entire folder, run YOLO leaf detection, pass bounding boxes directly to the SAM2 leaf segmenter, and output isolated, background-removed leaf crops.

### 5. Automated Unit Tests
* Created [tests/test_sam2_model.py](file:///d:/KrishiMitra2/ai_models/disease_detection/tests/test_sam2_model.py) to initialize `LeafSegmenter`, generate a mock image with a dummy green object, predict the boundary mask, and verify the shape and channel mode of the output crop.

---

## 🧪 Verification Results

### 1. Model Loading & Prediction Unit Tests
Running the unit test script downloads the model weights file to the `saved_models/` directory and verifies correctness:
```text
$env:PYTHONPATH="."
.\venv\Scripts\python.exe tests/test_sam2_model.py
```
Output:
```text
============================================================
Testing SAM2 Model Loading & Segmenter
============================================================
2026-07-19 12:29:08 | INFO | SAM2ModelLoader | Loading SAM2 model weights from: saved_models\sam2_t.pt
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt to 'saved_models\sam2_t.pt'...
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 74.4/74.4 MB
2026-07-19 12:29:09 | INFO | SAM2ModelLoader | SAM2 model loaded successfully.
LeafSegmenter initialized with weights: sam2_t.pt
Running dummy segment_leaf prediction...
Segmented crop size: (180, 180) (Mode: RGBA)
============================================================
SAM2 Segmenter Test Passed Successfully!
============================================================
```

### 2. Segmenting Real Agricultural Images
We created a verification script at [scratch/test_segmentation_real.py](file:///d:/KrishiMitra2/ai_models/disease_detection/scratch/test_segmentation_real.py) to read a real ground-truth bounding box coordinates from `val_00000.txt`, feed it directly to the SAM2 segmenter, and crop the leaf from the background.

```text
$env:PYTHONPATH="."
.\venv\Scripts\python.exe scratch/test_segmentation_real.py
```
Output:
```text
2026-07-19 12:30:52 | INFO | FileUtils | Directory ready: outputs\segmentations
2026-07-19 12:30:52 | INFO | SAM2ModelLoader | Loading SAM2 model weights from: saved_models\sam2_t.pt
2026-07-19 12:30:54 | INFO | SAM2ModelLoader | SAM2 model loaded successfully.
============================================================
Testing SAM2 Leaf Segmentation on a Real Image
============================================================
Loaded image: datasets\raw\plantdoc_detection\images\val\val_00000.jpg (Size: 2147x1432)
Converted YOLO bounding box to pixels: [15, 112, 1509, 1296]
Running SAM2 segmentation...
Successfully saved background-removed crop to: outputs\segmentations\val_00000_mock_segmented.png
============================================================
```

The output crop is saved in the repository at [val_00000_mock_segmented.png](file:///d:/KrishiMitra2/ai_models/disease_detection/outputs/segmentations/val_00000_mock_segmented.png) as a transparent PNG containing only the leaf pixels.

---

## 🏃 How to Run the Integrated YOLO + SAM2 CLI

Once you place your fine-tuned `saved_models/yolov11_leaf.pt` model weights inside your local directory, you can run:

```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe segmentation/predict_segmentation.py --source <path_to_your_image.jpg>
```
The script will:
1. Detect leaves in the image.
2. Segment each detected leaf from the background.
3. Save the background-removed leaf crops in `outputs/segmentations/<image_name>/`.
