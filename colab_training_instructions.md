# 🚀 Step-by-Step Training Guide: YOLOv11 Leaf Detector on Google Colab

We have packaged the dataset and updated the training notebook so that you are fully prepared to train the model on a GPU in Google Colab. Follow the steps below to complete the training.

---

## 📦 Step 1: Locate Your Zipped Dataset
We have generated the zip archive containing all images and single-class mapped annotations:
* **Location:** `datasets/processed/plantdoc_leaf.zip` (Size: **955.12 MB**)

---

## 🌐 Step 2: Open and Configure Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload the local Jupyter Notebook file:
   * [Disease_Detection_Training.ipynb](file:///d:/KrishiMitra2/ai_models/disease_detection/notebooks/Disease_Detection_Training.ipynb)
3. Set your runtime type to **GPU** for accelerated training:
   * In the top menu, go to **Runtime** > **Change runtime type**.
   * Select **T4 GPU** (or any other available GPU accelerator) and click **Save**.

---

## 📤 Step 3: Upload the Dataset to Colab
Choose one of the two options below:

### **Option A: Direct Upload (Recommended for fast start)**
1. Open the left sidebar panel in Colab (folder icon).
2. Drag and drop `plantdoc_leaf.zip` from your PC file explorer into the Colab storage panel (to `/content/`).
3. Run the **Option A** cell in the notebook to extract it.

### **Option B: Google Drive (Recommended if connection is unstable)**
1. Create a folder named `KrishiMitra` in your Google Drive.
2. Upload `plantdoc_leaf.zip` to that folder.
3. Run the **Option B** cells in the notebook to mount Google Drive and extract the zip folder.

---

## 🏋️ Step 4: Run Training in the Notebook
Run the notebook cells sequentially:
1. **Install Dependencies:** Installs `ultralytics`.
2. **Extract Dataset:** Unzips the dataset.
3. **Create YAML:** Creates `dataset_colab.yaml` pointing to the extracted path.
4. **Train Model:** Loads `yolo11n.pt` and runs training for **100 epochs** on the GPU (`device=0`).
5. **Evaluate Results:** Evaluates on the validation split and prints the final Precision, Recall, mAP50, and mAP50-95 scores.

---

## 💾 Step 5: Download and Update Model Weights
1. Once training is complete, the notebook will automatically download the `best.pt` file to your PC. If the direct download fails, copy it from your Google Drive destination.
2. Rename the downloaded file to **`yolov11_leaf.pt`**.
3. Place this file inside your project directory at:
   * [saved_models/yolov11_leaf.pt](file:///d:/KrishiMitra2/ai_models/disease_detection/saved_models/yolov11_leaf.pt) (Overwrite the existing file)

---

## 📸 Step 6: Test locally on Real Images
Take a few crop photos with your phone (e.g. tomato plant, mango leaf, cotton leaf, etc.) and save them as `test.jpg` in your project root.

Run the inference script to test the model:
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe detection/predict_detector.py --source test.jpg
```

### Inspect the output in `outputs/detections/`:
* **Is every leaf detected?** Check if the bounding boxes cover all leaves in the image.
* **Are there false detections?** Make sure random background elements aren't misdetected as leaves.
* **Does it miss leaves?** Ensure the recall is high.

---

> [!NOTE]
> Training 100 epochs on a Colab T4 GPU typically takes around **20-40 minutes**.
