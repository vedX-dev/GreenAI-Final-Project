# 🛰️ Satellite Drought Risk Detector

> **GreenAI · Skill4Future · Module 5 — Computer Vision for Green Technology**  
> Classify land use from satellite imagery and assess drought risk using a custom CNN trained on the EuroSAT dataset.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-4CAF50?style=flat-square)](https://github.com/phelber/EuroSAT)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95%25-27ae60?style=flat-square)](#results)

---

## Overview

This project trains a convolutional neural network (CNN) on Sentinel-2 satellite imagery to classify land cover into 10 categories, then maps each category to a contextual **drought risk interpretation**. The trained model is served through a dark-themed Streamlit web application where any user can upload a satellite or aerial image and get an instant land classification and drought advisory.

**The full pipeline:**

```
Satellite Image  →  Preprocessing  →  Custom CNN  →  Land Class  →  Drought Risk Label  →  Advisory
```

---

## Demo

Upload any satellite image (Google Earth screenshot, Landsat tile, Sentinel-2 image, or even a regular aerial photo) and the app returns:

- The predicted **land cover class** with confidence score
- A **drought risk interpretation** (Healthy Vegetation / Agricultural Land / Water Body / Urban)
- A **contextual advisory** specific to the land type
- A **probability bar chart** across all 10 classes

---

## Dataset — EuroSAT

| Property | Value |
|---|---|
| Source | Sentinel-2 satellite (ESA Copernicus programme) |
| Total images | 27,000 |
| Classes | 10 |
| Image size | 64 × 64 px |
| Channels | RGB (3-band) |
| Samples per class | ~2,000–3,000 |

### Land Classes & Drought Mapping

| Class | Drought Interpretation | Risk Level |
|---|---|---|
| 🌾 AnnualCrop | Agricultural Land | 🟡 Moderate |
| 🌲 Forest | Healthy Vegetation | 🟢 Low |
| 🌿 HerbaceousVegetation | Healthy Vegetation | 🟢 Low |
| 🛣️ Highway | Urban / Built-up | ⚪ Neutral |
| 🏭 Industrial | Urban / Built-up | ⚪ Neutral |
| 🐄 Pasture | Healthy Vegetation | 🟢 Low |
| 🍇 PermanentCrop | Agricultural Land | 🟡 Moderate |
| 🏘️ Residential | Urban / Built-up | ⚪ Neutral |
| 🏞️ River | Water Body | 🔵 Positive |
| 🌊 SeaLake | Water Body | 🔵 Positive |

---

## Model Architecture

Despite the notebook title referencing EfficientNetB0, the final trained model is a **custom 3-block CNN** built from scratch using TensorFlow/Keras.

```
Input (64×64×3)
    │
    ├── Rescaling(1/255)
    │
    ├── Block 1: Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
    ├── Block 2: Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
    ├── Block 3: Conv2D(128) → BN → Conv2D(128) → MaxPool → Dropout(0.35)
    │
    ├── GlobalAveragePooling2D
    ├── Dense(256, relu)
    ├── Dropout(0.5)
    └── Dense(10, softmax)
```

**Training config:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping | patience=5 on val_accuracy |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Best model checkpoint | saved on val_accuracy |

**Data split:** 70% train / 15% validation / 15% test (stratified)

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **~95%** |
| Test AUC | **~0.998** |

---

## Project Structure

```
your_project/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── drought_detection_final.ipynb   # Training notebook (Google Colab)
└── drought_model_artifacts/
    ├── drought_model.keras         # Trained model (~3.8 MB)
    ├── drought_model.h5            # Legacy HDF5 backup
    └── metadata.json               # Class names, accuracy, config
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/satellite-drought-detector.git
cd satellite-drought-detector
pip install -r requirements.txt
```

### 2. Train the model (Google Colab)

Open `drought_detection_final.ipynb` in Google Colab.

1. Upload `EuroSAT.zip` to `/content/` (download from [Kaggle — nilesh992/eurosat-dataset](https://www.kaggle.com/datasets/nilesh992/eurosat-dataset))
2. Run all cells in order
3. Download the artifacts from the final cell:
   - `drought_model_artifacts/drought_model.keras`
   - `drought_model_artifacts/metadata.json`
4. Place them into `drought_model_artifacts/` next to `app.py`

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Requirements

```
tensorflow>=2.16.0
streamlit
numpy
Pillow
matplotlib
scikit-learn
seaborn
```

> **Note on TF versions:** The model was saved in Google Colab (TF 2.16+). If you get a `batch_shape` deserialization error locally, run `pip install --upgrade tensorflow` and restart Streamlit. The app includes automatic fallback loading strategies to handle minor version mismatches.

---

## How the Prediction Works

```python
# 1. Load and resize the image
img = pil_img.convert('RGB').resize((64, 64))

# 2. Apply EfficientNet-style preprocessing (scales to [-1, 1])
arr = tf.keras.applications.efficientnet.preprocess_input(np.array(img, dtype=np.float32))

# 3. Run inference
probs = model.predict(np.expand_dims(arr, 0))[0]

# 4. Map to drought label
predicted_class = CLASS_NAMES[np.argmax(probs)]
drought_label   = DROUGHT_INTERPRETATION[predicted_class]
```

> ⚠️ **Preprocessing note:** The training pipeline used `efficientnet.preprocess_input()` which normalises to `[-1, 1]`. The model architecture also contains a `Rescaling(1/255)` layer, but this is effectively overridden by the EfficientNet preprocessing applied before input. The app replicates the exact training preprocessing.

---

## Acknowledgements

- **EuroSAT Dataset** — Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.*
- **Sentinel-2 imagery** — European Space Agency (ESA) Copernicus programme
- **Drought severity context** — Chuphal et al. (2024), Drought Atlas of India
- **GreenAI / Skill4Future** — Module 5: Computer Vision for Green Technology

---

## License

This project is part of the GreenAI Skill4Future coursework. Dataset usage is subject to the EuroSAT dataset licence.
