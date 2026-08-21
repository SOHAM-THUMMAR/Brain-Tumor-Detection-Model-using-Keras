# 🧠 Brain Tumor Detection using CNN (TensorFlow / Keras)

A deep learning project for detecting the presence of brain tumors from MRI images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

This project implements a complete deep learning pipeline including data preparation, model training, evaluation, performance visualization, and a web deployment interface with Grad-CAM heatmaps.

---

## 🌐 Web App

The project includes an interactive web interface powered by **Flask** that allows users to upload brain MRI scans, perform real-time CNN inference, and visualize model attention using **Grad-CAM heatmaps**.

### Web App Features
- **MRI Classification:** Fast inference yielding prediction label ("Tumor" or "No Tumor") and confidence score.
- **Grad-CAM Visualization:** Highlights the region of interest in the MRI scan driving model output.
- **Performance Dashboard:** Displays key test evaluation metrics and graphs directly in the web UI.
- **Drag-and-Drop Interface:** Responsive UI with client-side file format validation.

### Web App Directory Structure
```
app/
├── app.py               # Flask application entrypoint & routes
├── model_utils.py       # Model loading, preprocessing, inference & Grad-CAM logic
├── config.py            # App configurations and threshold parameters
├── templates/           # Jinja2 HTML templates
│   ├── base.html        # Shared layout, nav & disclaimer footer
│   ├── index.html       # Drag-and-drop upload form
│   ├── result.html      # Classification badge, confidence & Grad-CAM overlay
│   ├── stats.html       # Performance metrics & graphs
│   └── error.html       # User-friendly error handler
└── static/              # CSS, JS, uploads & generated heatmaps
```

### Running the Web App Locally

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the Flask server:
```bash
python app/app.py
```

4. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---


## 📌 Project Objective

The goal of this project is to classify brain MRI images into two categories:

- **Tumor**
- **No Tumor**

The model is designed to achieve high recall for tumor detection while maintaining strong overall classification performance.

---

## 🏆 Key Results

- **Test Accuracy:** 98%
- **Tumor Recall:** 100%
- **False Negatives:** 0
- **AUC Score:** 1.000
- **Total Test Samples:** 666

The model successfully detected all tumor cases in the test dataset without missing any positive instances.

---

## 📂 Repository Structure

```
Brain-Tumor-Detection-Model-using-Keras/
│
├── data/                     # Original dataset (tumor / no_tumor)
├── train/                    # Training images
├── val/                      # Validation images
├── test/                     # Testing images
│
├── graphs/                   # Evaluation graphs
│   ├── training_validation_recall.png
│   ├── training_validation_loss.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── splitter.py               # Dataset splitting script
├── brain_tumor.ipynb         # Model training & evaluation notebook
│
├── bestModel.keras           # Saved trained model
├── bestModel.h5              # Alternate saved format
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

- Binary classification dataset (Tumor vs No Tumor)
- Test set size: 666 images
  - Tumor: 458
  - No Tumor: 208

> Note: Performance may vary depending on dataset size and quality.

---

## ⚙️ Technologies Used

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Pillow (PIL)

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/SOHAM-THUMMAR/Brain-Tumor-Detection-Model-using-Keras.git
cd Brain-Tumor-Detection-Model-using-Keras
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install tensorflow numpy matplotlib seaborn pillow
```

---

## 🧪 Dataset Preparation

Organize your dataset as follows:

```
data/
│
├── tumor/
└── no_tumor/
```

Each folder should contain corresponding MRI images.

---

## 🔀 Splitting the Dataset

Run:

```bash
python splitter.py
```

This will create:

- train/
- val/
- test/

with proper class distribution.

---

## 🧠 Model Architecture

The model is a custom Convolutional Neural Network (CNN) designed for binary classification.

### Architecture Overview

- Convolutional Layers (16 → 32 → 64 → 128 filters)
- ReLU Activation
- MaxPooling Layers
- Dropout for regularization
- Flatten Layer
- Dense Layer (64 units)
- Sigmoid Output Layer

### Model Statistics

- **Total Parameters:** 5,635,361  
- **Trainable Parameters:** 5,635,361  
- **Non-Trainable Parameters:** 0  

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 📈 Training the Model

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
brain_tumor.ipynb
```

The notebook includes:

- Data loading & preprocessing
- Model building
- Training loop
- Evaluation metrics
- Visualization of performance curves
- Model saving

---

## 📊 Model Evaluation

### Training vs Validation Recall

<p align="center">
  <img src="graphs/Training vs Validation Recall.png" width="600">
</p>

The model shows steady improvement in recall across epochs.  
Validation recall closely follows training recall, indicating good generalization and minimal overfitting.

---

### Training vs Validation Loss

<p align="center">
  <img src="graphs/Training vs Validation Loss.png" width="600">
</p>

Both training and validation loss decrease consistently over epochs, demonstrating stable learning behavior.

---

### Confusion Matrix

<p align="center">
  <img src="graphs/Confusion Matrix.png" width="500">
</p>

Confusion Matrix:

```
[[195  13]
 [  0 458]]
```

- False Positives: 13  
- False Negatives: 0  
- Tumor Recall: 100%  

The model achieved zero false negatives, meaning no tumor cases were missed.

---

### ROC Curve

<p align="center">
  <img src="graphs/ROC Curve.png" width="600">
</p>

- **AUC Score: 1.000**

The ROC curve indicates near-perfect class separability on the test dataset.

> Note: A perfect AUC score suggests strong performance on this dataset. Further validation on external or larger datasets is recommended to confirm generalization capability.

---

## 🔍 Using the Saved Model (Inference)

Example:

```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("bestModel.keras")

prediction = model.predict(image_array)

if prediction > 0.5:
    print("Tumor Detected")
else:
    print("No Tumor Detected")
```

---

## 🔁 Reproducibility Steps

1. Clone repository  
2. Install dependencies  
3. Prepare dataset  
4. Run splitter.py  
5. Launch brain_tumor.ipynb  
6. Train model  

---

## ✨ Features

- Clean modular project structure
- Automated dataset splitting
- Model checkpoint saving
- Multiple evaluation metrics
- Performance visualization
- Beginner-friendly yet technically structured implementation

---

## 🧩 Possible Improvements

- Add Data Augmentation
- Implement Transfer Learning (ResNet, EfficientNet, MobileNet)
- Replace Flatten with GlobalAveragePooling2D to reduce parameters
- Add Grad-CAM visualization for interpretability
- Deploy using Streamlit or Flask
- Extend to multi-class tumor classification

---

## 📌 Limitations

- Binary classification only
- Performance dependent on dataset size and quality
- Not suitable for real clinical deployment without medical validation

---

## 📄 License

This project is open-source and intended for research and educational purposes.

---

## 🙌 Author

**Soham Thummar**

If you found this project helpful, consider giving it a ⭐ on GitHub.
