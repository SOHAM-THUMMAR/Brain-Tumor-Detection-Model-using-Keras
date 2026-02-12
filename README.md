# 🧠 Brain Tumor Detection Model using Keras

A deep learning project that detects the presence of a brain tumor from MRI images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

This project demonstrates end-to-end image classification — from dataset preparation and preprocessing to training, evaluation, and model saving.

---

## 📌 Project Objective

The goal of this project is to classify brain MRI images into two categories:

- Tumor
- No Tumor

This implementation focuses on clarity, simplicity, and clean structure while maintaining good model performance.

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
├── splitter.py               # Script to split dataset into train/val/test
├── brain tumor.ipynb                  # Model building, training & evaluation
│
├── bestModel.keras           # Saved best trained model
├── bestModel.h5              # Saved model (alternate format)
│
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Technologies Used

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- seaborn
- Pillow (PIL)

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SOHAM-THUMMAR/Brain-Tumor-Detection-Model-using-Keras.git
cd Brain-Tumor-Detection-Model-using-Keras
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a requirements file, install manually:

```bash
pip install tensorflow numpy matplotlib pillow
```

---

## 🧪 Dataset Preparation

Organize your dataset into two folders:

```
data/
│
├── tumor/
└── no_tumor/
```

Each folder should contain MRI images corresponding to its class.

---

## 🔀 Splitting the Dataset

To automatically split the dataset into training, validation, and testing sets:

```bash
python splitter.py
```

This script creates:

- train/
- val/
- test/

folders with proper class distribution.

---

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) designed for binary classification.

Typical architecture flow:

- Convolution Layer
- ReLU Activation
- MaxPooling
- Dropout (to reduce overfitting)
- Fully Connected (Dense) Layers
- Sigmoid Output Layer (Binary Classification)

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

Evaluation Metric:
- Accuracy

---

## 📈 Training the Model

Run the training notebook:

```bash
brain tumor.ipynb
```

During training:

- Images are loaded and normalized
- CNN model is built
- Model trains on training data
- Validation performance is monitored
- Best model is saved automatically

Saved Model Files:
- bestModel.keras
- bestModel.h5

---

## 📊 Model Evaluation

### Training vs Validation Recall

<p align="center">
  <img src="./graphs/Training vs Validation Recall.png" width="600">
</p>

The model shows steady improvement in recall across epochs.  
Validation recall closely follows training recall, indicating good generalization and minimal overfitting.

---

### Training vs Validation Loss

<p align="center">
  <img src="./graphs/Training vs Validation Loss.png" width="600">
</p>

Both training and validation loss decrease consistently over epochs.  
No significant divergence is observed, demonstrating stable learning behavior.

---

### Confusion Matrix

<p align="center">
  <img src="./graphs/Confusion Matrix.png" width="500">
</p>

Confusion Matrix:

```
[[195  13]
 [  0 458]]
```

- False Positives: 13  
- False Negatives: 0  
- Tumor Recall: 100%  

The model achieved **zero false negatives**, meaning no tumor cases were missed.

---

### ROC Curve

<p align="center">
  <img src="./graphs/ROC Curve.png" width="600">
</p>

- **AUC Score: 1.000**

The ROC curve demonstrates near-perfect class separability on the test dataset.

---

## 🔍 Using the Saved Model (Inference)

Example of loading the trained model:

```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("bestModel.keras")

prediction = model.predict(image_array)
```

If prediction > 0.5 → Tumor  
Else → No Tumor  

---

## ✨ Features

- Clean project structure
- Dataset splitting automation
- Model checkpoint saving
- Easy to extend
- Beginner-friendly implementation

---

## 🧩 Possible Improvements

This project can be enhanced by:

- Adding data augmentation
- Using Transfer Learning (MobileNet, ResNet, EfficientNet)
- Adding confusion matrix visualization
- Adding precision, recall, and F1-score
- Creating a web app interface (Streamlit / Flask)
- Extending to multi-class tumor classification

---

## 📌 Limitations

- Binary classification only
- Performance depends on dataset size
- Not suitable for real clinical deployment without validation

---

## 📄 License

This project is open-source and available for learning and research purposes.

---

## 🙌 Author

Soham Thummar

If you found this project helpful, consider giving it a ⭐ on GitHub.
