# Brain Tumor Detection Model using Keras

Brain-Tumor-Detection-Model-using-Keras is a Python-based deep learning project that detects brain tumors from MRI images using the Keras API with a TensorFlow backend. The project provides a complete pipeline from data preprocessing to model training and saving trained models for inference.

---

## 🚀 Features

- Deep learning-based brain tumor classification
- Built using **Keras** and **TensorFlow**
- Dataset preprocessing and splitting
- CNN-based model training
- Saves best performing model automatically
- Ready-to-use trained models for inference

---

## 📁 Project Structure

Brain-Tumor-Detection-Model-using-Keras/
│
├── data/ # MRI image dataset
├── bestModel.h5 # Trained model (HDF5 format)
├── bestModel.keras # Trained model (Keras format)
├── main.py # Model training and evaluation script
├── splitter.py # Dataset splitting script
└── README.md # Project documentation

yaml
Copy code

---

## 🧠 Requirements

Ensure Python 3.8 or higher is installed.

Install required dependencies:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
🛠️ Usage
1️⃣ Prepare Dataset
Place MRI brain images inside the data/ directory with proper class labels (e.g., tumor / no_tumor).

2️⃣ Split Dataset
bash
Copy code
python splitter.py
3️⃣ Train the Model
bash
Copy code
python main.py
4️⃣ Saved Models
After training, the best-performing model is saved as:

bestModel.h5

bestModel.keras

These models can be directly used for prediction or further training.

📊 Model Overview
Uses a Convolutional Neural Network (CNN)

Input: MRI brain scan images

Output: Tumor / No Tumor classification

Training includes validation and performance monitoring

🧪 Evaluation
Model performance can be evaluated using accuracy and loss metrics on the validation or test dataset. Further improvements can be made using:

Data augmentation

Transfer learning

Hyperparameter tuning

📌 Notes
Designed for educational and research purposes

Can be extended to multi-class tumor classification

Not intended for direct clinical diagnosis

📜 License
This project is open-source. You may add a license file to define usage and distribution rights.

🙌 Author
Developed by Soham Thummar
