# Brain Tumor Detection Model using Keras

Brain Tumor Detection Model using Keras is a Python project that implements a deep learning model to classify brain MRI images into tumor and non-tumor categories. The model is developed using Keras (a TensorFlow high-level API) and trained on a dataset of brain MRI images. The repository contains code for training and testing the model, preparing the dataset, and saving the best performing model for later use.

---

## Features

- Train a deep learning model to detect brain tumors from MRI scans  
- Uses Keras and TensorFlow for neural network implementation  
- Includes data preparation and splitting scripts  
- Saves the best performing models (`bestModel.h5`, `bestModel.keras`)  
- `main.py` for training and testing the model  
- `splitter.py` for organizing data into train/test splits  

---

## Tech Stack

- Python  
- Keras (TensorFlow backend)  
- NumPy and other scientific libraries  
- Deep learning with Convolutional Neural Networks  
- Model saving and evaluation using best weights  

---

## Project Structure

```
/
├── data/                    # Dataset folder for MRI images
├── bestModel.h5            # Saved Keras model
├── bestModel.keras         # Another saved model format
├── main.py                 # Model training and evaluation script
├── splitter.py             # Script to split dataset into train/test
└── other necessary files   # Any additional helper modules
```

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x  
- TensorFlow and Keras  
- NumPy, Pandas  
- Matplotlib (optional for visualization)  

Install dependencies:
```bash
pip install tensorflow keras numpy pandas
```

### Prepare Dataset

1. Place MRI images into the `data/` folder.  
2. Use `splitter.py` to divide the dataset into training and testing sets:
```bash
python splitter.py
```

### Train and Evaluate Model

Run the main training and testing script:
```bash
python main.py
```

During training, the model learns to classify MRI images as tumor or non-tumor. After training, the best performing model is saved as `bestModel.h5` (and optionally in the `.keras` format).

---

## How It Works

1. The data is loaded from the `data/` folder and split into train and test sets.  
2. Images are processed and resized for input to the model.  
3. A Convolutional Neural Network (CNN) model is defined using Keras layers.  
4. The model is trained using labeled brain MRI images.  
5. After training, the best model weights are saved for future prediction.  

Related deep learning concepts include using convolutional layers to extract features from MRI images and using fully connected layers for classification. CNNs are widely used for image classification tasks, including medical image analysis. :contentReference[oaicite:0]{index=0}

---

## Future Enhancements

- Add support for multi-class classification (glioma, meningioma, pituitary, etc.)  
- Include data augmentation for better generalization  
- Build a web interface to upload MRI images and show predictions  
- Use pre-trained architectures with transfer learning for higher accuracy  
- Visualize training metrics and confusion matrix  

---

## Contributing

To contribute:

1. Fork the repository  
2. Create a new branch:
```bash
git checkout -b feature-name
```
3. Make changes and commit:
```bash
git commit -m "Add new feature"
```
4. Push and open a pull request
