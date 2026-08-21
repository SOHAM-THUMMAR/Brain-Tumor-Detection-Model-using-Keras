import os
import numpy as np
from PIL import Image
from app.config import Config


def preprocess_image(image_input):
    """
    Loads and preprocesses input image to conform to notebook specifications:
    - RGB color channels
    - Resized to Config.IMG_SIZE (224, 224)
    - Normalized pixel values in [0, 1] range (1/255.0 scaling)
    - Batched shape (1, 224, 224, 3)
    """
    if isinstance(image_input, (str, os.PathLike)):
        img = Image.open(image_input)
    else:
        img = Image.open(image_input)

    img = img.convert("RGB")
    img = img.resize(Config.IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    processed_image = np.expand_dims(img_array, axis=0)
    return processed_image, img
