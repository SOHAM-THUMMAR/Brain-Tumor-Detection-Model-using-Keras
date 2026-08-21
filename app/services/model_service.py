import os
import logging
import tensorflow as tf
from app.config import Config

logger = logging.getLogger(__name__)

# Singleton instance for loaded Keras model
_MODEL_INSTANCE = None


def load_model():
    """
    Loads bestModel.keras once at startup (module-level singleton).
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")
        logger.info(f"Loading Keras model from {Config.MODEL_PATH}...")
        _MODEL_INSTANCE = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
        logger.info("Keras model loaded successfully.")
    return _MODEL_INSTANCE


def predict(model, processed_image):
    """
    Executes model inference and applies the classification threshold (0.3).
    Returns (label: str, confidence: float 0-100).
    """
    raw_pred = model.predict(processed_image, verbose=0)
    prob = float(raw_pred[0][0])

    if prob >= Config.CLASSIFICATION_THRESHOLD:
        label = "Tumor"
        confidence = prob * 100.0
    else:
        label = "No Tumor"
        confidence = (1.0 - prob) * 100.0

    return label, round(confidence, 2)
