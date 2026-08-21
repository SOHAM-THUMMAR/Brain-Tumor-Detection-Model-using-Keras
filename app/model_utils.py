"""
Facade module maintaining backward compatibility for model utilities.
Delegates calls to modular services in app/services/.
"""
from app.services.model_service import load_model, predict
from app.services.image_service import preprocess_image
from app.services.gradcam_service import generate_gradcam

__all__ = ["load_model", "predict", "preprocess_image", "generate_gradcam"]
