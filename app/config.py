import os


class Config:
    """Application configuration parameters."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "bestModel.keras")
    IMG_SIZE = (224, 224)
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_FILE_SIZE_MB = 10
    MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    HEATMAP_FOLDER = os.path.join(BASE_DIR, "app", "static", "heatmaps")
    REPORTS_FOLDER = os.path.join(BASE_DIR, "app", "static", "reports")
    GRAPHS_FOLDER = os.path.join(BASE_DIR, "graphs")
    STATIC_GRAPHS_FOLDER = os.path.join(BASE_DIR, "app", "static", "graphs")

    # Lower classification threshold tuned for high recall (medical safety margin)
    CLASSIFICATION_THRESHOLD = 0.3
