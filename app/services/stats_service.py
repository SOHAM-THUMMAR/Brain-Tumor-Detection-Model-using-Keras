import os
import shutil
from app.config import Config


def ensure_directories_and_graphs():
    """Ensures static upload directories and synchronizes evaluation graphs to static folder."""
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)
    os.makedirs(Config.STATIC_GRAPHS_FOLDER, exist_ok=True)

    if os.path.exists(Config.GRAPHS_FOLDER):
        for filename in os.listdir(Config.GRAPHS_FOLDER):
            if filename.endswith(".png"):
                src = os.path.join(Config.GRAPHS_FOLDER, filename)
                dst = os.path.join(Config.STATIC_GRAPHS_FOLDER, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)


def get_performance_metrics():
    """Returns static benchmark evaluation metrics dictionary for the test set."""
    return {
        "accuracy": "98%",
        "recall": "100%",
        "false_negatives": "0",
        "auc": "1.000",
        "total_test": 666,
        "tumor_count": 458,
        "no_tumor_count": 208,
        "confusion_matrix": [[195, 13], [0, 458]],
    }
