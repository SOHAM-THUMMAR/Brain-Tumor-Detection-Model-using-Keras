import os
from app.config import Config


def is_allowed_file(filename):
    """Checks if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def is_valid_file_size(file_path):
    """Checks if file size is within maximum limit."""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return file_size_mb <= Config.MAX_FILE_SIZE_MB
