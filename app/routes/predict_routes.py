import os
import uuid
import logging
from flask import Blueprint, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from app.config import Config
from app.services.validation_service import is_allowed_file, is_valid_file_size
from app.services.image_service import preprocess_image
from app.services.model_service import load_model, predict
from app.services.gradcam_service import generate_gradcam

logger = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["GET", "POST"])
def handle_predict():
    # If accessed directly via GET (e.g., browser refresh), redirect to home upload page
    if request.method == "GET":
        return redirect(url_for("main.index"))

    try:
        if "file" not in request.files:
            return (
                render_template(
                    "error.html",
                    error_title="Bad Request",
                    error_message="No file part in the upload request. Please select a valid MRI image.",
                ),
                400,
            )

        file = request.files["file"]
        if file.filename == "":
            return (
                render_template(
                    "error.html",
                    error_title="Bad Request",
                    error_message="No file selected. Please select an MRI scan image to analyze.",
                ),
                400,
            )

        if not is_allowed_file(file.filename):
            return (
                render_template(
                    "error.html",
                    error_title="Invalid File Type",
                    error_message=f"File format not supported. Please upload an image file ({', '.join(Config.ALLOWED_EXTENSIONS)}).",
                ),
                400,
            )

        # Generate safe unique filename
        filename = secure_filename(file.filename)
        unique_prefix = uuid.uuid4().hex[:8]
        saved_filename = f"{unique_prefix}_{filename}"
        upload_path = os.path.join(Config.UPLOAD_FOLDER, saved_filename)

        file.save(upload_path)

        # Check file size after saving
        if not is_valid_file_size(upload_path):
            os.remove(upload_path)
            return (
                render_template(
                    "error.html",
                    error_title="File Too Large",
                    error_message=f"Uploaded file exceeds maximum limit of {Config.MAX_FILE_SIZE_MB}MB.",
                ),
                400,
            )

        # Load model instance
        model = load_model()

        # Preprocess image
        processed_img, orig_pil_img = preprocess_image(upload_path)

        # Run prediction
        label, confidence = predict(model, processed_img)

        # Generate Grad-CAM heatmap
        heatmap_filename = f"heatmap_{saved_filename}"
        heatmap_relative_path = generate_gradcam(
            model, processed_img, orig_pil_img, heatmap_filename
        )

        original_image_url = f"uploads/{saved_filename}"

        return render_template(
            "result.html",
            prediction=label,
            confidence=confidence,
            original_image_url=original_image_url,
            heatmap_image_url=heatmap_relative_path,
        )

    except Exception as e:
        logger.error(f"Internal server error during prediction: {str(e)}", exc_info=True)
        return (
            render_template(
                "error.html",
                error_title="Internal Processing Error",
                error_message="An unexpected error occurred while analyzing the image. Please try again with a valid brain MRI scan.",
            ),
            500,
        )
