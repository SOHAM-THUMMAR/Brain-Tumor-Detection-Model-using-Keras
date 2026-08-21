import os
import uuid
import logging
import numpy as np
import cv2
import tensorflow as tf
from app.config import Config

logger = logging.getLogger(__name__)


import os
import uuid
import logging
import numpy as np
import cv2
def generate_gradcam(model, processed_image, original_img, save_filename, prediction_label="Tumor"):
    """
    Generates:
    1. Model-based Grad-CAM heatmap overlay image (heatmap_*.jpg) directly from last Conv2D layer
    2. Medical HUD tumor focus region contour & bounding box highlight image (highlight_*.jpg)
    
    Returns (heatmap_relative_path, highlight_relative_path).
    If generation fails, logs error and returns (None, None) gracefully.
    """
    try:
        # Find the last Conv2D layer in the Keras model (conv2d_7)
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            logger.warning("No Conv2D layer found in model for Grad-CAM generation.")
            return None, None

        # Extract weights from the final Dense classification layer (dense_3)
        dense_layer = model.layers[-1]
        dense_w, dense_b = dense_layer.get_weights()

        # Build functional model mapping input tensor to [last_conv_output, penultimate_output]
        inp = tf.keras.Input(shape=processed_image.shape[1:])
        x = inp
        conv_output_tensor = None

        for layer in model.layers:
            if layer == dense_layer:
                break
            x = layer(x)
            if layer == last_conv_layer:
                conv_output_tensor = x

        penultimate_output_tensor = x

        grad_model = tf.keras.Model(inputs=inp, outputs=[conv_output_tensor, penultimate_output_tensor])
        img_tensor = tf.cast(processed_image, tf.float32)

        # Evaluate gradients of linear raw logit w.r.t last Conv2D activations
        with tf.GradientTape() as tape:
            conv_eval, pen_eval = grad_model(img_tensor)
            logit = tf.matmul(pen_eval, dense_w) + dense_b
            loss = logit[:, 0]

        grads = tape.gradient(loss, conv_eval)
        if grads is None:
            logger.warning("Gradients evaluated to None during Grad-CAM backpropagation.")
            return None, None

        # Standard Grad-CAM: channel-pooled gradients * feature activation map
        pooled_grads = tf.reduce_mean(grads[0], axis=(0, 1))
        cam = conv_eval[0] @ pooled_grads[..., tf.newaxis]
        cam = tf.squeeze(cam)
        cam = tf.maximum(cam, 0)

        # Normalize CAM to [0, 1] range
        max_cam = tf.math.reduce_max(cam)
        if max_cam > 0:
            cam_norm = (cam / max_cam).numpy()
        else:
            cam_norm = cam.numpy()

        orig_np = np.array(original_img)
        h_orig, w_orig = orig_np.shape[:2]

        # Resize CAM to match original image dimensions & smooth with Gaussian Blur
        cam_resized = cv2.resize(cam_norm, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        cam_blurred = cv2.GaussianBlur(cam_resized, (15, 15), 0)
        cam_uint8 = np.uint8(255 * cam_blurred)

        # Skull/head tissue masking (isolates head tissue from pure black background)
        gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
        _, head_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        head_contours, _ = cv2.findContours(head_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        head_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        if head_contours:
            largest_head = max(head_contours, key=cv2.contourArea)
            cv2.drawContours(head_mask, [largest_head], -1, 255, -1)

        cam_head = cv2.bitwise_and(cam_uint8, cam_uint8, mask=head_mask)
        max_focus = cam_head.max()

        os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

        # -------------------------------------------------------------
        # 1. HEATMAP OVERLAY IMAGE (heatmap_*.jpg)
        # -------------------------------------------------------------
        heatmap_colored = cv2.applyColorMap(cam_head, cv2.COLORMAP_JET)
        blend_alpha = (cam_head / 255.0) * 0.60
        blend_alpha_3ch = cv2.merge([blend_alpha] * 3)

        heatmap_overlay = (orig_bgr * (1.0 - blend_alpha_3ch) + heatmap_colored * blend_alpha_3ch).astype(np.uint8)
        heatmap_path = os.path.join(Config.HEATMAP_FOLDER, f"heatmap_{save_filename}")
        cv2.imwrite(heatmap_path, heatmap_overlay)

        # -------------------------------------------------------------
        # 2. MEDICAL HUD HIGHLIGHT IMAGE (highlight_*.jpg)
        # -------------------------------------------------------------
        highlight_bgr = orig_bgr.copy()
        is_tumor = (str(prediction_label).strip().lower() == "tumor")

        if is_tumor and max_focus > 30:
            # Threshold at top 50% of the model's peak focus activation
            thresh_val = int(0.50 * max_focus)
            _, thresh = cv2.threshold(cam_head, thresh_val, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Select primary contour representing top model focus area
                primary_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

                # Semi-transparent Crimson Red fill inside ROI
                red_mask = np.zeros_like(highlight_bgr)
                cv2.drawContours(red_mask, primary_contours, -1, (30, 30, 240), -1)
                highlight_bgr = cv2.addWeighted(highlight_bgr, 0.78, red_mask, 0.42, 0)

                # Glowing Cyan contour boundary
                cv2.drawContours(highlight_bgr, primary_contours, -1, (255, 235, 0), 2, cv2.LINE_AA)

                # Bounding box around primary model activation area
                x_b, y_b, w_b, h_b = cv2.boundingRect(primary_contours[0])
                pad = 8
                x1, y1 = max(0, x_b - pad), max(0, y_b - pad)
                x2, y2 = min(w_orig, x_b + w_b + pad), min(h_orig, y_b + h_b + pad)

                # Outer HUD Box
                cv2.rectangle(highlight_bgr, (x1, y1), (x2, y2), (0, 50, 240), 2, cv2.LINE_AA)

                # HUD Corner Brackets (Neon Yellow)
                c_len = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
                corner_col = (0, 255, 255)
                ct = 3

                # Top-Left
                cv2.line(highlight_bgr, (x1, y1), (x1 + c_len, y1), corner_col, ct)
                cv2.line(highlight_bgr, (x1, y1), (x1, y1 + c_len), corner_col, ct)
                # Top-Right
                cv2.line(highlight_bgr, (x2, y1), (x2 - c_len, y1), corner_col, ct)
                cv2.line(highlight_bgr, (x2, y1), (x2, y1 + c_len), corner_col, ct)
                # Bottom-Left
                cv2.line(highlight_bgr, (x1, y2), (x1 + c_len, y2), corner_col, ct)
                cv2.line(highlight_bgr, (x1, y2), (x1, y2 - c_len), corner_col, ct)
                # Bottom-Right
                cv2.line(highlight_bgr, (x2, y2), (x2 - c_len, y2), corner_col, ct)
                cv2.line(highlight_bgr, (x2, y2), (x2, y2 - c_len), corner_col, ct)

                # Target Crosshair (+) at centroid of focus region
                M = cv2.moments(primary_contours[0])
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                else:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                ret_size = 7
                cv2.line(highlight_bgr, (cx - ret_size, cy), (cx + ret_size, cy), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(highlight_bgr, (cx, cy - ret_size), (cx, cy + ret_size), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(highlight_bgr, (cx, cy), 3, (0, 0, 255), -1)

                # Sleek Badge Label above box
                badge_text = "TUMOR FOCUS REGION (MODEL ATN)"
                badge_y = max(y1 - 10, 22)
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                cv2.rectangle(highlight_bgr, (x1, badge_y - th - 6), (x1 + tw + 14, badge_y + 4), (15, 15, 25), -1)
                cv2.rectangle(highlight_bgr, (x1, badge_y - th - 6), (x1 + tw + 14, badge_y + 4), (0, 50, 240), 1)
                cv2.putText(highlight_bgr, badge_text, (x1 + 7, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Clean scan badge
            status_text = "NO TUMOR REGION DETECTED"
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(highlight_bgr, (12, 12), (18 + tw + 12, 12 + th + 14), (15, 25, 15), -1)
            cv2.rectangle(highlight_bgr, (12, 12), (18 + tw + 12, 12 + th + 14), (0, 180, 0), 1)
            cv2.putText(highlight_bgr, status_text, (18, 12 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 100), 1, cv2.LINE_AA)

        highlight_path = os.path.join(Config.HEATMAP_FOLDER, f"highlight_{save_filename}")
        cv2.imwrite(highlight_path, highlight_bgr)

        return f"heatmaps/heatmap_{save_filename}", f"heatmaps/highlight_{save_filename}"

    except Exception as e:
        logger.error(f"Grad-CAM generation failed with error: {str(e)}", exc_info=True)
        return None, None
