import os
import logging
import numpy as np
import cv2
import tensorflow as tf
from app.config import Config

logger = logging.getLogger(__name__)


def generate_gradcam(model, processed_image, original_img, save_filename):
    """
    Generates:
    1. Grad-CAM heatmap overlay image
    2. Explicit tumor region contour and bounding box highlight image
    Returns (heatmap_relative_path, highlight_relative_path).
    If generation fails, catches exception and returns (None, None) gracefully.
    """
    try:
        # Find last Conv2D layer index
        last_conv_idx = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_idx = i

        if last_conv_idx is None:
            logger.warning("No Conv2D layer found in model for Grad-CAM generation.")
            return None, None

        img_tensor = tf.cast(processed_image, tf.float32)

        with tf.GradientTape() as tape:
            x = img_tensor
            conv_outputs = None
            for i, layer in enumerate(model.layers):
                x = layer(x)
                if i == last_conv_idx:
                    conv_outputs = x
                    tape.watch(conv_outputs)
            predictions = x
            loss = predictions[:, 0]

        # Compute gradients of output prediction w.r.t last conv layer activation
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            logger.warning("Gradients evaluated to None during Grad-CAM backprop.")
            return None, None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight conv output channels by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU activation and normalize heatmap to [0, 1]
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        # Prepare original image array
        orig_np = np.array(original_img)
        h, w = orig_np.shape[:2]

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_np, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # -------------------------------------------------------------
        # 1. HEATMAP OVERLAY IMAGE
        # -------------------------------------------------------------
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

        # Overlay heatmap on original image (60% scan, 40% heatmap)
        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)

        os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)
        heatmap_path = os.path.join(Config.HEATMAP_FOLDER, f"heatmap_{save_filename}")
        cv2.imwrite(heatmap_path, overlay)

        # -------------------------------------------------------------
        # 2. TUMOR BOUNDING BOX & CONTOUR HIGHLIGHT IMAGE
        # -------------------------------------------------------------
        highlight_bgr = orig_bgr.copy()

        # Threshold heatmap at 40% max intensity to isolate activation region
        _, thresh = cv2.threshold(heatmap_uint8, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_tumor_region = False
        for c in contours:
            area = cv2.contourArea(c)
            # Filter out tiny noisy points
            if area > (w * h * 0.005):
                found_tumor_region = True
                x_b, y_b, w_b, h_b = cv2.boundingRect(c)

                # Draw yellow exact contour boundary
                cv2.drawContours(highlight_bgr, [c], -1, (0, 255, 255), 2)

                # Draw bright red bounding box around tumor region
                cv2.rectangle(highlight_bgr, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 0, 255), 3)

                # Add text label above bounding box
                label_text = "TUMOR REGION"
                label_y = max(y_b - 10, 20)
                # Background text fill
                cv2.rectangle(highlight_bgr, (x_b, label_y - 18), (x_b + 140, label_y + 4), (0, 0, 255), -1)
                cv2.putText(
                    highlight_bgr,
                    label_text,
                    (x_b + 5, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if not found_tumor_region and len(contours) > 0:
            # Fallback to largest contour if no contour met area threshold
            c = max(contours, key=cv2.contourArea)
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            cv2.drawContours(highlight_bgr, [c], -1, (0, 255, 255), 2)
            cv2.rectangle(highlight_bgr, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 0, 255), 3)

        highlight_path = os.path.join(Config.HEATMAP_FOLDER, f"highlight_{save_filename}")
        cv2.imwrite(highlight_path, highlight_bgr)

        return f"heatmaps/heatmap_{save_filename}", f"heatmaps/highlight_{save_filename}"

    except Exception as e:
        logger.error(f"Grad-CAM generation failed with error: {str(e)}", exc_info=True)
        return None, None
