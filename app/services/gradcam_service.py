import os
import uuid
import logging
import numpy as np
import cv2
import tensorflow as tf
from app.config import Config

logger = logging.getLogger(__name__)


def generate_gradcam(model, processed_image, original_img, save_filename, prediction_label="Tumor"):
    """
    Generates:
    1. Grad-CAM heatmap overlay image
    2. Explicit tumor region contour and bounding box highlight image (or normal scan badge)
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

        last_conv_layer = model.layers[last_conv_idx]

        # Construct sub-model to get activations from input up to last Conv2D layer
        conv_outputs_model = tf.keras.Model(inputs=model.inputs, outputs=last_conv_layer.output)

        # Construct sub-model from post-Conv2D layers to pre-sigmoid raw logit
        conv_shape = last_conv_layer.output.shape[1:]
        conv_input = tf.keras.Input(shape=conv_shape)
        x = conv_input
        final_dense_layer = model.layers[-1]
        w, b = final_dense_layer.get_weights()

        unique_dense_name = f"gradcam_dense_{uuid.uuid4().hex[:8]}"

        for layer in model.layers[last_conv_idx + 1:]:
            if layer == final_dense_layer:
                # Omit sigmoid activation to get linear raw logit (avoids vanishing gradients)
                x = tf.keras.layers.Dense(1, activation=None, name=unique_dense_name)(x)
            else:
                x = layer(x)

        sub_model = tf.keras.Model(inputs=conv_input, outputs=x)
        sub_model.get_layer(unique_dense_name).set_weights([w, b])

        img_tensor = tf.cast(processed_image, tf.float32)
        conv_outs = conv_outputs_model(img_tensor)

        with tf.GradientTape() as tape:
            tape.watch(conv_outs)
            logits = sub_model(conv_outs)
            loss = logits[:, 0]

        grads = tape.gradient(loss, conv_outs)
        if grads is None:
            logger.warning("Gradients evaluated to None during Grad-CAM backprop.")
            return None, None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU activation and normalize heatmap to [0, 1]
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        # Prepare original image array
        orig_np = np.array(original_img)
        h, w_img = orig_np.shape[:2]

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_np, (w_img, h))
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

        is_tumor = (str(prediction_label).strip().lower() == "tumor")

        if is_tumor and max_val > 0:
            # Otsu thresholding + Morphological closing to group activation regions
            _, thresh = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                primary_contours = [c for c in contours if cv2.contourArea(c) > (w_img * h * 0.001)]
                if not primary_contours:
                    primary_contours = [contours[0]]

                # Draw translucent red fill over detected tumor region
                mask = np.zeros_like(highlight_bgr)
                cv2.drawContours(mask, primary_contours, -1, (0, 0, 255), -1)
                highlight_bgr = cv2.addWeighted(highlight_bgr, 0.85, mask, 0.35, 0)

                # Draw yellow boundary contour lines
                cv2.drawContours(highlight_bgr, primary_contours, -1, (0, 255, 255), 2)

                # Draw bright red bounding box around primary tumor area
                all_pts = np.concatenate(primary_contours)
                x_b, y_b, w_b, h_b = cv2.boundingRect(all_pts)
                cv2.rectangle(highlight_bgr, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 0, 255), 3)

                # Add text label badge above bounding box
                label_text = "TUMOR REGION"
                label_y = max(y_b - 10, 25)
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(
                    highlight_bgr,
                    (x_b, label_y - text_size[1] - 6),
                    (x_b + text_size[0] + 10, label_y + 4),
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    highlight_bgr,
                    label_text,
                    (x_b + 5, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            # Draw green badge indicating clean scan
            status_text = "NO TUMOR REGION DETECTED"
            cv2.rectangle(highlight_bgr, (10, 10), (320, 42), (0, 180, 0), -1)
            cv2.putText(
                highlight_bgr,
                status_text,
                (18, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        highlight_path = os.path.join(Config.HEATMAP_FOLDER, f"highlight_{save_filename}")
        cv2.imwrite(highlight_path, highlight_bgr)

        return f"heatmaps/heatmap_{save_filename}", f"heatmaps/highlight_{save_filename}"

    except Exception as e:
        logger.error(f"Grad-CAM generation failed with error: {str(e)}", exc_info=True)
        return None, None

