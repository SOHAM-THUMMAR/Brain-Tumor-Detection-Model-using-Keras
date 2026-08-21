import os
import logging
import numpy as np
import cv2
import tensorflow as tf
from app.config import Config

logger = logging.getLogger(__name__)


def generate_gradcam(model, processed_image, original_img, save_filename):
    """
    Generates a Grad-CAM heatmap overlay for the uploaded image using tf.GradientTape
    by computing activations and gradients through sequential layer forward passes.
    Compatible with Keras 3.x and Keras 2.x.
    """
    try:
        # Find last Conv2D layer index
        last_conv_idx = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_idx = i

        if last_conv_idx is None:
            logger.warning("No Conv2D layer found in model for Grad-CAM generation.")
            return None

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
            return None

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
        heatmap = heatmap.numpy()

        # Prepare original image array
        orig_np = np.array(original_img)
        h, w = orig_np.shape[:2]

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Convert original RGB image to BGR for OpenCV blending
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

        # Overlay heatmap on original image (60% scan, 40% heatmap)
        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)

        # Save heatmap to static heatmaps directory
        os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)
        heatmap_path = os.path.join(Config.HEATMAP_FOLDER, save_filename)
        cv2.imwrite(heatmap_path, overlay)

        return f"heatmaps/{save_filename}"
    except Exception as e:
        logger.error(f"Grad-CAM generation failed with error: {str(e)}", exc_info=True)
        return None
