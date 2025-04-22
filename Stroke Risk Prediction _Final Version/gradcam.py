import numpy as np
import tensorflow as tf
import cv2
import os

def make_gradcam_heatmap(model, image, last_conv_layer_name="conv2d_2", output_path="static/heatmap.png"):
    try:
        # Debugging model structure
        print("🔍 [DEBUG] Grad-CAM input shape:", image.shape)
        print("🔍 [DEBUG] Model.inputs:", model.inputs)
        print("🔍 [DEBUG] Model.outputs:", model.outputs)

        # Build grad_model for Grad-CAM
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        print("✅ [DEBUG] Grad model built.")

        # Forward + Gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, 0]

        print("✅ [DEBUG] Predictions for Grad-CAM:", predictions.numpy())

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))

        # Compute heatmap
        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (256, 256))
        cam = np.uint8(255 * cam)

        # Apply color map
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save image
        success = cv2.imwrite(output_path, heatmap)
        if success:
            print(f"✅ [DEBUG] Heatmap saved successfully to '{output_path}'")
        else:
            print(f"❌ [ERROR] Failed to save heatmap to '{output_path}'")

    except Exception as e:
        print("❌ [Grad-CAM ERROR]:", str(e))
