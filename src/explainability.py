import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src import config

def find_last_conv_layer(model):
    """
    Search recursively or sequentially through the model layers to locate the last Conv2D layer.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If it's a nested base_model or functional layer
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError(f"Could not find a Conv2D layer in model {model.name}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Compute Grad-CAM heatmap for a given input image tensor and target convolution layer.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Check if target conv layer is in base_model or model root
    try:
        target_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
    except Exception:
        # Target conv layer inside backbone base_model
        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            target_layer = base_model.get_layer(last_conv_layer_name)
            grad_base = tf.keras.models.Model(
                inputs=base_model.inputs,
                outputs=target_layer.output
            )
            
            # Reconstruction for full forward pass tape tracking
            def forward_pass(x):
                conv_out = grad_base(x)
                # forward through top layers
                feat = base_model(x)
                preds = model(x)
                return conv_out, preds
            
            with tf.GradientTape() as tape:
                img_tensor = tf.cast(img_array, tf.float32)
                tape.watch(img_tensor)
                conv_outputs, predictions = forward_pass(img_tensor)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                loss = predictions[:, pred_index]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
            return heatmap.numpy()
        else:
            raise ValueError(f"Unable to trace layer {last_conv_layer_name}")

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_visualizations(model, sample_paths_by_class, preprocess_fn=None, model_name="Best_Model"):
    """
    Generate and save overlaid Grad-CAM heatmaps for representative MRI samples across all 4 classes.
    """
    print(f"\n==================================================")
    print(f"12. GENERATING GRAD-CAM INTERPRETABILITY MAPS")
    print(f"==================================================")
    print("NOTE: Grad-CAM provides model feature interpretability for engineering validation, NOT clinical diagnostic proof.\n")

    classes = config.CLASSES
    fig, axes = plt.subplots(len(classes), 3, figsize=(10, 3 * len(classes)))

    for i, class_name in enumerate(classes):
        paths = sample_paths_by_class.get(class_name, [])
        if not paths:
            continue
        
        sample_path = paths[0]

        # Load & Preprocess
        orig_img = cv2.imread(sample_path)
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, config.IMAGE_SIZE)

        img_tensor = tf.cast(resized_img, tf.float32)
        if preprocess_fn is not None:
            inp = preprocess_fn(img_tensor)
        else:
            inp = img_tensor / 255.0

        inp_batch = tf.expand_dims(inp, axis=0)

        # Prediction
        probs = model.predict(inp_batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Heatmap
        try:
            heatmap = make_gradcam_heatmap(inp_batch, model, pred_index=pred_idx)
            heatmap_resized = cv2.resize(heatmap, (orig_img_rgb.shape[1], orig_img_rgb.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            jet_heatmap_rgb = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
            
            overlay = cv2.addWeighted(orig_img_rgb, 0.6, jet_heatmap_rgb, 0.4, 0)
        except Exception as e:
            print(f"[WARNING] Grad-CAM generation failed for class {class_name}: {e}")
            overlay = orig_img_rgb
            heatmap_resized = np.zeros((orig_img_rgb.shape[0], orig_img_rgb.shape[1]))

        # Column 1: Original MRI
        ax1 = axes[i, 0]
        ax1.imshow(orig_img_rgb)
        ax1.set_title(f"Original MRI ({class_name})", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Column 2: Heatmap
        ax2 = axes[i, 1]
        ax2.imshow(heatmap_resized, cmap="jet")
        ax2.set_title("Grad-CAM Activation", fontsize=10, fontweight="bold")
        ax2.axis("off")

        # Column 3: Overlay
        ax3 = axes[i, 2]
        ax3.imshow(overlay)
        ax3.set_title(f"Pred: {classes[pred_idx]} ({confidence*100:.1f}%)",
                      fontsize=10, fontweight="bold",
                      color="navy" if pred_idx == i else "crimson")
        ax3.axis("off")

    plt.tight_layout()
    out_path = os.path.join(config.GRADCAM_DIR, f"{model_name}_gradcam_grid.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[GRAD-CAM] Heatmap visualizations saved to {out_path}")
