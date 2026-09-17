import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config
from src.data_loader import load_dataset_file_paths
from src.preprocessing import get_preprocessing_function
from src.models import apply_cbam_attention

os.makedirs(config.GRADCAM_DIR, exist_ok=True)

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        if len(layer.output_shape) == 4 and "conv" in layer.name:
            return layer.name
    return None

def compute_gradcam(model, img_array, target_layer_name, pred_index=None):
    base_m = getattr(model, "base_model", None)
    
    # Standard flat model or backbone model
    if base_m is None:
        try:
            target_layer = model.get_layer(target_layer_name)
        except Exception:
            target_layer_name = get_last_conv_layer_name(model)
            target_layer = model.get_layer(target_layer_name)
            
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[target_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    else:
        # Transfer model with base_model property
        try:
            conv_layer = base_m.get_layer(target_layer_name)
        except Exception:
            target_layer_name = get_last_conv_layer_name(base_m)
            conv_layer = base_m.get_layer(target_layer_name)
            
        base_grad_model = tf.keras.models.Model(inputs=base_m.inputs, outputs=conv_layer.output)
        
        with tf.GradientTape() as tape:
            conv_outputs = base_grad_model(img_array)
            tape.watch(conv_outputs)
            x = conv_outputs
            for l in model.layers:
                if l.name not in [base_m.name, "input_image", "input_1"]:
                    x = l(x)
            predictions = x
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

def generate_gradcam_grid_for_model(m_name, m_path, sample_paths):
    print(f"\n==================================================")
    print(f"GENERATING GRAD-CAM EXPLAINABILITY GRID FOR: {m_name}")
    print(f"==================================================")

    custom_objects = {"apply_cbam_attention": apply_cbam_attention}
    model = tf.keras.models.load_model(m_path, custom_objects=custom_objects)
    
    input_shape = model.input_shape[1:3] if model.input_shape[1] is not None else (224, 224)
    pfn = get_preprocessing_function(m_name)
    
    if m_name == "ResNet50":
        target_layer = "conv5_block3_out"
    elif m_name == "DenseNet121":
        target_layer = "relu"
    elif m_name == "EfficientNetV2B0":
        target_layer = "top_activation"
    else:
        target_layer = get_last_conv_layer_name(model)

    print(f"Using target layer: '{target_layer}'")

    fig, axes = plt.subplots(4, 3, figsize=(10, 13), dpi=300)
    plt.suptitle(f"{m_name} — Grad-CAM Explainability Grid", fontsize=14, fontweight='bold', y=0.98)

    for idx, (cls_name, img_p) in enumerate(sample_paths):
        raw_img = cv2.imread(img_p)
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img_resized = cv2.resize(raw_img_rgb, input_shape)

        img_arr = np.expand_dims(np.array(raw_img_resized, dtype=np.float32), axis=0)
        preprocessed_x = pfn(img_arr)

        preds = model.predict(preprocessed_x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_cls = config.CLASSES[pred_idx]
        conf = float(preds[pred_idx]) * 100.0

        try:
            heatmap = compute_gradcam(model, preprocessed_x, target_layer, pred_index=pred_idx)
            heatmap_resized = cv2.resize(heatmap, input_shape)
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(raw_img_resized, 0.6, color_heatmap_rgb, 0.4, 0)
        except Exception as e:
            print(f"Warning: GradCAM error for class {cls_name}: {e}")
            color_heatmap_rgb = np.zeros_like(raw_img_resized)
            overlay = raw_img_resized

        # Column 0: Original Scan
        axes[idx, 0].imshow(raw_img_resized)
        axes[idx, 0].set_title(f"True: {cls_name.capitalize()}", fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')

        # Column 1: Grad-CAM Heatmap
        axes[idx, 1].imshow(color_heatmap_rgb)
        axes[idx, 1].set_title(f"Grad-CAM Heatmap", fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')

        # Column 2: Overlay + Prediction
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f"Pred: {pred_cls.capitalize()} ({conf:.1f}%)", fontsize=10, fontweight='bold', color='green' if pred_cls == cls_name else 'red')
        axes[idx, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_key = m_name.replace(" ", "_")
    output_path = os.path.join(config.GRADCAM_DIR, f"{save_key}_gradcam_grid.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM grid to: {output_path}")

def main():
    dataset_dir = "/Users/hitesh/Downloads/final_dataset_clean/dataset "
    if not os.path.exists(dataset_dir):
        dataset_dir = "/Users/hitesh/Downloads/final_dataset_clean"

    splits = load_dataset_file_paths(data_dir=dataset_dir)
    test_paths = splits['test_paths']
    test_labels = splits['test_labels']

    sample_paths = []
    for i, c in enumerate(config.CLASSES):
        indices = np.where(test_labels == i)[0]
        if len(indices) > 0:
            sample_paths.append((c, test_paths[indices[0]]))

    models_info = [
        ("ResNet50", os.path.join(config.MODELS_DIR, "ResNet50_best.keras")),
        ("DenseNet121", os.path.join(config.MODELS_DIR, "DenseNet121_best.keras")),
        ("EfficientNetV2B0", os.path.join(config.MODELS_DIR, "EfficientNetV2B0_best.keras")),
        ("Baseline Attention CNN", os.path.join(config.MODELS_DIR, "Baseline_CNN.keras")),
    ]

    for m_name, m_path in models_info:
        generate_gradcam_grid_for_model(m_name, m_path, sample_paths)

    print("\n==================================================")
    print("ALL 4 GRAD-CAM EXPLAINABILITY GRIDS SUCCESSFULLY CREATED!")
    print("==================================================")

if __name__ == "__main__":
    main()
