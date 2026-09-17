import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from src import config

def evaluate_model(model, test_ds, model_name="model", history=None):


    y_true_list = []
    y_pred_probs_list = []

    start_time = time.time()
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_pred_probs_list.append(probs)
        y_true_list.append(y_batch.numpy())
    end_time = time.time()

    y_true_onehot = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    num_samples = len(y_true)

    inference_time_ms = ((end_time - start_time) / num_samples) * 1000.0

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    
    try:
        roc_auc = roc_auc_score(y_true_onehot, y_pred_probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = 0.0

    total_params = model.count_params()

    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(config.NUM_CLASSES)
    )

    metrics_summary = {
        "model_name": model_name,
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(w_f1),
        "roc_auc": float(roc_auc),
        "parameters": int(total_params),
        "inference_time_ms": float(inference_time_ms),
        "per_class": {
            config.CLASSES[i]: {
                "precision": float(p_class[i]),
                "recall": float(r_class[i]),
                "f1_score": float(f1_class[i]),
                "support": int(support_class[i])
            } for i in range(config.NUM_CLASSES)
        }
    }

    print(f"Accuracy         : {acc:.4f}")
    print(f"Macro Precision  : {macro_p:.4f}")
    print(f"Macro Recall     : {macro_r:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print(f"Weighted F1-Score: {w_f1:.4f}")
    print(f"ROC-AUC (OvR)    : {roc_auc:.4f}")
    print(f"Parameters       : {total_params:,}")
    print(f"Inference Speed  : {inference_time_ms:.2f} ms/sample\n")

    print("Per-Class Classification Metrics:")
    report_str = classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4)
    print(report_str)

    _plot_confusion_matrix(y_true, y_pred, model_name)
    _plot_roc_curves(y_true_onehot, y_pred_probs, model_name)
    _plot_pr_curves(y_true_onehot, y_pred_probs, model_name)
    
    if history is not None:
        _plot_learning_curves(history, model_name)

    return metrics_summary, y_true, y_pred, y_pred_probs

def _plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    out_path = os.path.join(config.CONFUSION_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Confusion matrix saved to {out_path}")

def _plot_roc_curves(y_true_onehot, y_pred_probs, model_name):
    plt.figure(figsize=(8, 6))
    colors = ['navy', 'turquoise', 'darkorange', 'crimson']

    for i, color in zip(range(config.NUM_CLASSES), colors):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        auc_score = roc_auc_score(y_true_onehot[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{config.CLASSES[i]} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves (One-vs-Rest) — {model_name}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(config.PLOTS_DIR, f"{model_name}_roc_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] ROC curve saved to {out_path}")

def _plot_pr_curves(y_true_onehot, y_pred_probs, model_name):
    """Plot and save multi-class Precision-Recall curves."""
    plt.figure(figsize=(8, 6))
    colors = ['navy', 'turquoise', 'darkorange', 'crimson']

    for i, color in zip(range(config.NUM_CLASSES), colors):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2, label=f"{config.CLASSES[i]}")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curves — {model_name}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(config.PLOTS_DIR, f"{model_name}_pr_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def _plot_learning_curves(history, model_name):
    """Plot and save training vs validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    if "loss" in history:
        plt.plot(history["loss"], label="Train Loss", lw=2, color="crimson")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Val Loss", lw=2, color="navy")
    plt.title(f"Loss Curves — {model_name}", fontsize=13, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    if "accuracy" in history:
        plt.plot(history["accuracy"], label="Train Acc", lw=2, color="crimson")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Val Acc", lw=2, color="navy")
    plt.title(f"Accuracy Curves — {model_name}", fontsize=13, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, f"{model_name}_learning_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Learning curves saved to {out_path}")

def generate_comparison_table(metrics_list):
    """Create and print a formatted summary comparison table across all evaluated models."""
    df = pd.DataFrame(metrics_list)
    cols = ["model_name", "accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "roc_auc", "parameters", "inference_time_ms"]
    df = df[cols]

    # Rename columns for presentation
    df.columns = ["Model", "Accuracy", "Macro Precision", "Macro Recall",
                  "Macro F1", "Weighted F1", "ROC-AUC", "Parameters", "Inference Time (ms)"]

    print("\n==================================================")
    print("FINAL MODEL COMPARISON TABLE")
    print("==================================================")
    print(df.to_string(index=False))
    print("==================================================\n")

    csv_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[REPORT] Model comparison saved to {csv_path}")
    return df

def perform_error_analysis(model, test_paths, test_labels, preprocess_fn=None, model_name="Best_Model"):
    """
    Perform deep qualitative error analysis:
    - Identify correctly vs incorrectly classified samples
    - Save sample plot grid to results/plots/error_analysis_samples.png
    - Document findings in results/reports/error_analysis.md
    """
    print(f"\n==================================================")
    print(f"PERFORMING ERROR ANALYSIS FOR: {model_name}")
    print(f"==================================================")

    classes = config.CLASSES
    correct_samples = []
    incorrect_samples = []

    for path, true_label in zip(test_paths, test_labels):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        img_tensor = tf.cast(img, tf.float32)
        
        if preprocess_fn is not None:
            inp = preprocess_fn(img_tensor)
        else:
            inp = img_tensor / 255.0

        inp_batch = tf.expand_dims(inp, axis=0)
        probs = model.predict(inp_batch, verbose=0)[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])

        item = {
            "path": path,
            "true_class": classes[true_label],
            "pred_class": classes[pred_label],
            "confidence": confidence,
            "probs": probs.tolist()
        }

        if pred_label == true_label:
            correct_samples.append(item)
        else:
            incorrect_samples.append(item)

    print(f"Total Test Samples: {len(test_paths)}")
    print(f"Correctly Classified: {len(correct_samples)}")
    print(f"Misclassified: {len(incorrect_samples)}")

    # Plot incorrect samples grid
    if incorrect_samples:
        fig, axes = plt.subplots(min(4, len(incorrect_samples)), 3, figsize=(12, 3 * min(4, len(incorrect_samples))))
        if len(incorrect_samples) == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, sample in enumerate(incorrect_samples[:4]):
            img = tf.image.decode_jpeg(tf.io.read_file(sample["path"]), channels=3).numpy().astype("uint8")
            ax = axes[idx, 0] if axes.ndim > 1 else axes[idx]
            ax.imshow(img)
            ax.set_title(f"True: {sample['true_class']}\nPred: {sample['pred_class']} ({sample['confidence']*100:.1f}%)",
                         fontsize=10, color="crimson", fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        out_plot = os.path.join(config.PLOTS_DIR, "error_analysis_samples.png")
        plt.savefig(out_plot, dpi=300)
        plt.close()

    report_content = f"""# Qualitative Error Analysis Report — {model_name}

## Summary Statistics
- **Total Test Samples**: {len(test_paths)}
- **Correct Predictions**: {len(correct_samples)} ({len(correct_samples)/len(test_paths)*100:.2f}%)
- **Misclassifications**: {len(incorrect_samples)} ({len(incorrect_samples)/len(test_paths)*100:.2f}%)

## Confused Tumor Class Patterns
Below are representative misclassified samples and confidence breakdown:

"""
    for i, item in enumerate(incorrect_samples[:10]):
        report_content += f"### Sample {i+1}: {os.path.basename(item['path'])}\n"
        report_content += f"- **True Class**: `{item['true_class']}`\n"
        report_content += f"- **Predicted Class**: `{item['pred_class']}` (Confidence: {item['confidence']*100:.2f}%)\n"
        report_content += f"- **Probabilities**: Glioma: {item['probs'][0]:.3f}, Meningioma: {item['probs'][1]:.3f}, NoTumor: {item['probs'][2]:.3f}, Pituitary: {item['probs'][3]:.3f}\n\n"

    report_path = os.path.join(config.REPORTS_DIR, "error_analysis.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"[REPORT] Error analysis report written to {report_path}")
