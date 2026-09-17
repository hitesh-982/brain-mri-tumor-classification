import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report
import tensorflow as tf

from src import config
from src.data_loader import load_dataset_file_paths
from src.preprocessing import get_preprocessing_function
from src.models import apply_cbam_attention

def main():
    print("==================================================")
    print("GENERATING ALL CURVES FOR BASELINE ATTENTION CNN")
    print("==================================================")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    m_name = "Baseline Attention CNN"
    file_key = "Baseline_Attention_CNN"
    model_path = os.path.join(config.MODELS_DIR, "Baseline_CNN.keras")
    history_path = os.path.join(config.MODELS_DIR, "Baseline_Attention_CNN_history.json")

    # 1. GENERATE COMBINED LEARNING CURVES (LOSS & ACCURACY OVER EPOCHS)
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            
        epochs = range(1, len(history["accuracy"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Plot Loss
        ax1.plot(epochs, history["loss"], "o-", color="#ef4444", lw=2, label="Train Loss")
        ax1.plot(epochs, history["val_loss"], "s--", color="#f97316", lw=2, label="Validation Loss")
        ax1.set_title(f"{m_name} — Training & Validation Loss", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Epochs", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper right", frameon=True, facecolor="#f8fafc")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Plot Accuracy
        ax2.plot(epochs, [a * 100 for a in history["accuracy"]], "o-", color="#10b981", lw=2, label="Train Accuracy")
        ax2.plot(epochs, [a * 100 for a in history["val_accuracy"]], "s--", color="#06b6d4", lw=2, label="Validation Accuracy")
        ax2.set_title(f"{m_name} — Training & Validation Accuracy", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Epochs", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
        ax2.legend(loc="lower right", frameon=True, facecolor="#f8fafc")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(f"{m_name} — Training Dynamics & Convergence", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        lc_path = os.path.join(config.PLOTS_DIR, f"{file_key}_learning_curves_combined.png")
        plt.savefig(lc_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"1. Saved Combined Learning Curves: {lc_path}")

    # 2. RUN INFERENCE FOR ROC & PER-CLASS METRICS
    dataset_dir = "/Users/hitesh/Downloads/final_dataset_clean/dataset "
    if not os.path.exists(dataset_dir):
        dataset_dir = "/Users/hitesh/Downloads/final_dataset_clean"

    splits = load_dataset_file_paths(data_dir=dataset_dir)
    test_paths = splits['test_paths']
    test_labels = splits['test_labels']

    print(f"\nLoading model from {model_path} for ROC & Per-Class Metrics...")
    custom_objects = {"apply_cbam_attention": apply_cbam_attention}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    input_shape = model.input_shape[1:3] if model.input_shape[1] is not None else (256, 256)
    pfn = get_preprocessing_function(m_name)

    test_images = []
    for p in test_paths:
        img = tf.keras.preprocessing.image.load_img(p, target_size=input_shape)
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        test_images.append(img_arr)
        
    test_images = np.array(test_images, dtype=np.float32)
    preprocessed_x = pfn(test_images)
    preds = model.predict(preprocessed_x, batch_size=32, verbose=1)
    pred_labels = np.argmax(preds, axis=1)

    # 3. MULTI-CLASS ROC CURVES PLOT
    plt.figure(figsize=(8, 6), dpi=300)
    test_labels_onehot = tf.one_hot(test_labels, 4).numpy()
    colors = ['#ef4444', '#f59e0b', '#10b981', '#8b5cf6']
    
    for i, c in enumerate(config.CLASSES):
        fpr, tpr, _ = roc_curve(test_labels_onehot[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.5, label=f'{c.capitalize()} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=10, fontweight='bold')
    plt.title(f'{m_name} — Multi-Class ROC Curves', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, facecolor="#f8fafc")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    roc_path = os.path.join(config.PLOTS_DIR, f"{file_key}_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"2. Saved Multi-Class ROC Curves plot: {roc_path}")

    # 4. PER-CLASS PERFORMANCE METRICS BAR CHART
    report_dict = classification_report(test_labels, pred_labels, target_names=config.CLASSES, output_dict=True)
    classes_data = []
    for c in config.CLASSES:
        classes_data.append({
            'Class': c.capitalize(),
            'Precision': report_dict[c]['precision'] * 100,
            'Recall': report_dict[c]['recall'] * 100,
            'F1-Score': report_dict[c]['f1-score'] * 100
        })
    df_p = pd.DataFrame(classes_data)
    df_melt = pd.melt(df_p, id_vars=['Class'], var_name='Metric', value_name='Score (%)')

    plt.figure(figsize=(8.5, 5.5), dpi=300)
    ax = sns.barplot(data=df_melt, x='Class', y='Score (%)', hue='Metric', palette='viridis')
    plt.title(f'{m_name} — Per-Class Performance Metrics (Test Set)', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Score (%)', fontsize=10, fontweight='bold')
    plt.xlabel('Tumor Diagnosis Class', fontsize=10, fontweight='bold')
    plt.ylim([60, 105])
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold', xytext=(0, 2),
                        textcoords='offset points')
                        
    plt.legend(loc='lower right', frameon=True, facecolor="#f8fafc")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    per_class_path = os.path.join(config.PLOTS_DIR, f"{file_key}_per_class_metrics.png")
    plt.savefig(per_class_path, dpi=300)
    plt.close()
    print(f"3. Saved Per-Class Metrics Bar Chart: {per_class_path}")

    print("\n==================================================")
    print("ALL BASELINE ATTENTION CNN CURVES SUCCESSFULLY GENERATED!")
    print("==================================================")

if __name__ == "__main__":
    main()
