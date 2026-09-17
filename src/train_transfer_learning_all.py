"""
Unified Transfer Learning Training Pipeline for Brain MRI Classification
Serves: ResNet50, DenseNet121, and EfficientNetV2B0
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers, models, regularizers

from src import config
from src.utils import set_seeds, get_hardware_info
from src.data_loader import load_dataset_file_paths, get_class_weights, create_tf_dataset
from src.preprocessing import get_preprocessing_function
from src.augmentation import build_augmentation_pipeline
from src.evaluate import evaluate_model, generate_comparison_table

IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# ==============================================================================
# 1. RESNET50 ARCHITECTURE & UNFREEZING
# ==============================================================================
def build_resnet50(num_classes=config.NUM_CLASSES, weight_decay=1e-4):
    """
    Build ResNet50 Transfer Learning Model.
    Head: GAP -> BN -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(4, Softmax)
    """
    inputs = layers.Input(shape=INPUT_SHAPE, name="input_image")
    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    # Freeze backbone & Batch Normalization
    for layer in backbone.layers:
        layer.trainable = False
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(weight_decay), name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_256")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50")
    model.base_model = backbone
    return model

def unfreeze_resnet50_stage2(model):
    """Unfreeze conv4_block* and conv5_block* while keeping BN layers frozen."""
    unfrozen_count = 0
    for layer in model.layers:
        if "conv5_block" in layer.name or "conv4_block" in layer.name:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                unfrozen_count += 1
        elif "gap" in layer.name or "bn_head" in layer.name or "dense" in layer.name or "predictions" in layer.name:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            layer.trainable = False

    print(f"[STAGE 2 UNFREEZE] ResNet50: Unfrozen {unfrozen_count} layers in conv4/5 blocks (BN frozen).")
    return model


# ==============================================================================
# 2. DENSENET121 ARCHITECTURE & UNFREEZING
# ==============================================================================
def build_densenet121(num_classes=config.NUM_CLASSES, weight_decay=1e-4):
    """
    Build DenseNet121 Transfer Learning Model.
    Head: GAP -> BN -> Dense(256, ReLU) -> Dropout(0.4) -> Dense(128, ReLU) -> Dropout(0.3) -> Softmax(4)
    """
    inputs = layers.Input(shape=INPUT_SHAPE, name="input_image")
    backbone = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    for layer in backbone.layers:
        layer.trainable = False
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(weight_decay), name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout_256")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(weight_decay), name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_128")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DenseNet121")
    model.base_model = backbone
    return model

def unfreeze_densenet121_stage2(model):
    """Unfreeze denseblock3 and denseblock4 while keeping BN layers frozen."""
    unfrozen_count = 0
    for layer in model.layers:
        if "conv4" in layer.name or "conv5" in layer.name or "denseblock3" in layer.name or "denseblock4" in layer.name:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                unfrozen_count += 1
        elif "gap" in layer.name or "bn_head" in layer.name or "dense" in layer.name or "predictions" in layer.name:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            layer.trainable = False

    print(f"[STAGE 2 UNFREEZE] DenseNet121: Unfrozen {unfrozen_count} layers in denseblock3/4 (BN frozen).")
    return model


# ==============================================================================
# 3. EFFICIENTNETV2B0 ARCHITECTURE & UNFREEZING
# ==============================================================================
def build_efficientnetv2b0(num_classes=config.NUM_CLASSES, weight_decay=1e-4):
    """
    Build EfficientNetV2B0 Transfer Learning Model.
    Head: GAP -> BN -> Dense(256, Swish) -> Dropout(0.3) -> Dense(128, Swish) -> Dropout(0.3) -> Softmax(4)
    """
    inputs = layers.Input(shape=INPUT_SHAPE, name="input_image")
    backbone = tf.keras.applications.EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    for layer in backbone.layers:
        layer.trainable = False
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(256, activation="swish", kernel_regularizer=regularizers.l2(weight_decay), name="dense_256")(x)
    x = layers.Dropout(0.3, name="dropout_256")(x)
    x = layers.Dense(128, activation="swish", kernel_regularizer=regularizers.l2(weight_decay), name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_128")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetV2B0")
    model.base_model = backbone
    return model

def unfreeze_efficientnetv2_stage2(model):
    """Unfreeze top 30% of blocks in EfficientNetV2B0 while keeping BN layers frozen."""
    unfrozen_count = 0
    total_layers = len(model.layers)
    start_unfreeze_idx = int(total_layers * 0.7)

    for idx, layer in enumerate(model.layers):
        if idx >= start_unfreeze_idx:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                unfrozen_count += 1
        else:
            layer.trainable = False

    print(f"[STAGE 2 UNFREEZE] EfficientNetV2B0: Unfrozen {unfrozen_count} layers in top blocks (BN frozen).")
    return model


# ==============================================================================
# 4. UNIFIED PIPELINE RUNNER FOR ANY MODEL
# ==============================================================================
def train_and_evaluate_transfer_model(model_name, splits, class_weights, aug):
    """
    Executes 2-Stage Training for a given model_name: 'ResNet50', 'DenseNet121', or 'EfficientNetV2B0'.
    """
    print(f"\n==================================================")
    print(f"STARTING UNIFIED PIPELINE FOR: {model_name}")
    print(f"==================================================")

    train_paths, train_labels = splits["train_paths"], splits["train_labels"]
    val_paths, val_labels = splits["val_paths"], splits["val_labels"]
    test_paths, test_labels = splits["test_paths"], splits["test_labels"]

    pfn = get_preprocessing_function(model_name)

    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=config.BATCH_SIZE,
                                 is_training=True, preprocess_fn=pfn,
                                 augmentation_model=aug, img_size=IMG_SIZE)

    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=config.BATCH_SIZE,
                               is_training=False, preprocess_fn=pfn,
                               img_size=IMG_SIZE)

    test_ds = create_tf_dataset(test_paths, test_labels, batch_size=config.BATCH_SIZE,
                                is_training=False, preprocess_fn=pfn,
                                img_size=IMG_SIZE)

    # Instantiate model & unfreeze function based on model_name
    if model_name == "ResNet50":
        model = build_resnet50()
        unfreeze_fn = unfreeze_resnet50_stage2
    elif model_name == "DenseNet121":
        model = build_densenet121()
        unfreeze_fn = unfreeze_densenet121_stage2
    elif model_name == "EfficientNetV2B0":
        model = build_efficientnetv2b0()
        unfreeze_fn = unfreeze_efficientnetv2_stage2
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Stage 1: Train classification head
    print(f"\n[STAGE 1] Training adaptation head for {model_name}...")
    opt1 = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02)
    metrics = ["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]

    model.compile(optimizer=opt1, loss=loss_fn, metrics=metrics)

    ckpt_path_s1 = os.path.join(config.MODELS_DIR, f"{model_name}_stage1.keras")
    callbacks_s1 = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path_s1, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]

    h1 = model.fit(train_ds, validation_data=val_ds, epochs=5,
                   class_weight=class_weights, callbacks=callbacks_s1, verbose=1)

    hist_combined = {k: [float(x) for x in v] for k, v in h1.history.items()}

    # Stage 2: Fine-Tuning
    print(f"\n[STAGE 2] Fine-tuning deep layers for {model_name}...")
    best_s1_model = tf.keras.models.load_model(ckpt_path_s1)
    model = unfreeze_fn(best_s1_model)

    opt2 = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    model.compile(optimizer=opt2, loss=loss_fn, metrics=metrics)

    ckpt_path_best = os.path.join(config.MODELS_DIR, f"{model_name}_best.keras")
    callbacks_s2 = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path_best, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]

    h2 = model.fit(train_ds, validation_data=val_ds, epochs=10,
                   class_weight=class_weights, callbacks=callbacks_s2, verbose=1)

    for k, v in h2.history.items():
        if k in hist_combined:
            hist_combined[k].extend([float(x) for x in v])
        else:
            hist_combined[k] = [float(x) for x in v]

    # Save canonical model & training history
    canonical_path = os.path.join(config.MODELS_DIR, f"{model_name}.keras")
    model.save(canonical_path)
    
    hist_json_path = os.path.join(config.MODELS_DIR, f"{model_name}_history.json")
    with open(hist_json_path, "w") as f:
        json.dump(hist_combined, f, indent=2)

    # Evaluation & Plots
    print(f"\n[EVALUATION] Generating test evaluation metrics & plots for {model_name}...")
    best_model = tf.keras.models.load_model(ckpt_path_best)
    m_metrics, _, _, _ = evaluate_model(best_model, test_ds, model_name=model_name, history=hist_combined)

    print(f"\n>>> {model_name} PIPELINE COMPLETE <<<")
    print(f"Test Accuracy : {m_metrics['Accuracy']*100:.2f}%")
    print(f"Macro F1      : {m_metrics['Macro F1']*100:.2f}%")
    return m_metrics


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Transfer Learning Models (ResNet50, DenseNet121, EfficientNetV2B0)")
    parser.add_argument("--model", type=str, default="all", choices=["resnet50", "densenet121", "efficientnetv2", "all"],
                        help="Select model to train: resnet50, densenet121, efficientnetv2, or all")
    args = parser.parse_args()

    set_seeds(config.SEED)
    get_hardware_info()

    splits = load_dataset_file_paths()
    class_weights = get_class_weights(splits["train_labels"])
    aug = build_augmentation_pipeline(enabled=True)

    models_to_run = []
    if args.model == "all":
        models_to_run = ["ResNet50", "DenseNet121", "EfficientNetV2B0"]
    elif args.model == "resnet50":
        models_to_run = ["ResNet50"]
    elif args.model == "densenet121":
        models_to_run = ["DenseNet121"]
    elif args.model == "efficientnetv2":
        models_to_run = ["EfficientNetV2B0"]

    results = {}
    for m_name in models_to_run:
        res = train_and_evaluate_transfer_model(m_name, splits, class_weights, aug)
        results[m_name] = res

    # Generate model comparison table
    if len(results) > 1:
        generate_comparison_table()

if __name__ == "__main__":
    main()
