import os
import json
import tensorflow as tf
from src import config
from src.models import unfreeze_top_layers

def get_callbacks(model_name, stage_suffix=""):
    """Construct training callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau."""
    ckpt_path = os.path.join(config.MODELS_DIR, f"{model_name}{stage_suffix}.keras")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks, ckpt_path

def compile_model(model, learning_rate=config.BASELINE_LR, weight_decay=config.WEIGHT_DECAY, total_steps=None):
    """Compile model with AdamW optimizer, optional CosineDecay schedule, CategoricalCrossentropy with label smoothing (0.02), and evaluation metrics."""
    if total_steps is not None:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps
        )
        opt_lr = lr_schedule
    else:
        opt_lr = learning_rate

    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=opt_lr,
            weight_decay=weight_decay
        )
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)
    
    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_model(model, train_ds, val_ds, class_weights=None, model_name="model",
                stage1_epochs=config.STAGE1_EPOCHS, stage2_epochs=config.STAGE2_EPOCHS,
                stage1_lr=config.STAGE1_LR, stage2_lr=config.STAGE2_LR,
                is_two_stage=True, unfreeze_layers=30):
    """
    Execute full training pipeline (Stage 1 frozen backbone + optional Stage 2 fine-tuning).
    Returns the trained model and merged training history dictionary.
    """
    print(f"\n==================================================")
    print(f"TRAINING MODEL: {model_name} (Two-Stage = {is_two_stage})")
    print(f"==================================================")

    history_combined = {}

    # STAGE 1
    print(f"\n--- STAGE 1: Training Classification Head (LR = {stage1_lr}) ---")
    try:
        steps_per_epoch = len(train_ds)
    except Exception:
        steps_per_epoch = 151
    total_steps = steps_per_epoch * stage1_epochs
    compile_model(model, learning_rate=stage1_lr, total_steps=total_steps)
    callbacks_s1, ckpt_s1 = get_callbacks(model_name, stage_suffix="_stage1")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_s1,
        verbose=1
    )

    for k, v in history1.history.items():
        history_combined[k] = [float(x) for x in v]

    # STAGE 2 FINE-TUNING (For transfer learning models)
    if is_two_stage and stage2_epochs > 0:
        print(f"\n--- STAGE 2: Fine-Tuning Upper Backbone Layers (LR = {stage2_lr}) ---")
        model = unfreeze_top_layers(model, num_layers_to_unfreeze=unfreeze_layers)
        compile_model(model, learning_rate=stage2_lr)
        callbacks_s2, ckpt_s2 = get_callbacks(model_name, stage_suffix="_best")

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage2_epochs,
            class_weight=class_weights,
            callbacks=callbacks_s2,
            verbose=1
        )

        for k, v in history2.history.items():
            if k in history_combined:
                history_combined[k].extend([float(x) for x in v])
            else:
                history_combined[k] = [float(x) for x in v]

        final_model_path = ckpt_s2
    else:
        final_model_path = ckpt_s1

    # Save best model to canonical name
    canonical_path = os.path.join(config.MODELS_DIR, f"{model_name}.keras")
    model.save(canonical_path)
    print(f"[SAVE] Final best model saved to: {canonical_path}")

    # Save training history
    hist_json_path = os.path.join(config.MODELS_DIR, f"{model_name}_history.json")
    with open(hist_json_path, "w") as f:
        json.dump(history_combined, f, indent=2)

    return model, history_combined
