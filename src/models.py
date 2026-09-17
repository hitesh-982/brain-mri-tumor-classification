import tensorflow as tf
import keras
from keras import layers, models, regularizers
from src import config

def apply_cbam_attention(x, reduction_ratio=16):
    """
    Convolutional Block Attention Module (CBAM):
    Combines Channel Attention (SE style) and Spatial Attention.
    """
    channels = x.shape[-1]
    
    # 1. Channel Attention
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(x)
    
    shared_dense_one = layers.Dense(channels // reduction_ratio, activation='relu', use_bias=False)
    shared_dense_two = layers.Dense(channels, use_bias=False)
    
    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    max_out = shared_dense_two(shared_dense_one(max_pool))
    
    channel_attention = layers.Activation('sigmoid')(avg_out + max_out)
    x = layers.Multiply()([x, channel_attention])
    
    # 2. Spatial Attention
    avg_spatial = keras.ops.mean(x, axis=-1, keepdims=True)
    max_spatial = keras.ops.max(x, axis=-1, keepdims=True)
    spatial_concat = keras.ops.concatenate([avg_spatial, max_spatial], axis=-1)
    spatial_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid', use_bias=False)(spatial_concat)
    
    x = layers.Multiply()([x, spatial_attention])
    return x

def build_baseline_cnn(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES,
                       weight_decay=config.WEIGHT_DECAY, dropout_rate=config.DROPOUT_RATE):
    """
    Model 1 — Baseline Attention CNN.
    Input 256x256
    Conv32 -> BN -> Conv32 -> BN -> MaxPool
    Conv64 -> BN -> Conv64 -> BN -> MaxPool
    Conv128 -> BN -> Conv128 -> BN -> MaxPool
    Conv256 -> BN -> Conv256 -> BN -> MaxPool
    Conv512 -> BN -> Conv512 -> BN
    Attention Block (CBAM)
    GlobalAveragePooling
    Dense(256) -> Dropout(0.4)
    Dense(4, activation='softmax')
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs

    # Block 1 (32 filters)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2 (64 filters)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3 (128 filters)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4 (256 filters)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 5 (512 filters)
    x = layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Attention Block
    x = apply_cbam_attention(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_Attention_CNN")
    return model

def build_transfer_learning_model(model_name, input_shape=config.INPUT_SHAPE,
                                  num_classes=config.NUM_CLASSES, weight_decay=config.WEIGHT_DECAY,
                                  dropout_rate=config.DROPOUT_RATE):
    """
    Factory to construct ImageNet pretrained transfer-learning architectures:
      - ResNet50 (Modified Adaptation Head: GAP -> BN -> Dense(512, ReLU) -> Dropout(0.5) -> Dense(128, ReLU) -> Dropout(0.3) -> Softmax(4))
      - DenseNet121
      - EfficientNetV2B0
      - ConvNeXt-Tiny (Optional)
    """
    name_key = model_name.lower().replace("-", "").replace("_", "")
    
    # ResNet50 uses 224x224 input
    if "resnet" in name_key:
        shape = (224, 224, 3)
    else:
        shape = input_shape

    inputs = layers.Input(shape=shape, name="input_image")

    if "resnet50" in name_key:
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        base_name = "ResNet50"
        base_model.trainable = False

        # Modified Adaptation Head for ResNet50
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.BatchNormalization(name="bn_post_gap")(x)
        x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(weight_decay), name="dense_512")(x)
        x = layers.Dropout(0.5, name="dropout_512")(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(weight_decay), name="dense_128")(x)
        x = layers.Dropout(0.3, name="dropout_128")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name=base_name)
        model.base_model = base_model
        return model

    elif "densenet121" in name_key:
        base_model = tf.keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        base_name = "DenseNet121"

    elif "efficientnet" in name_key:
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        base_name = "EfficientNetV2B0"

    elif "convnext" in name_key:
        base_model = tf.keras.applications.ConvNeXtTiny(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        base_name = "ConvNeXtTiny"

    else:
        raise ValueError(f"Unsupported transfer learning architecture: {model_name}")

    # Freeze base model for Stage 1 training
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=base_name)
    model.base_model = base_model # store reference to base model for stage 2 unfreezing
    return model

def unfreeze_top_layers(model, num_layers_to_unfreeze=30):
    """
    Unfreeze top layers for Stage 2 fine-tuning.
    For ResNet50: Unfreezes conv5_block1, conv5_block2, conv5_block3, and conv4_block* layers.
    Maintains BatchNormalization layers in frozen inference mode to avoid destabilizing statistics.
    """
    if not hasattr(model, "base_model"):
        print(f"[WARNING] Model {model.name} has no base_model attribute. Unfreezing full model layers.")
        target_model = model
    else:
        target_model = model.base_model

    target_model.trainable = True

    model_key = model.name.lower()

    if "resnet" in model_key:
        # Unfreeze conv5_block* and conv4_block* for ResNet50
        unfrozen_count = 0
        for layer in target_model.layers:
            if "conv5_block" in layer.name or "conv4_block" in layer.name:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
                    unfrozen_count += 1
            else:
                layer.trainable = False
        print(f"[STAGE 2 FINE-TUNING] ResNet50: Unfrozen conv5_block* and conv4_block* ({unfrozen_count} trainable layers). BN layers kept frozen.")
    else:
        total_layers = len(target_model.layers)
        freeze_until = max(0, total_layers - num_layers_to_unfreeze)

        for i, layer in enumerate(target_model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

        trainable_count = sum([1 for l in target_model.layers if l.trainable])
        print(f"[STAGE 2 FINE-TUNING] Unfrozen top {num_layers_to_unfreeze} layers of {model.name}. Trainable layers: {trainable_count}")

    return model
