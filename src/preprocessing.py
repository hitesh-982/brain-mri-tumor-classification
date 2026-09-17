import tensorflow as tf

def get_preprocessing_function(model_name):
    """
    Return architecture-specific preprocessing function/layer for pretrained ImageNet models and custom CNN.
    
    IMPORTANT: Pretrained models use distinct normalization strategies:
      - Baseline CNN    : Simple [0, 1] rescaling.
      - ResNet50        : Zero-centered Caffe-style BGR channel preprocessing.
      - DenseNet121     : Scaled to [-1, 1].
      - EfficientNetV2B0: Scaled to [-1, 1] or architectural normalization.
      - ConvNeXt-Tiny   : Preprocessing layer for ConvNeXt.
    """
    model_key = model_name.lower().replace("-", "").replace("_", "")

    if "baseline" in model_key or "scratch" in model_key:
        # Standard [0, 1] rescaling
        return lambda x: x / 255.0
    
    elif "resnet50" in model_key:
        # ResNet50 expects BGR inputs with ImageNet mean subtracted
        return tf.keras.applications.resnet50.preprocess_input

    elif "densenet" in model_key:
        # DenseNet expects inputs scaled to [-1, 1]
        return tf.keras.applications.densenet.preprocess_input

    elif "efficientnet" in model_key:
        # EfficientNetV2 expects inputs scaled to [-1, 1]
        return tf.keras.applications.efficientnet_v2.preprocess_input

    elif "convnext" in model_key:
        return tf.keras.applications.convnext.preprocess_input

    else:
        # Default fallback
        return lambda x: x / 255.0
