import tensorflow as tf

def build_augmentation_pipeline(enabled=True):

    if not enabled:
        return tf.keras.Sequential([tf.keras.layers.Identity()], name="no_augmentation")

    augmentation_layers = [
        tf.keras.layers.RandomRotation(factor=0.03, fill_mode="nearest"), # ~10 degree rotation
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=0.05, width_factor=0.05, fill_mode="nearest"),
        tf.keras.layers.RandomFlip(mode="horizontal"), # MRI axial planes are medially symmetric
        tf.keras.layers.RandomContrast(factor=0.1),
    ]

    return tf.keras.Sequential(augmentation_layers, name="mri_data_augmentation")
