import os
import random
import numpy as np
import tensorflow as tf
from src import config

def set_seeds(seed=config.SEED):
    """Set random seed for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[REPRODUCIBILITY] Random seeds set to {seed}")

def get_hardware_info():
    """Detect and log hardware availability and TensorFlow runtime configuration."""
    tf_version = tf.__version__
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    gpu_name = gpus[0].name if gpu_available else "None (CPU Execution)"
    mixed_precision = tf.keras.mixed_precision.global_policy().name

    info = {
        "tf_version": tf_version,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "mixed_precision": mixed_precision
    }

    print("==================================================")
    print("HARDWARE & ENVIRONMENT DIAGNOSTICS")
    print("==================================================")
    print(f"TensorFlow Version : {tf_version}")
    print(f"GPU Available      : {gpu_available}")
    print(f"GPU Device Name    : {gpu_name}")
    print(f"Mixed Precision    : {mixed_precision}")
    print("==================================================\n")
    return info
