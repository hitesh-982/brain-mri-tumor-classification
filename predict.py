import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
from src import config
from src.preprocessing import get_preprocessing_function

def predict_mri(image_path, model_path=None, model_name="ResNet50"):
    """
    Run single-image inference using a saved model checkpoint.
    Outputs predicted class, confidence, and class probabilities.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist: {image_path}")
        sys.exit(1)

    if model_path is None or not os.path.exists(model_path):
        canonical_path = os.path.join(config.MODELS_DIR, f"{model_name}.keras")
        if os.path.exists(canonical_path):
            model_path = canonical_path
        else:
            # Look for any available saved model in models/
            available_models = [f for f in os.listdir(config.MODELS_DIR) if f.endswith(".keras")]
            if not available_models:
                print(f"Error: No saved models found in {config.MODELS_DIR}. Please train a model first.")
                sys.exit(1)
            model_path = os.path.join(config.MODELS_DIR, available_models[0])

    print(f"[INFERENCE] Loading model checkpoint from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Preprocessing
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, config.IMAGE_SIZE)
    img_tensor = tf.cast(resized_img, tf.float32)

    # Determine preprocessing function from model name or path
    inferred_name = os.path.basename(model_path).replace(".keras", "")
    preprocess_fn = get_preprocessing_function(inferred_name)
    inp = preprocess_fn(img_tensor)
    inp_batch = tf.expand_dims(inp, axis=0)

    # Run Prediction
    probs = model.predict(inp_batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = config.CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0

    print("\n==================================================")
    print("BRAIN MRI TUMOR INFERENCE RESULT")
    print("==================================================")
    print(f"Input Image Path : {image_path}")
    print(f"Predicted Class  : {pred_class.upper()}")
    print(f"Confidence       : {confidence:.2f}%")
    print("--------------------------------------------------")
    print("Probability Breakdown across Classes:")
    for idx, c in enumerate(config.CLASSES):
        bar = "█" * int(probs[idx] * 30)
        print(f"  {c:<12}: {probs[idx]*100:>6.2f}% {bar}")
    print("==================================================\n")

    return pred_class, confidence, probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Classification Inference Script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input MRI image file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved .keras model checkpoint")
    parser.add_argument("--model_name", type=str, default="ResNet50", help="Model name if model_path is not specified")

    args = parser.parse_args()
    predict_mri(args.image_path, args.model_path, args.model_name)
