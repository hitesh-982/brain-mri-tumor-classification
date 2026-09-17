import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR 

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

SEED = 42

IMAGE_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 3)

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

BATCH_SIZE = 32
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 5
BASELINE_EPOCHS = 8

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
BASELINE_LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.02

UNFREEZE_LAYERS = {
    "resnet50": 30,
    "densenet121": 30,
    "efficientnetv2b0": 30,
}

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

def ensure_dirs():

    for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, CONFUSION_DIR, GRADCAM_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_dirs()
