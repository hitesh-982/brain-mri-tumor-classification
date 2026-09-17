import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src import config

def load_dataset_file_paths(data_dir=config.DATA_DIR, seed=config.SEED):

    classes = config.CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_paths = []
    all_labels = []

    for c in classes:
        c_dir = os.path.join(data_dir, c)
        files = glob.glob(os.path.join(c_dir, "*"))
        for f in files:
            if os.path.isfile(f) and not os.path.basename(f).startswith('.'):
                all_paths.append(f)
                all_labels.append(class_to_idx[c])

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=all_labels,
        random_state=seed
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5, 
        stratify=temp_labels,
        random_state=seed
    )

    print("==================================================")
    print("2. STRATIFIED DATASET SPLIT SUMMARY")
    print("==================================================")
    print(f"Total Dataset Images : {len(all_paths)}")
    print(f"Training Set (70%)   : {len(train_paths)} samples")
    print(f"Validation Set (15%) : {len(val_paths)} samples")
    print(f"Test Set (15%)       : {len(test_paths)} samples")
    print("==================================================\n")

    split_data = {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "val_paths": val_paths,
        "val_labels": val_labels,
        "test_paths": test_paths,
        "test_labels": test_labels,
    }
    return split_data

def get_class_weights(train_labels):
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[CLASS WEIGHTS] Computed balanced class weights: {class_weight_dict}")
    return class_weight_dict

def parse_image(path, label, img_size=config.IMAGE_SIZE):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    label_one_hot = tf.one_hot(label, config.NUM_CLASSES)
    return img, label_one_hot

def create_tf_dataset(paths, labels, batch_size=config.BATCH_SIZE, is_training=False,
                      preprocess_fn=None, augmentation_model=None, img_size=config.IMAGE_SIZE):

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    dataset = dataset.map(lambda p, l: parse_image(p, l, img_size=img_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if is_training and augmentation_model is not None:
        dataset = dataset.map(lambda x, y: (augmentation_model(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if preprocess_fn is not None:
        dataset = dataset.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
