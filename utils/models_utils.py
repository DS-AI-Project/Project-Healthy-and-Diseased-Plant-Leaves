from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Save the trained model, the class names and the training history
import json
from datetime import datetime

import random
from tqdm import tqdm
import sys

def test_cuda(verbose):
    tf.debugging.set_log_device_placement(verbose)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(gpus)

    print("Python:", sys.version)
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

# Classifies each image
def classify_set(dataset, possible_diseases):
    paths = []
    labels = []

    disease_to_index = {
        disease: i
        for i, disease in enumerate(possible_diseases)
    }

    for item in tqdm(dataset):
        path = Path(item["text"])

        # raw/<type>/<class>/<filename>
        class_name = path.parts[path.parts.index("raw") + 2]

        # Extract disease
        disease = class_name.split("___")[1]

        paths.append(
            str(Path("PlantVillage-Dataset") / item["text"])
        )
        labels.append(disease_to_index[disease])

    return np.array(paths), np.array(labels, dtype=np.int32)

# Loads the images as needed, saving memory usage
# The code is currently made for the colored images, which have consistent sizes and 3 channels
def load_image(path, label):
    image = tf.io.read_file(path)
    
    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# Separates the sets in batches, allowing for better control of the memory usage, espcially to fit the GPU
def create_tf_dataset(X, y, batch_size=16, shuffle=False, seed=42):
    tf.keras.utils.set_random_seed(seed)
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(len(X))

    ds = (
        ds
        # load_image been used for every batch
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds

def randomize_set(diseases, train_set, train_size, test_set, test_size, valuation_size=0.2, seed=42):
    # Randomizes some images for training, evaluating and testing
    tf.keras.utils.set_random_seed(seed)

    reduced_train = random.sample(train_set, train_size)
    reduced_train_set, validation_set = train_test_split(
        reduced_train,
        test_size=valuation_size,
        shuffle=True
    )

    reduced_test_set = random.sample(test_set, test_size)

    # Classifies each image
    X_train, y_train = classify_set(reduced_train_set, diseases)
    X_val, y_val = classify_set(validation_set, diseases)
    X_test, y_test = classify_set(reduced_test_set, diseases)

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def test_model(model, train_dataset, valuation_dataset, patience=5, learning_rate=0.0001, epochs=10, seed=42):
    tf.keras.utils.set_random_seed(seed)
    
    # If the model isn't learning, stops it early to save on processing time
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history = model.fit(
        train_dataset,
        validation_data=valuation_dataset,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    return history

def save_model(save_dir, model, history, diseases):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = save_dir / f"plant_disease_cnn_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Full model: architecture + weights + optimizer state
    model_path = model_dir / "model.keras"
    model.save(model_path)

    # Training history (accuracy/loss per epoch)
    with open(model_dir / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2
        )

    print(f"Model saved to: {model_dir.resolve()}")

def load_model(save_dir, timestamp):
    model_dir = save_dir / f"plant_disease_cnn_{timestamp}"

    model_path = model_dir / "model.keras"
    history_path = model_dir / "history.json"

    # Load the full model: architecture + weights + optimizer state
    model = tf.keras.models.load_model(model_path)

    # Load training history
    with open(history_path, "r") as f:
        history_data = json.load(f)

    # Reconstruct a Keras History object
    history = tf.keras.callbacks.History()
    history.history = history_data
    history.epoch = list(range(len(history_data["loss"])))

    print(f"Model loaded from: {model_dir.resolve()}")

    return model, history