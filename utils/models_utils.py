from pathlib import Path
from PIL import Image
import os
from functools import partial

import numpy as np
import tensorflow as tf
import shap

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input

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
def load_image(path, label, resnet_preprocess=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)

    if resnet_preprocess:
        image = preprocess_input(image)
    else:
        image = image / 255.0

    return image, label


# Separates the sets in batches, allowing for better control of memory usage,
# especially when fitting the GPU.
def create_tf_dataset(X, y, resnet_preprocess=False, batch_size=16, shuffle=False, seed=42):
    tf.keras.utils.set_random_seed(seed)
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(len(X), seed=seed)

    load_image_fn = partial(load_image, resnet_preprocess=resnet_preprocess)

    ds = (
        ds
        .map(load_image_fn, num_parallel_calls=tf.data.AUTOTUNE)
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

def save_model(save_dir, model_name, model, history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = save_dir / model_name / f"plant_disease_cnn_{timestamp}"
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

def load_model(save_dir, model_name, timestamp):
    model_dir = save_dir / model_name / f"plant_disease_cnn_{timestamp}"

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

# Generate a Grad-CAM heatmap for one image.
def make_gradcam_heatmap(image, model, last_conv_layer_name, class_idx=None):
    # Find the requested layer, including inside nested models
    last_conv_layer = None
    nested_model = None

    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            last_conv_layer = layer
            break

        if isinstance(layer, tf.keras.Model):
            try:
                last_conv_layer = layer.get_layer(last_conv_layer_name)
                nested_model = layer
                break
            except ValueError:
                continue

    if last_conv_layer is None:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' was not found in the model "
            "or its nested models."
        )

    # Build the Grad-CAM model
    if nested_model is None:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
    else:
        # Expose the convolutional activations and backbone output
        backbone_grad_model = tf.keras.Model(
            inputs=nested_model.input,
            outputs=[last_conv_layer.output, nested_model.output]
        )

        # Reuse the original classification head
        head = model.layers[model.layers.index(nested_model) + 1:]

        image_input = tf.keras.Input(shape=image.shape)
        conv_outputs, backbone_output = backbone_grad_model(image_input)

        x = backbone_output
        for layer in head:
            x = layer(x)

        grad_model = tf.keras.Model(
            inputs=image_input,
            outputs=[conv_outputs, x]
        )

    image_batch = tf.convert_to_tensor(image[np.newaxis, ...])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)

        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]).numpy())

        class_score = predictions[:, class_idx]

    # Gradients of the selected class score with respect to conv activations
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise ValueError(
            "Could not calculate gradients. Check that the selected layer "
            "contributes to the model's predictions."
        )

    # Average gradients over spatial dimensions to obtain filter weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each activation map by its importance for this class
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    # Keep positive contributions and normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    return heatmap.numpy(), int(class_idx)

# Generate a Grad-CAM heatmap of all target_classes for all mistakes of the same group
def analyze_gradcam_mistakes(model, X, y, diseases, target_classes, last_conv_layer_name, resnet_preprocess=False, batch_size=16):
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=1)
    else:   
        y = y.reshape(-1).astype(int)

    target_indices = [diseases.index(name) for name in target_classes]

    # Predict in batches
    predictions = []
    for start in tqdm(range(0, len(X), batch_size)):
        images = [
            load_image(path, label, resnet_preprocess=resnet_preprocess)[0]
            for path, label in zip(
                X[start:start + batch_size],
                y[start:start + batch_size]
            )
        ]
        predictions.append(
            model.predict(tf.stack(images), verbose=0)
        )

    predictions = np.concatenate(predictions, axis=0)
    predicted_labels = np.argmax(predictions, axis=1)

    mistake_indices = np.flatnonzero(np.isin(y, target_indices) & (predicted_labels != y))

    print(f"Found {len(mistake_indices)} misclassified images.")

    results = []

    for image_idx in tqdm(mistake_indices):
        image_tensor, _ = load_image(
            X[image_idx], y[image_idx], resnet_preprocess=resnet_preprocess
        )
        image = image_tensor.numpy()

        prediction = model.predict(
            image[np.newaxis, ...], verbose=0
        )[0]

        true_idx = int(y[image_idx])
        pred_idx = int(predicted_labels[image_idx])

        # Get the top 3 predictions.
        top3_indices = np.argsort(prediction)[-3:][::-1]

        # Grad-CAM for the true class
        true_heatmap, _ = make_gradcam_heatmap(
            image, model, last_conv_layer_name,
            class_idx=true_idx
        )

        # Grad-CAM for the predicted class
        pred_heatmap, _ = make_gradcam_heatmap(
            image, model, last_conv_layer_name,
            class_idx=pred_idx
        )

        def resize_heatmap(heatmap):
            return tf.image.resize(
                heatmap[..., np.newaxis],
                image.shape[:2]
            ).numpy().squeeze()

        results.append({
            "index": int(image_idx),
            "path": X[image_idx],
            "image": image,
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "prediction": prediction,
            "top3_indices": top3_indices,
            "confidence": float(predictions[image_idx, pred_idx]),
            "true_heatmap": resize_heatmap(true_heatmap),
            "pred_heatmap": resize_heatmap(pred_heatmap)
        })

    return results

# Generate a Shapley image plot for all images in indices for its true and predicted classes
def analyze_shap(model, X_val, y_val, diseases, indices, resnet_preprocess=False, max_evals=100, batch_size=16, masker_method="blur(64,64)"):
    y = np.asarray(y_val).reshape(-1).astype(int)

    masker = shap.maskers.Image(masker_method, (256, 256, 3))

    explainer = shap.Explainer(
        lambda images: model(images, training=False).numpy(),
        masker,
        output_names=diseases,
        algorithm="partition"
    )

    results = []

    for idx in tqdm(indices):
        idx = int(idx)

        if idx < 0 or idx >= len(X_val):
            print(f"Index {idx} is out of range. Skipping.")
            continue

        image_tensor, _ = load_image(
            X_val[idx], y[idx], resnet_preprocess=resnet_preprocess
        )
        image = image_tensor.numpy()

        prediction = model.predict(
            image[np.newaxis, ...], verbose=0
        )[0]

        true_idx = int(y[idx])
        pred_idx = int(np.argmax(prediction))

        # Get the top 3 predictions.
        top3_indices = np.argsort(prediction)[-3:][::-1]

        # Calculate SHAP values for the true and predicted classes.
        output_indices = sorted({true_idx, pred_idx})

        shap_values = explainer(
            image[np.newaxis, ...],
            max_evals=max_evals,
            batch_size=batch_size,
            outputs=output_indices
        )

        results.append({
            "index": idx,
            "path": str(X_val[idx]),
            "image": image,
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "prediction": prediction,
            "top3_indices": top3_indices,
            "output_indices": output_indices,
            "shap_values": shap_values
        })

    return results