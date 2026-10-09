from pathlib import Path
from PIL import Image
from IPython.display import display, HTML

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import pandas as pd

import random

def plot_random_dataset_images(dataset_path, sets):
    arrays = {
        "Train - Color": sets["train_color"],
        "Train - Grayscale": sets["train_grayscale"],
        "Train - Segmented": sets["train_segmented"],
        "Test - Color": sets["test_color"],
        "Test - Grayscale": sets["test_grayscale"],
        "Test - Segmented": sets["test_segmented"]
    }

    fig, axes = plt.subplots(6, 3, figsize=(15, 24))

    for row, (name, array) in enumerate(arrays.items()):
        samples = random.sample(array, 3)

        for col, item in enumerate(samples):
            path = Path(item["text"])

            # raw/<type>/<class>/<filename>
            class_name = path.parts[path.parts.index("raw") + 2]

            image_path = Path(dataset_path) / item["text"]
            image = Image.open(image_path)

            axes[row, col].imshow(image)
            axes[row, col].set_title(
                f"{name}\n{class_name}",
                fontsize=10
            )
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def plot_diseases_table(diseases, disease_counts, precision):
    rows = []

    for disease in diseases:
        row = {"Disease": disease}

        for key, counts in disease_counts.items():
            total = sum(counts.values())
            count = counts.get(disease, 0)
            percentage = count / total * 100 if total > 0 else 0

            row[key] = f"{count:,} ({percentage:.{precision}f}%)"

        rows.append(row)

    counts_table = pd.DataFrame(rows)

    return counts_table

def plot_model_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def model_confusion_matrix(model, test_ds, diseases):
    # Predictions
    y_pred = model.predict(test_ds)
    y_pred = y_pred.argmax(axis=1)

    # Get true labels from the dataset
    y_test = np.concatenate([
        labels.numpy()
        for _, labels in test_ds
    ])

    # Forces all 21 classes to be included
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=range(len(diseases))
    )

    fig, ax = plt.subplots(figsize=(14, 14))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=diseases
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Build per-class statistics
    rows = []

    for i, disease in enumerate(diseases):
        total = cm[i].sum()
        correct = cm[i, i]
        mistakes = []

        for j, predicted_disease in enumerate(diseases):
            if i != j and cm[i, j] > 0:
                count = cm[i, j]
                percentage = count / total * 100 if total else 0

                mistakes.append(
                    f"{predicted_disease}: {count} ({percentage:.2f}%)"
                )

        accuracy_percentage = (
            correct / total * 100 if total else 0
        )

        rows.append({
            "Real value": disease,
            "Accuracy (absolute and %)": (
                f"{correct}/{total} ({accuracy_percentage:.2f}%)"
            ),
            "Mistakes (list and %)": (
                "; ".join(mistakes) if mistakes else "None"
            )
        })

    results = pd.DataFrame(rows)

    # Display table inside a scrollable container
    display(HTML(
        results.to_html(index=False, escape=True)
        .replace(
            '<table ',
            '<table style="white-space: nowrap; border-collapse: collapse;" '
        )
        .join([]) if False else
        f"""
        <div style="max-width: 100%; max-height: 500px;
                    overflow: auto; border: 1px solid #ccc;">
            {results.to_html(index=False, escape=True)}
        </div>
        """
    ))

    #return results