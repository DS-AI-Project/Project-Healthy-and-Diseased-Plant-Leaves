from pathlib import Path
from PIL import Image
from IPython.display import display, HTML
import html as html_lib

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import pandas as pd

import random
import math

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

    # Add overall accuracy as the final row
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    overall_accuracy = (
        total_correct / total_samples * 100
        if total_samples else 0
    )

    rows.append({
        "Real value": "Overall Accuracy",
        "Accuracy (absolute and %)": (
            f"{total_correct}/{total_samples} "
            f"({overall_accuracy:.2f}%)"
        ),
        "Mistakes (list and %)": ""
    })

    results = pd.DataFrame(rows)

    # Display table inside a scrollable container
    display(HTML(
        f"""
        <div style="max-width: 100%; max-height: 500px;
                    overflow: auto; border: 1px solid #ccc;">
            {results.to_html(index=False, escape=True)}
        </div>
        """
    ))

    # return results

# Plots all premade Grad-CAMs heatmaps
def plot_cached_gradcam(mistakes, diseases, resnet_preprocess=False, images_per_row=2, max_rows_per_figure=3, 
                        alpha=0.4, cmap="jet", figsize_per_image=(4, 3)):
    per_figure = images_per_row * max_rows_per_figure

    for start in range(0, len(mistakes), per_figure):
        subset = mistakes[start:start + per_figure]
        n = len(subset)
        rows = math.ceil(n / images_per_row)

        fig, axes = plt.subplots(
            rows, images_per_row * 3,
            figsize=(
                figsize_per_image[0] * images_per_row * 3,
                figsize_per_image[1] * rows
            ),
            squeeze=False
        )

        for i, item in enumerate(subset):
            r = i // images_per_row
            c = (i % images_per_row) * 3

            true_name = diseases[item["true_idx"]]
            pred_name = diseases[item["pred_idx"]]

            # Convert the image to displayable RGB values
            image = item["image"].copy()

            if resnet_preprocess:
                # Undo ResNet50 preprocessing: BGR -> RGB, add ImageNet means
                image = image[..., ::-1]
                image += np.array([103.939, 116.779, 123.68], dtype=np.float32)
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 1)

            idx = item["index"]

            top3_indices = item.get(
                "top3_indices",
                np.argsort(item["prediction"])[-3:][::-1]
            )

            # Original image
            axes[r, c].imshow(image)
            prediction_text = "\n".join(
                f"{rank}. {diseases[class_idx]}: "
                f"{item['prediction'][class_idx]:.2%}"
                for rank, class_idx in enumerate(top3_indices, start=1)
            )
            axes[r, 0].set_title(
                f"Image index: {idx}\n"
                f"True: {true_name}\n"
                f"Top 3 predictions:\n{prediction_text}",
                fontsize=9
            )
            axes[r, 0].axis("off")
            
            # Grad-CAM for true class
            axes[r, c + 1].imshow(image)
            axes[r, c + 1].imshow(
                item["true_heatmap"],
                cmap=cmap, alpha=alpha, vmin=0, vmax=1
            )
            axes[r, c + 1].set_title(
                f"True-class Grad-CAM: {true_name}",
                fontsize=8
            )
            axes[r, c + 1].axis("off")
            
            # Grad-CAM for predicted class
            axes[r, c + 2].imshow(image)
            axes[r, c + 2].imshow(
                item["pred_heatmap"],
                cmap=cmap, alpha=alpha, vmin=0, vmax=1
            )
            axes[r, c + 2].set_title(
                f"Predicted-class Grad-CAM: {pred_name}",
                fontsize=8
            )
            axes[r, c + 2].axis("off")

        # Hide unused subplot cells
        for i in range(n, rows * images_per_row):
            r = i // images_per_row
            c = (i % images_per_row) * 3
            for j in range(3):
                axes[r, c + j].axis("off")

        fig.suptitle(
            f"Misclassified images ({start + 1}–{start + n} "
            f"of {len(mistakes)})"
        )
        plt.tight_layout()
        plt.show()

# Plots all premade Shapleys image plots
def plot_shap(results, diseases, resnet_preprocess=False, max_images_per_figure=4):
    for start in range(0, len(results), max_images_per_figure):
        subset = results[start:start + max_images_per_figure]

        fig, axes = plt.subplots(
            len(subset), 3,
            figsize=(16, 4.5 * len(subset)),
            squeeze=False
        )

        for row, item in enumerate(subset):
            # Convert the image to displayable RGB values
            image = item["image"].copy()

            if resnet_preprocess:
                # Undo ResNet50 preprocessing: BGR -> RGB, add ImageNet means
                image = image[..., ::-1]
                image += np.array([103.939, 116.779, 123.68], dtype=np.float32)
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 1)
            values = item["shap_values"]

            # Original image and top 3 predictions.
            axes[row, 0].imshow(image)

            top3_indices = item.get(
                "top3_indices",
                np.argsort(item["prediction"])[-3:][::-1]
            )

            prediction_text = "\n".join(
                f"{rank}. {diseases[class_idx]}: "
                f"{item['prediction'][class_idx]:.2%}"
                for rank, class_idx in enumerate(top3_indices, start=1)
            )

            axes[row, 0].set_title(
                f"X_val index: {item['index']}\n"
                f"True: {diseases[item['true_idx']]}\n"
                f"Top 3 predictions:\n{prediction_text}",
                fontsize=9
            )
            axes[row, 0].axis("off")

            vals = np.asarray(values.values)
            output_indices = item["output_indices"]

            # Remove the batch dimension.
            if vals.shape[0] == 1:
                vals = vals[0]

            for col, class_idx, title in [
                (1, item["true_idx"], "True-class SHAP"),
                (2, item["pred_idx"], "Predicted-class SHAP")
            ]:
                if class_idx not in output_indices:
                    axes[row, col].axis("off")
                    continue

                output_pos = output_indices.index(class_idx)

                # Expected SHAP shape after removing batch:
                # (height, width, channels, selected_outputs)
                if vals.ndim == 4:
                    class_values = vals[..., output_pos]
                elif vals.ndim == 3 and len(output_indices) == 1:
                    class_values = vals
                else:
                    raise ValueError(
                        f"Unexpected SHAP shape: "
                        f"{np.asarray(values.values).shape}"
                    )

                # Sum channel attributions to get a signed spatial map.
                spatial_values = class_values.sum(axis=-1)

                vmax = np.max(np.abs(spatial_values)) + 1e-12

                axes[row, col].imshow(image)
                axes[row, col].imshow(
                    spatial_values,
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                    alpha=0.65
                )

                axes[row, col].set_title(
                    f"{title}\n{diseases[class_idx]}",
                    fontsize=9
                )
                axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()

# Plots all commom mistakes of two models images and Grad-CAMs heatmaps for their true and predicted classes
def plot_coinciding_mistakes(mistakes_1, mistakes_2, diseases, resnet_preprocess=(False, False),
                             figsize_per_image=(16, 4), alpha=0.4, cmap="jet",):
    if len(resnet_preprocess) != 2:
        raise ValueError(
            "resnet_preprocess must be a tuple of two booleans."
        )

    # Match mistakes by their original dataset index.
    mistakes_by_index_1 = {
        mistake["index"]: mistake for mistake in mistakes_1
    }
    mistakes_by_index_2 = {
        mistake["index"]: mistake for mistake in mistakes_2
    }

    common_indices = sorted(
        set(mistakes_by_index_1) & set(mistakes_by_index_2)
    )

    if not common_indices:
        print("No coinciding mistakes found.")
        return

    print(f"Found {len(common_indices)} coinciding mistakes.")

    # Plot one row per shared mistake.
    fig, axes = plt.subplots(
        len(common_indices),
        5,
        figsize=(
            figsize_per_image[0],
            figsize_per_image[1] * len(common_indices),
        ),
        squeeze=False,
    )

    for row, idx in enumerate(common_indices):
        m1 = mistakes_by_index_1[idx]
        m2 = mistakes_by_index_2[idx]

        # Convert ResNet-preprocessed images back to RGB [0, 1].
        # ResNet50 preprocessing: RGB -> BGR, then subtract means.
        image = m1["image"].astype(np.float32).copy()

        image = np.asarray(
            Image.open(m1["path"]).convert("RGB")
        ).astype(np.float32) / 255.0

        # Original image.
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Index {idx}\n"
            f"True: {diseases[m1['true_idx']]}"
        )

        # Model 1: true-class and predicted-class heatmaps.
        for col, heatmap_key, title in [
            (1, "true_heatmap", "Model 1: True class"),
            (2, "pred_heatmap", "Model 1: Predicted class"),
        ]:
            axes[row, col].imshow(image)
            axes[row, col].imshow(
                m1[heatmap_key], cmap=cmap, alpha=alpha
            )
            axes[row, col].set_title(
                f"{title}\n"
                f"{diseases[m1['pred_idx'] if col == 2 else m1['true_idx']]}"
            )

        # Model 2: true-class and predicted-class heatmaps.
        for col, heatmap_key, title in [
            (3, "true_heatmap", "Model 2: True class"),
            (4, "pred_heatmap", "Model 2: Predicted class"),
        ]:
            axes[row, col].imshow(image)
            axes[row, col].imshow(
                m2[heatmap_key], cmap=cmap, alpha=alpha
            )
            axes[row, col].set_title(
                f"{title}\n"
                f"{diseases[m2['pred_idx'] if col == 4 else m2['true_idx']]}"
            )

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

# Plots a table with all unique mistakes of each model and its probabilities
def plot_unique_mistakes(rows, diseases):
    if not rows:
        print("No unique mistakes found.")
        return

    table_data = [
        {
            "Mistook image": row["index"],
            "True value": diseases[row["true_idx"]],
            "First model top 3 predictions": "<br>".join(
                f"{i}. {diseases[int(class_idx)]} "
                f"({row['prediction_1'][class_idx]:.2%})"
                for i, class_idx in enumerate(
                    row["top3_indices_1"], start=1
                )
            ),
            "Second model top 3 predictions": "<br>".join(
                f"{i}. {diseases[int(class_idx)]} "
                f"({row['prediction_2'][class_idx]:.2%})"
                for i, class_idx in enumerate(
                    row["top3_indices_2"], start=1
                )
            )
        }
        for row in rows
    ]

    df = pd.DataFrame(table_data)

    # Escape ordinary cell values, but allow <br> in prediction columns.
    columns = list(df.columns)
    html_rows = []

    for _, row in df.iterrows():
        cells = []

        for column in columns:
            value = str(row[column])

            if column in (
                "First model top 3 predictions",
                "Second model top 3 predictions"
            ):
                cells.append(f"<td>{value}</td>")
            else:
                cells.append(f"<td>{html_lib.escape(value)}</td>")

        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = (
        '<table><colgroup>'
        '<col style="width: 14.3%;">'
        '<col style="width: 28.57%;">'
        '<col style="width: 28.57%;">'
        '<col style="width: 28.57%;">'
        '</colgroup><thead><tr>'
        + "".join(
            f"<th>{html_lib.escape(col)}</th>"
            for col in columns
        )
        + "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )

    display(HTML(
        f"""
        <div style="max-width:100%; max-height:600px; overflow:auto;">
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 8px;
                    text-align: left;
                    vertical-align: top;
                }}
                th {{
                    position: sticky;
                    top: 0;
                    background: #e0e0e0;
                    color: #222222;
                }}
            </style>
            {html}
        </div>
        """
    ))