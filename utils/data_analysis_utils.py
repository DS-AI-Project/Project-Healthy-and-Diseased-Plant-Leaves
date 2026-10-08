from pathlib import Path
from PIL import Image

import subprocess
from datasets import load_dataset

from tqdm import tqdm
from collections import Counter

def download_images(dataset_path):
    # Downloads the dataset's images from GitHub
    repo_url = "https://github.com/spMohanty/PlantVillage-Dataset.git"

    if dataset_path.exists():
        print(f"Images already exists: {dataset_path.resolve()}")
    else:
        print("Images not found. Cloning...")
        subprocess.run(
            ["git", "clone", repo_url, str(dataset_path)],
            check=True
        )
        print(f"Dataset cloned to: {dataset_path.resolve()}")

def download_dataset():
    # Downloads the paths to the images, along with it's descriptors
    # load_dataset automatically checks if it's already been downloaded, so this block can be ran everytime without major delays
    dataset = load_dataset(
        "mohanty/PlantVillage",
        "default"
    )
    return dataset

def separate_sets(dataset):
    sets = {
        "train_color": [],
        "train_grayscale": [],
        "train_segmented": [],
        "test_color": [],
        "test_grayscale": [],
        "test_segmented": []
    }

    for split in ["train", "test"]:
        types = []

        # Separate images
        for item in tqdm(dataset[split]):
            path = Path(item["text"])
            parts = path.parts

            # raw/<type>/...
            image_type = parts[parts.index("raw") + 1]

            types.append(image_type)
            sets[f"{split}_{image_type}"].append(item)

        print(f"\n===== {split.upper()} =====")

        # Find contiguous ranges for each type
        ranges = {
            "color": [],
            "grayscale": [],
            "segmented": []
        }

        if types:
            start = 0
            current = types[0]

            for i in range(1, len(types)):
                if types[i] != current:
                    ranges[current].append((start, i - 1))
                    start = i
                    current = types[i]

            ranges[current].append((start, len(types) - 1))

        # Print each type separately
        for image_type in ["color", "grayscale", "segmented"]:
            print(f"\n{image_type.upper()}:")

            if not ranges[image_type]:
                print("  Not present")
            else:
                for start, end in ranges[image_type]:
                    print(
                        f"  Indexes: {start:,} - {end:,} "
                        f"({end - start + 1:,} images)"
                    )

    return sets

def define_conditions(sets):
    disease_counts = {}

    for set_name, dataset in sets.items():
        disease_counts[set_name] = Counter()

        for item in tqdm(dataset):
            # If item is a dictionary, then it's checking the datasets descriptors
            if isinstance(item, dict):
                path = Path(item["text"])
            # If it doesn't, is a training/valuation/test set
            else:
                path = Path(item)

            parts = path.parts

            # raw/<type>/<class>/...
            class_name = parts[parts.index("raw") + 2]

            disease = class_name.split("___")[1]

            disease_counts[set_name][disease] += 1

    # All conditions present in any set
    diseases = sorted({
        disease
        for counts in disease_counts.values()
        for disease in counts
    })

    return diseases, disease_counts

def check_images_sizes(dataset_path, sets):
    for name, array in sets.items():
        sizes = set()

        for item in tqdm(array):
            image_path = Path(dataset_path) / item["text"]

            with Image.open(image_path) as image:
                sizes.add(image.size)

        print(f"{name}: {sizes}")