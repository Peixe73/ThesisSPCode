import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

from core import datasets
from pathlib import Path


# ---- PATHS ----
IMAGES_PATH = Path("data/gtsrb_dataset")
DEBUG_DIR = Path("debugGTSRB")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def load_original_image(row):
    path = IMAGES_PATH / row["Filename"]
    return Image.open(path).convert("RGB")


def crop_roi(img, row):
    x1, y1 = row["Roi.X1"], row["Roi.Y1"]
    x2, y2 = row["Roi.X2"], row["Roi.Y2"]
    return img.crop((x1, y1, x2, y2))


def make_square(img):
    w, h = img.size
    size = max(w, h)

    new_img = Image.new("RGB", (size, size))
    new_img.paste(img, ((size - w) // 2, (size - h) // 2))
    return new_img


def resize_128(img):
    return img.resize((128, 128))


def tensor_to_image(tensor):
    return tensor.permute(1, 2, 0).cpu().numpy()

"""
def visualize_all(original, cropped, squared, final_manual, final_dataset, idx):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    # ---- ORIGINAL ----
    axes[0].imshow(original)
    axes[0].set_title(f"Original\n{original.size}")
    axes[0].axis("off")

    # ---- CROPPED ----
    axes[1].imshow(cropped)
    axes[1].set_title(f"Cropped\n{cropped.size}")
    axes[1].axis("off")

    # ---- SQUARED ----
    axes[2].imshow(squared)
    axes[2].set_title(f"Squared\n{squared.size}")
    axes[2].axis("off")

    # ---- FINAL (MANUAL) ----
    axes[3].imshow(final_manual)
    axes[3].set_title(f"Manual 128x128\n{final_manual.size}")
    axes[3].axis("off")

    # ---- FINAL (DATASET) ----
    axes[4].imshow(final_dataset)
    axes[4].set_title(f"Dataset\n{final_dataset.shape[:2]}")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(DEBUG_DIR / f"debug_pipeline_{idx}.png")
    plt.close()
"""
    
def visualize_all(original, cropped, final_manual, final_dataset, idx):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # ---- ORIGINAL ----
    axes[0].imshow(original)
    axes[0].set_title(f"Original\n{original.size}")
    axes[0].axis("off")

    # ---- CROPPED ----
    axes[1].imshow(cropped)
    axes[1].set_title(f"Cropped\n{cropped.size}")
    axes[1].axis("off")

    # ---- FINAL (MANUAL) ----
    axes[2].imshow(final_manual)
    axes[2].set_title(f"Manual 128x128\n{final_manual.size}")
    axes[2].axis("off")

    # ---- FINAL (DATASET) ----
    axes[3].imshow(final_dataset)
    axes[3].set_title(f"Dataset\n{final_dataset.shape[:2]}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(DEBUG_DIR / f"debug_pipeline_{idx}.png")
    plt.close()


def main():
    dataset = datasets.get_dataset("gtsrb")
    dataset.for_training()

    train_ds = dataset.train_data
    rows = train_ds.rows

    N = 10

    for i in range(N):
        row = rows.iloc[i]

        # ---- MANUAL PIPELINE ----
        original = load_original_image(row)
        cropped = crop_roi(original, row)
        #squared = make_square(cropped)
        #final_manual = resize_128(squared)
        final_manual = resize_128(cropped)

        # ---- DATASET OUTPUT ----
        features, _ = train_ds[i]
        final_dataset = tensor_to_image(features)

        # ---- DEBUG DIFFERENCE ----
        #diff = np.abs(final_manual - final_dataset).mean()
        final_manual_np = np.array(final_manual).astype(np.float32) / 255.00
        diff = np.abs(final_manual_np - final_dataset).mean()
        print(f"[{i}] Mean difference manual vs dataset:", diff)

        #visualize_all(original, cropped, squared, final_manual, final_dataset, i)
        visualize_all(original, cropped, final_manual, final_dataset, i)

    print(f"Saved {N} debug images with full pipeline (including squared)")


if __name__ == "__main__":
    main()