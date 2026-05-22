from pathlib import Path
import torch
import matplotlib.pyplot as plt

from core.datasets import get_dataset
from analysis_tools.gtsrb_utils import CLASS_COLS


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "gtsrb"
OUTPUT_PATH = Path("storage/analysis_results/gtsrb")


# ============================================================
# COUNT CLASS DISTRIBUTIONS
# ============================================================

def extract_class_counts(loader, class_names):
    """
    Counts how many samples belong to each class.
    Assumes y is multi-hot or one-hot tensor: (batch, num_classes)
    """
    counts = {c: 0 for c in class_names}

    for _, y in loader:
        # FORCE SAFE CPU TENSOR
        y = y.detach().cpu()

        # convert to boolean mask
        positive = (y > 0)

        # sum per class in batch -> (num_classes,)
        batch_counts = positive.sum(dim=0).tolist()

        # accumulate into dict
        for i, class_name in enumerate(class_names):
            counts[class_name] += int(batch_counts[i])

    return counts


# ============================================================
# PLOTTING
# ============================================================

def plot_counts(counts, title, save_path: Path):
    class_ids = list(counts.keys())
    values = [counts[c] for c in class_ids]

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 6))
    plt.bar(class_ids, values)

    plt.xticks(class_ids, rotation=90)
    plt.xlabel("Class ID")
    plt.ylabel("Number of samples")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_train_val(train_counts, val_counts, save_path: Path):
    class_ids = list(train_counts.keys())

    train_values = [train_counts[c] for c in class_ids]
    val_values = [val_counts[c] for c in class_ids]

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 6))

    plt.bar(class_ids, train_values, alpha=0.6, label="Train")
    plt.bar(class_ids, val_values, alpha=0.6, label="Validation")

    plt.xticks(class_ids, rotation=90)
    plt.xlabel("Class ID")
    plt.ylabel("Number of samples")
    plt.title("Train vs Validation Class Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# DATA LOADER WRAPPER
# ============================================================

def make_loader(dataset, batch_size=64):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )


# ============================================================
# MAIN
# ============================================================

def main():
    dataset = get_dataset(DATASET_NAME)

    # Ensure dataset is loaded
    if not dataset.loaded:
        dataset._load()

    train_loader = make_loader(dataset.train_data)
    val_loader = make_loader(dataset.val_data)
    test_loader = make_loader(dataset.test_data) if dataset.test_data is not None else None

    class_names = CLASS_COLS

    print(f"Number of classes: {len(class_names)}")

    # ========================================================
    # COMPUTE DISTRIBUTIONS
    # ========================================================

    train_counts = extract_class_counts(train_loader, class_names)
    val_counts = extract_class_counts(val_loader, class_names)

    test_counts = (
        extract_class_counts(test_loader, class_names)
        if test_loader is not None
        else None
    )

    # ========================================================
    # PLOTS
    # ========================================================

    plot_counts(
        train_counts,
        "Train Set Class Distribution",
        OUTPUT_PATH / "train_distribution.png"
    )

    plot_counts(
        val_counts,
        "Validation Set Class Distribution",
        OUTPUT_PATH / "val_distribution.png"
    )

    if test_counts is not None:
        plot_counts(
            test_counts,
            "Test Set Class Distribution",
            OUTPUT_PATH / "test_distribution.png"
        )

    # Combined plot
    plot_train_val(
        train_counts,
        val_counts,
        OUTPUT_PATH / "train_val_distribution.png"
    )

    print(f"Saved plots to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()