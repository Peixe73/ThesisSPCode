import json
import csv
import os
import torch

def export_dataset_debug(dataset, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 1. COLLECT DATA
    # -----------------------------
    samples = []
    labels = []

    for i in range(len(dataset)):
        x, y = dataset[i]

        x_flat = x.flatten().tolist() if torch.is_tensor(x) else list(x)

        samples.append((i, x_flat, y))
        labels.append(y)

    # -----------------------------
    # 2. STATS
    # -----------------------------
    class_counts = {}
    for y in labels:
        class_counts[str(y)] = class_counts.get(str(y), 0) + 1

    stats = {
        "num_samples": len(dataset),
        "class_distribution": class_counts,
    }

    # -----------------------------
    # 3. WRITE JSON STATS
    # -----------------------------
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # -----------------------------
    # 4. WRITE CSV OF SAMPLES
    # -----------------------------
    with open(os.path.join(output_dir, "dataset_samples.csv"), "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["index", "features", "label"])

        for i, x, y in samples:
            writer.writerow([i, x, y])

    # -----------------------------
    # 5. WRITE HUMAN REPORT
    # -----------------------------
    with open(os.path.join(output_dir, "dataset_report.txt"), "w") as f:
        f.write("=== DATASET REPORT ===\n")
        f.write(f"Samples: {len(dataset)}\n")
        f.write(f"Class distribution: {class_counts}\n")