import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    # Load CSV
    df = pd.read_csv("storage/studies/gtsrb_hn_1/C2/metrics_val.csv")

    # Select class-specific columns
    ba_cols = [col for col in df.columns if col.startswith("balanced_accuracy_ClassId")]

    # Plot
    plt.figure(figsize=(14, 8))

    for col in ba_cols:
        plt.plot(df["epoch"], df[col], alpha=0.4)

    # Baseline
    plt.axhline(y=0.5, linestyle='--')

    # Formatting
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.title("Validation Balanced Accuracy per Class (Over Epochs)")
    plt.ylim(0, 1)

    plt.tight_layout()

    # Save
    plt.savefig("balanced_accuracy_barplot_hn_val_curves.png", dpi=300)

    plt.show()