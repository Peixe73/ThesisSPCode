import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    # Load CSV
    df = pd.read_csv("storage/studies/gtsrb_hn_1/C2/metrics_val.csv")

    # Select balanced accuracy columns (exclude global one if you want)
    ba_cols = [col for col in df.columns if col.startswith("balanced_accuracy_ClassId")]

    # Use last epoch
    values = df.iloc[-1][ba_cols]

    # Sort values
    values = values.sort_values(ascending=False)

    # Clean labels
    labels = [col.replace("balanced_accuracy_", "") for col in values.index]

    # Highlight underperforming (< 0.5)
    colors = ['red' if v < 0.5 else 'blue' for v in values]

    # Plot
    plt.figure(figsize=(16, 7))
    plt.bar(labels, values, color=colors)

    # Baseline
    plt.axhline(y=0.5, linestyle='--')

    # Formatting
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Balanced Accuracy")
    plt.title("Validation Balanced Accuracy per Class (Last Epoch)")

    plt.tight_layout()

    # Save
    plt.savefig("balanced_accuracy_barplot_hn_val.png", dpi=300)

    plt.show()