
import json
import re
from pathlib import Path

import numpy as np


# Configuration

JSON_FILE = Path("storage/studies/gtsrb_hn_EvalConfig_Multiple/results.json")

# Number of decimal places in the final table
DECIMALS = 4

# Use sample standard deviation (ddof=1)
DDOF = 1


# Helper functions

def mean_std(values, decimals=4):
    """
    Calculate mean and standard deviation while ignoring NaN values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    mean = np.mean(values)

    # If there is only one run, SD is undefined.
    if len(values) > 1:
        std = np.std(values, ddof=DDOF)
    else:
        std = np.nan

    return mean, std


def format_mean_std(values, decimals=4):
    """
    Return a string of the form:

        0.9535 ± 0.0041
    """
    mean, std = mean_std(values, decimals)

    if np.isnan(mean):
        return "--"

    if np.isnan(std):
        return f"{mean:.{decimals}f} $\\pm$ --"

    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def get_model_name(experiment_name):
    """
    Remove the final '_runN' from the experiment name.

    Example:
        C1_L0_ONTO_BASE_untRN_run0
        -> C1_L0_ONTO_BASE_untRN

        C1_L0_ONTO_BASE_untRN_run12
        -> C1_L0_ONTO_BASE_untRN
    """
    return re.sub(r"_run\d+$", "", experiment_name)


# Load JSON

with open(JSON_FILE, "r") as f:
    data = json.load(f)


# Ignore "best_experiment"
experiments = data["experiments"]


# Group experiments by model/configuration

grouped = {}

for experiment_name, experiment_data in experiments.items():

    model_name = get_model_name(experiment_name)

    if model_name not in grouped:
        grouped[model_name] = []

    grouped[model_name].append(
        {
            "name": experiment_name,
            "data": experiment_data,
        }
    )


# Extract validation metrics

results = {}

for model_name, runs in grouped.items():

    losses = []
    accuracies = []
    balanced_accuracies = []
    f1_scores = []

    for run in runs:

        experiment_name = run["name"]
        experiment_data = run["data"]
        
        if "train" not in experiment_data:
            print(
                f"WARNING: {experiment_name} has no 'train' section."
            )
            continue
        
        train = experiment_data["train"] 

        if "val" not in experiment_data:
            print(
                f"WARNING: {experiment_name} has no 'val' section."
            )
            continue

        val = experiment_data["val"]

        # Training loss
        if "loss" in train:
            losses.append(train["loss"])
        else:
            print(
                f"WARNING: {experiment_name} has no training loss."
            )

        # Validation accuracy
        if "accuracy" in val:
            accuracies.append(val["accuracy"])

        # Validation balanced accuracy
        if "balanced_accuracy" in val:
            balanced_accuracies.append(
                val["balanced_accuracy"]
            )

        # Validation Macro F1
        if "f1" in val:
            f1_scores.append(val["f1"])

    results[model_name] = {
        "n_runs": len(runs),
        "loss": losses,
        "accuracy": accuracies,
        "balanced_accuracy": balanced_accuracies,
        "f1": f1_scores,
    }


# Print summary

print("\nNumber of runs per model:")
print("-" * 50)

for model_name, result in results.items():
    print(
        f"{model_name}: "
        f"{result['n_runs']} runs"
    )


# Generate LaTeX table

latex_lines = []

latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\begin{tabular}{l c c c c}")
latex_lines.append(r"\toprule")

latex_lines.append(
    r"Model & Loss $\pm$ S.D. & Accuracy $\pm$ S.D. "
    r"& Balanced Acc. $\pm$ S.D. & Macro F1 $\pm$ S.D. \\"
)

latex_lines.append(r"\midrule")


for model_name, result in results.items():

    loss = format_mean_std(
        result["loss"],
        DECIMALS,
    )

    accuracy = format_mean_std(
        result["accuracy"],
        DECIMALS,
    )

    balanced_accuracy = format_mean_std(
        result["balanced_accuracy"],
        DECIMALS,
    )

    f1 = format_mean_std(
        result["f1"],
        DECIMALS,
    )

    latex_lines.append(
        f"{model_name} & "
        f"{loss} & "
        f"{accuracy} & "
        f"{balanced_accuracy} & "
        f"{f1} \\\\"
    )


latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")

latex_lines.append(
    r"\caption{Validation-set performance across multiple "
    r"independent runs. Results are reported as mean $\pm$ "
    r"standard deviation.}"
)

latex_lines.append(
    r"\label{tab:validation_results}"
)

latex_lines.append(r"\end{table}")


latex_table = "\n".join(latex_lines)


# Save LaTeX table

output_file = Path("validation_results_table.tex")

with open(output_file, "w") as f:
    f.write(latex_table)


print("\n")
print(latex_table)

print(
    f"\nLaTeX table saved to: {output_file}"
)
