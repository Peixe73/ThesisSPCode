
import json
import re
from pathlib import Path

import numpy as np


# Configuration

JSON_FILE = Path("storage/studies/gtsrb_rn_Multiple_HN_CfgWithCategoryInLoss/results.json")

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

    catA = []
    catB = []
    catC = []
    catD = []

    for run in runs:

        experiment_name = run["name"]
        experiment_data = run["data"]
        
        if "train" not in experiment_data:
            print(
                f"WARNING: {experiment_name} has no 'train' section."
            )
            continue
        
        train = experiment_data["train"] 

        """
        if "val" not in experiment_data:
            print(
                f"WARNING: {experiment_name} has no 'val' section."
            )
            continue

        val = experiment_data["val"]
        """

        # Training loss
        """
        if "loss" in train:
            losses.append(train["loss"])
        else:
            print(
                f"WARNING: {experiment_name} has no training loss."
            )
        """

        if "SignClass_A_balanced_accuracy" in train:
            catA.append(train["SignClass_A_balanced_accuracy"])

        if "SignClass_B_balanced_accuracy" in train:
            catB.append(
                train["SignClass_B_balanced_accuracy"]
            )

        if "SignClass_C_balanced_accuracy" in train:
                    catC.append(train["SignClass_C_balanced_accuracy"])
                    
        if "SignClass_D_balanced_accuracy" in train:
                    catD.append(train["SignClass_D_balanced_accuracy"])

    results[model_name] = {
        "n_runs": len(runs),
        "Category A": catA,
        "Category B": catB,
        "Category C": catC,
        "Category D": catD,
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
    r"Model & Category A $\pm$ S.D. & Category B $\pm$ S.D. "
    r"& Category C $\pm$ S.D. & Category D $\pm$ S.D. \\"
)

latex_lines.append(r"\midrule")


for model_name, result in results.items():

    catA = format_mean_std(
        result["Category A"],
        DECIMALS,
    )

    catB = format_mean_std(
        result["Category B"],
        DECIMALS,
    )

    catC = format_mean_std(
        result["Category C"],
        DECIMALS,
    )

    catD = format_mean_std(
        result["Category D"],
        DECIMALS,
    )

    latex_lines.append(
        f"{model_name} & "
        f"{catA} & "
        f"{catB} & "
        f"{catC} & "
        f"{catD} \\\\"
    )


latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")

latex_lines.append(
    r"\caption{Training set performance of categories balanced accuracy of the Reasoning Networks across multiple "
    r"independent runs. Results are reported as mean $\pm$ "
    r"standard deviation.}"
)

latex_lines.append(
    r"\label{tab:training_rn_results}"
)

latex_lines.append(r"\end{table}")


latex_table = "\n".join(latex_lines)


# Save LaTeX table

output_file = Path("training_results_table.tex")

with open(output_file, "w") as f:
    f.write(latex_table)


print("\n")
print(latex_table)

print(
    f"\nLaTeX table saved to: {output_file}"
)
