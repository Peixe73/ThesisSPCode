from pathlib import Path
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field

import numpy as np

from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import option, positional

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from analysis_tools.gtsrb_utils import SHORT_CONCEPTS
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class Options:
    results_path: Path = field(
        metadata=positional(
            Path,
            help_="Path to the directory containing the results."
        )
    )

    target_csv: Path = field(
        metadata=positional(
            Path,
            help_="CSV file containing the concept/neuron scores."
        )
    )

    runs: int = field(
        default=100,
        metadata=option(
            int,
            help_="Number of random assignments to evaluate."
        )
    )

    seed: Optional[int] = field(
        default=100,
        metadata=option(
            int,
            help_="Random seed for reproducibility."
        )
    )

    mid_point: Optional[float] = field(
        default=None,
        metadata=option(
            float,
            help_="Mid point used for determining negation. "
                  "Defaults to 0.5 for accuracy and 0.0 for correlation."
        )
    )


def determine_mid_point(csv_file: Path) -> float:
    if 'accuracy' in csv_file.name:
        return 0.5
    elif 'correlation' in csv_file.name:
        return 0.0
    else:
        raise ValueError(
            f"Cannot determine mid point for {csv_file}"
        )


def main(options: Options):
    if options.mid_point is None:
        mid_point = determine_mid_point(options.target_csv)
    else:
        mid_point = options.mid_point

    training_file = options.results_path.joinpath(
        'train',
        options.target_csv
    )
    validation_file = options.results_path.joinpath(
        'val',
        options.target_csv
    )

    training_results = pd.read_csv(
        training_file,
        index_col=0
    )

    validation_results = pd.read_csv(
        validation_file,
        index_col=0
    )

    concepts = SHORT_CONCEPTS

    concepts = [
        concept
        for concept in concepts
        if concept in validation_results.columns
    ]

    neurons = list(training_results.index)

    if len(neurons) < len(concepts):
        raise ValueError(
            f"Cannot construct a one-to-one assignment: "
            f"{len(neurons)} neurons but {len(concepts)} concepts."
        )

    if len(neurons) != len(concepts):
        logger.warning(
            "There are %d neurons and %d concepts. "
            "Only %d neurons will be used in each random assignment.",
            len(neurons),
            len(concepts),
            len(concepts),
        )

    rng = np.random.default_rng(options.seed)

    training_runs = pd.DataFrame(
        index=range(options.runs),
        columns=concepts,
        dtype=float,
    )

    validation_runs = pd.DataFrame(
        index=range(options.runs),
        columns=concepts,
        dtype=float,
    )

    neuron_runs = pd.DataFrame(
        index=range(options.runs),
        columns=concepts,
        dtype=object,
    )

    for run in range(options.runs):

        # Randomly choose the neurons used in this assignment.
        selected_neurons = rng.choice(
            neurons,
            size=len(concepts),
            replace=False,
        )

        # Randomly permute them across concepts.
        shuffled_neurons = rng.permutation(selected_neurons)

        for concept, neuron in zip(concepts, shuffled_neurons):

            train_result = training_results.loc[neuron, concept]
            validation_result = validation_results.loc[neuron, concept]
            
            """
            if train_result < mid_point:
                train_result = (
                    2 * mid_point - train_result
                )

                validation_result = (
                    2 * mid_point - validation_result
                )
            """

            training_runs.loc[run, concept] = train_result
            validation_runs.loc[run, concept] = validation_result
            neuron_runs.loc[run, concept] = neuron

    summary = pd.DataFrame(
        index=concepts,
        columns=[
            'Training Mean',
            'Training Std',
            'Validation Mean',
            'Validation Std',
        ],
        dtype=float,
    )

    for concept in concepts:
        summary.loc[concept, 'Training Mean'] = (
            training_runs[concept].mean()
        )

        summary.loc[concept, 'Training Std'] = (
            training_runs[concept].std(ddof=1)
        )

        summary.loc[concept, 'Validation Mean'] = (
            validation_runs[concept].mean()
        )

        summary.loc[concept, 'Validation Std'] = (
            validation_runs[concept].std(ddof=1)
        )
        
    training_run_means = training_runs.mean(axis=1)
    validation_run_means = validation_runs.mean(axis=1)

    summary.loc['Mean', 'Training Mean'] = (
        training_run_means.mean()
    )
    summary.loc['Mean', 'Training Std'] = (
        training_run_means.std(ddof=1)
    )
    summary.loc['Mean', 'Validation Mean'] = (
        validation_run_means.mean()
    )
    summary.loc['Mean', 'Validation Std'] = (
        validation_run_means.std(ddof=1)
    )

    base_name = options.target_csv.with_suffix('').name

    summary_path = options.results_path.joinpath(
        base_name + '_random_summary.csv'
    )

    summary.to_csv(summary_path)
    
    training_runs_path = options.results_path.joinpath(
        base_name + '_random_training_runs.csv'
    )

    validation_runs_path = options.results_path.joinpath(
        base_name + '_random_validation_runs.csv'
    )

    neuron_runs_path = options.results_path.joinpath(
        base_name + '_random_neuron_assignments.csv'
    )

    training_runs.to_csv(training_runs_path)
    validation_runs.to_csv(validation_runs_path)
    neuron_runs.to_csv(neuron_runs_path)

    latex_path = options.results_path.joinpath(
        base_name + '_random_summary.tex'
    )

    summary.to_latex(
        latex_path,
        formatters={
            'Training Mean': lambda x: f"{x:.4f}",
            'Training Std': lambda x: f"{x:.4f}",
            'Validation Mean': lambda x: f"{x:.4f}",
            'Validation Std': lambda x: f"{x:.4f}",
        },
        na_rep='',
    )

    logger.info(
        "Random baseline completed with %d runs.",
        options.runs
    )
    
    latex_summary = pd.DataFrame(index=concepts)

    latex_summary['Train.'] = (
        summary['Training Mean'].map(lambda x: f"{x:.4f}")
        + r" $\pm$ "
        + summary['Training Std'].map(lambda x: f"{x:.4f}")
    )

    latex_summary['Val.'] = (
        summary['Validation Mean'].map(lambda x: f"{x:.4f}")
        + r" $\pm$ "
        + summary['Validation Std'].map(lambda x: f"{x:.4f}")
    )

    # Mean row
    latex_summary.loc['Mean', 'Train.'] = (
        f"{summary.loc['Mean', 'Training Mean']:.4f}"
        + r" $\pm$ "
        + f"{summary.loc['Mean', 'Training Std']:.4f}"
    )

    latex_summary.loc['Mean', 'Val.'] = (
        f"{summary.loc['Mean', 'Validation Mean']:.4f}"
        + r" $\pm$ "
        + f"{summary.loc['Mean', 'Validation Std']:.4f}"
    )
    
    latex_summary.to_latex(
    latex_path,
    escape=False,
    column_format='@{}lrr@{}',
    caption=(
        'Random one-to-one concept attribution for '
        'model with intermediate classification '
        'supervision. Values are reported as mean $\\pm$ '
        'standard deviation over 100 random one-to-one assignments.'
    ),
    label='tab:app_model_intermediate_random_alignment',
    na_rep='',
    longtable=True,
)

    logger.info("Summary saved to %s", summary_path)
    logger.info("Training runs saved to %s", training_runs_path)
    logger.info("Validation runs saved to %s", validation_runs_path)
    logger.info("Neuron assignments saved to %s", neuron_runs_path)