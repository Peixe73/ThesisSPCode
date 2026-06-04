from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from torcheval.metrics import BinaryRecall
from datetime import timedelta

from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import option

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    import torch
    import logging
    from torch.utils.data import DataLoader

    from core.datasets import get_dataset, dataset_wrappers
    from core.eval.metrics import (
        BinaryBalancedAccuracy,
        BinarySpecificity,
        PearsonCorrelationCoefficient,
        metric_wrappers,
    )
    from core.eval.metrics_crosser import MetricCrosser
    from analysis_tools.datasets import analyze_dataset
    from analysis_tools.gtsrb_utils import CLASS_COLS #CONCEPTS, SIGNCLASS_BINARY,
    from core.util.progress_trackers import LogProgressContextManager

    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))


# ============================================================
# OPTIONS
# ============================================================

@dataclass
class Options:
    dataset_name: str = field(
        #default="gtsrb_with_concepts",
        default="gtsrb",
        metadata=option(str, help_="Name of the dataset to evaluate"),
    )
    batch_size: int = field(
        default=64,
        metadata=option(int, help_="Batch size for data loaders"),
    )
    destination: Path = field(
        default=Path("storage/analysis_results"),
        metadata=option(Path, help_="Destination for analysis results"),
    )


# ============================================================
# CORE EVALUATION FUNCTION
# ============================================================

def cross_classes(
    crosser: "MetricCrosser",
    loader: "DataLoader",
    destination: "Path",
):
    destination.mkdir(parents=True, exist_ok=True)
    crosser.reset()

    with progress_cm.track("Cross class evaluation", "batches", loader) as progress_tracker:
        for _, y in loader:
            crosser.update(y, y)
            progress_tracker.tick(y.shape[0])

    logger.info("Computing cross class metrics...")

    results = crosser.compute()
    for k, v in results.items():
        v.to_csv(destination / f"{k}_correlation.csv")


# ============================================================
# MAIN
# ============================================================

def main(options: Options):

    def make_loader(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=False,
        )

    destination = options.destination / options.dataset_name
    destination.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(options.dataset_name)

    # Speed-up: disable image loading
    if hasattr(dataset, "skip_image_loading"):
        dataset.skip_image_loading = True  # type: ignore

    # ========================================================
    # TARGET SELECTION
    # ========================================================

    #target_hist_columns = CLASS_COLS + CONCEPTS #CONCEPTS + SIGNCLASS_BINARY
    target_hist_columns = CLASS_COLS

    label_indices = dataset.get_column_references().get_label_indices(
        target_hist_columns
    )

    selected_dataset = dataset_wrappers.SelectCols(
        dataset,
        select_y=label_indices,
    )

    logger.info(f"Evaluating targets: {target_hist_columns}")

    # ========================================================
    # METRIC CROSSES
    # ========================================================

    def build_crosser():
        return MetricCrosser(
            target_hist_columns,
            target_hist_columns,
            {
                "correlation": PearsonCorrelationCoefficient,
                "balanced_accuracy": lambda: metric_wrappers.ToDtype(
                    BinaryBalancedAccuracy(), torch.int32, apply_to_pred=False
                ),
                "recall": lambda: metric_wrappers.ToDtype(
                    BinaryRecall(), torch.int32, apply_to_pred=False
                ),
                "specificity": lambda: metric_wrappers.ToDtype(
                    BinarySpecificity(), torch.int32, apply_to_pred=False
                ),
            },
        )

    # ========================================================
    # TRAIN / VAL
    # ========================================================

    logger.info("On training set")
    cross_classes(
        build_crosser(),
        make_loader(selected_dataset.for_training()),
        destination / "train",
    )

    logger.info("On validation set")
    cross_classes(
        build_crosser(),
        make_loader(selected_dataset.for_validation()),
        destination / "val",
    )

    analyze_dataset(
        make_loader(selected_dataset.for_training()),
        destination,
        "train",
        target_hist_columns,
    )

    analyze_dataset(
        make_loader(selected_dataset.for_validation()),
        destination,
        "val",
        target_hist_columns,
    )

    # ========================================================
    # TEST SET (if available)
    # ========================================================
    '''
    if get_dataset(options.dataset_name).name == "gtsrb_with_concepts":

        logger.info("On official test set")

        test_dataset = get_dataset(options.dataset_name + "_test")

        if hasattr(test_dataset, "skip_image_loading"):
            test_dataset.skip_image_loading = True  # type: ignore

        test_selected_dataset = dataset_wrappers.SelectCols(
            test_dataset,
            select_y=label_indices,
        )

        cross_classes(
            build_crosser(),
            make_loader(test_selected_dataset.for_validation()),
            destination / "test",
        )

        analyze_dataset(
            make_loader(test_selected_dataset.for_validation()),
            destination,
            "test",
            target_hist_columns,
        )
    '''