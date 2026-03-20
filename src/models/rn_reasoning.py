import logging
import torch
from torch import nn

from analysis_tools.random_reasoning_dataset import RandomReasoningDataset

from core.training import Trainer
from core.training.metrics_recorder import TrainingRecorder
from core.training.checkpoint_triggers.best_metric import BestMetric
from core.training.stop_criteria import EarlyStop, GoalReached

from core.eval.metrics import Elapsed
from core.eval import metrics as core_metrics
from core.eval.metrics import metric_wrappers
from core.eval.objectives import Maximize, Minimize

from torch.utils.data import random_split

from core.datasets import SplitDataset

logger = logging.getLogger("RN_Reasoning")


# =========================
# MODEL
# =========================
def create_model(num_features: int, layer_sizes: list[int], num_outputs: int):
    layers = []
    in_features = num_features

    for size in layer_sizes:
        layers.append(nn.Linear(in_features, size))
        layers.append(nn.ReLU())
        in_features = size

    layers.append(nn.Linear(in_features, num_outputs))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


# =========================
# TRAINER ENTRYPOINT
# =========================
def create_trainer(
        valid_path: str,
        feature_cols: list[str],
        layer_sizes: list[int],
        dataset_size: int = 20000,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 20
) -> Trainer:

    logger.info("Building RandomReasoningDataset...")

    dataset = RandomReasoningDataset(
        valid_path=valid_path,
        feature_cols=feature_cols,
        dataset_size=dataset_size
    )

    num_features = len(feature_cols)
    num_outputs = dataset.num_outputs

    logger.info("Dataset size: %d", len(dataset))
    logger.info("Model: %d -> %s -> %d", num_features, layer_sizes, num_outputs)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    prev_device = torch.get_default_device()
    torch.set_default_device("cpu")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    torch.set_default_device(prev_device)

    split_dataset = SplitDataset(
        train_dataset,
        val_dataset
    )
    # =========================
    # METRICS (same system as old version)
    # =========================
    metrics_per_class = {
        "balanced_accuracy": metric_wrappers.to_int(
            core_metrics.BinaryBalancedAccuracy
        )
    }

    def metrics_factory():
        metrics = {
            "epoch_elapsed": Elapsed()
        }

        metric_wrappers.SelectCol.col_wise(
            split_dataset,
            metrics_per_class,
            reduction="min",
            out_dict=metrics
        )

        return metrics

    train_metrics = TrainingRecorder(
        metric_functions=metrics_factory()
    )

    # =========================
    # OBJECTIVES
    # =========================
    objective = Maximize("train", "balanced_accuracy", threshold=0.01)
    patience_objective = Minimize("train", "loss", threshold=0.001)

    # =========================
    # TRAINER
    # =========================
    return Trainer(
        model=create_model(num_features, layer_sizes, num_outputs),
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        training_set=dataset,
        batch_size=batch_size,

        metric_loggers=[train_metrics],

        objective=objective,

        stop_criteria=[
            EarlyStop(patience_objective, patience=patience),
            GoalReached(1.0)
        ],

        checkpoint_triggers=[
            BestMetric(objective)
        ],
    )