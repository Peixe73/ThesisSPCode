import logging
import torch
from torch import nn
import os
import pandas as pd
import threading

from analysis_tools.random_reasoning_dataset import RandomReasoningDataset

from core.training import Trainer
from core.training.metrics_recorder import TrainingRecorder
from core.training.checkpoint_triggers.best_metric import BestMetric
from core.training.stop_criteria import EarlyStop, GoalReached

from core.eval.metrics import Elapsed
from core.eval import metrics as core_metrics
from core.eval.metrics import metric_wrappers
from core.eval.objectives import Maximize, Minimize

from core.datasets import SplitDataset

logger = logging.getLogger("RN_Reasoning")


# =========================
# GLOBAL DEBUG STATE
# =========================
class EpochDebugState:
    epoch = 0
    lock = threading.Lock()


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
# DEBUG DUMP
# =========================
def dump_epoch_dataset(dataset, feature_cols, path):
    rows = []

    for i in range(len(dataset)):
        x, y = dataset[i]

        x = x.detach().cpu()
        y = y.detach().cpu()

        row = x.numpy()

        entry = {
            feature_cols[j]: float(row[j])
            for j in range(len(feature_cols))
        }
        entry["label"] = float(y.item())
        rows.append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    pd.DataFrame(rows).to_csv(path, index=False)


# =========================
# TRAINER
# =========================
def create_trainer(
        valid_path: str,
        feature_cols: list[str],
        layer_sizes: list[int],
        classid_rules: str,
        dataset_size: int = 640,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 20
) -> Trainer:

    logger.info("Building dataset + model...")

    model_tag = f"L{layer_sizes}"
    debug_dir = f"debug_runs/{model_tag}"

    # =========================
    # FORCE CPU CONTEXT BEFORE DATASET SPLIT
    # =========================
    device_backup = torch.get_default_device() if hasattr(torch, "get_default_device") else None

    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")

        dataset = RandomReasoningDataset(
            valid_path=valid_path,
            feature_cols=feature_cols,
            dataset_size=dataset_size,
            debug=True
        )

        num_features = len(feature_cols)
        num_outputs = dataset.num_outputs

        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_len, val_len]
        )

    finally:
        if device_backup is not None and hasattr(torch, "set_default_device"):
            torch.set_default_device(device_backup)

    split_dataset = SplitDataset(train_dataset, val_dataset)

    # =========================
    # METRICS
    # =========================
    metrics_per_class = {
        "balanced_accuracy": metric_wrappers.to_int(
            core_metrics.BinaryBalancedAccuracy
        ),
    }

    def metrics_factory():

        class EpochDebugTrigger:
            def reset(self): pass
            def update(self, *args, **kwargs): pass

            def compute(self):
                with EpochDebugState.lock:
                    epoch = EpochDebugState.epoch

                    path = f"{debug_dir}/epoch_{epoch}.csv"

                    dump_epoch_dataset(
                        dataset=dataset,
                        feature_cols=feature_cols,
                        path=path
                    )

                    logger.info(f"[DEBUG] epoch {epoch} → {path}")

                    EpochDebugState.epoch += 1

                return 0.0

        metrics = {
            "epoch_elapsed": Elapsed(),
            "epoch_debug": EpochDebugTrigger(),
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

    objective = Maximize("train", "balanced_accuracy", threshold=0.01)
    patience_objective = Minimize("train", "loss", threshold=0.001)

    trainer = Trainer(
        model=create_model(num_features, layer_sizes, num_outputs),
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,

        training_set=split_dataset,
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

    return trainer