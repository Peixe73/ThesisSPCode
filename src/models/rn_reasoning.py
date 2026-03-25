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

    layers.append(nn.Linear(in_features, num_outputs))  # NO sigmoid

    return nn.Sequential(*layers)


# =========================
# DEBUG DUMP (FIXED)
# =========================
def dump_epoch_dataset(dataset, feature_cols, class_cols, path):
    rows = []

    for i in range(len(dataset)):
        x, y = dataset[i]

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        entry = {
            feature_cols[j]: float(x[j])
            for j in range(len(feature_cols))
        }

        entry["valid"] = float(y[0])

        for j, col in enumerate(class_cols):
            entry[col] = float(y[j + 1])

        rows.append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# =========================
# TRAINER
# =========================
def create_trainer(
        valid_path: str,
        feature_cols: list[str],
        class_cols: list[str],
        layer_sizes: list[int],
        dataset_size: int = 640,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 20
) -> Trainer:

    logger.info("Building dataset + model...")

    model_tag = f"L{layer_sizes}"
    debug_dir = f"debug_runs/{model_tag}"

    # datasets
    train_dataset = RandomReasoningDataset(
        valid_path=valid_path,
        feature_cols=feature_cols,
        class_cols=class_cols,
        dataset_size=dataset_size
    )

    val_dataset = RandomReasoningDataset(
        valid_path=valid_path,
        feature_cols=feature_cols,
        class_cols=class_cols,
        dataset_size=dataset_size // 4,
        seed=42  # fixed validation
    )

    split_dataset = SplitDataset(train_dataset, val_dataset)

    num_features = len(feature_cols)
    num_outputs = train_dataset.num_outputs

    # =========================
    # METRICS
    # =========================
    metrics_per_class = {
        "balanced_accuracy": metric_wrappers.to_int(
            core_metrics.BinaryBalancedAccuracy
        ),
    }

    """
    def metrics_factory():

        class EpochDebugTrigger:
            def reset(self): pass
            def update(self, *args, **kwargs): pass

            def compute(self):
                with EpochDebugState.lock:
                    epoch = EpochDebugState.epoch

                    path = f"{debug_dir}/epoch_{epoch}.csv"

                    dump_epoch_dataset(
                        dataset=train_dataset,  # ONLY TRAIN SET
                        feature_cols=feature_cols,
                        class_cols=class_cols,
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
            reduction="none",  # per class!
            out_dict=metrics
        )

        return metrics
    """
    """
    def metrics_factory():

        class MultiTaskBalancedAccuracy:
            def __init__(self):
                self.metrics = None
                self.num_outputs = None

            def reset(self):
                if self.metrics is not None:
                    for m in self.metrics:
                        m.reset()

            def update(self, y_pred, y_true):
                if self.metrics is None:
                    self.num_outputs = y_true.shape[1]
                    self.metrics = [
                        core_metrics.BinaryBalancedAccuracy()
                        for _ in range(self.num_outputs)
                    ]

                y_pred = torch.sigmoid(y_pred)

                for i, metric in enumerate(self.metrics):
                    metric.update(y_pred[:, i], y_true[:, i])

            def compute(self):
                results = {}

                # ✅ CRITICAL FIX
                if self.metrics is None:
                    # no data was seen → return NaNs but DO NOT crash
                    results["balanced_accuracy"] = torch.tensor(float("nan"))
                    return results

                values = []

                for i, metric in enumerate(self.metrics):
                    val = metric.compute()

                    if i == 0:
                        name = "balanced_accuracy_valid"
                    else:
                        class_name = class_cols[i - 1]
                        name = f"balanced_accuracy_{class_name}"

                    results[name] = val

                    if not torch.isnan(val):
                        values.append(val)

                # global average
                if values:
                    results["balanced_accuracy"] = torch.stack(values).mean()
                else:
                    results["balanced_accuracy"] = torch.tensor(float("nan"))

                return results

        class EpochDebugTrigger:
            def reset(self): pass
            def update(self, *args, **kwargs): pass

            def compute(self):
                with EpochDebugState.lock:
                    epoch = EpochDebugState.epoch

                    path = f"{debug_dir}/epoch_{epoch}.csv"

                    dump_epoch_dataset(
                        dataset=train_dataset,
                        feature_cols=feature_cols,
                        class_cols=class_cols,
                        path=path
                    )

                    logger.info(f"[DEBUG] epoch {epoch} → {path}")

                    EpochDebugState.epoch += 1

                return 0.0

        metrics = {
            "epoch_elapsed": Elapsed(),
            "epoch_debug": EpochDebugTrigger(),
            "multi_balanced_accuracy": MultiTaskBalancedAccuracy(),
        }

        return metrics
    """
    
    def metrics_factory():

        metrics = {
            "epoch_elapsed": Elapsed(),
        }

        # =========================
        # DEBUG
        # =========================
        class EpochDebugTrigger:
            def reset(self): pass
            def update(self, *args, **kwargs): pass

            def compute(self):
                with EpochDebugState.lock:
                    epoch = EpochDebugState.epoch

                    path = f"{debug_dir}/epoch_{epoch}.csv"

                    dump_epoch_dataset(
                        dataset=train_dataset,
                        feature_cols=feature_cols,
                        class_cols=class_cols,
                        path=path
                    )

                    logger.info(f"[DEBUG] epoch {epoch} → {path}")

                    EpochDebugState.epoch += 1

                return 0.0

        metrics["epoch_debug"] = EpochDebugTrigger()

        # =========================
        # SHARED STORAGE
        # =========================
        class MultiOutputBalancedAccuracy:
            def __init__(self):
                self.metrics = None

            def reset(self):
                if self.metrics is not None:
                    for m in self.metrics:
                        m.reset()

            def update(self, y_pred, y_true):
                if self.metrics is None:
                    num_outputs = y_true.shape[1]
                    self.metrics = [
                        core_metrics.BinaryBalancedAccuracy()
                        for _ in range(num_outputs)
                    ]

                y_pred = torch.sigmoid(y_pred)

                for i in range(len(self.metrics)):
                    self.metrics[i].update(y_pred[:, i], y_true[:, i])

            def compute(self):
                if self.metrics is None:
                    return None

                return [m.compute() for m in self.metrics]

        shared_metric = MultiOutputBalancedAccuracy()

        # wrapper to expose each column separately
        def make_column_metric(idx, name):

            class ColumnMetric:
                def reset(self):
                    pass  # handled globally

                def update(self, y_pred, y_true):
                    shared_metric.update(y_pred, y_true)

                def compute(self):
                    values = shared_metric.compute()
                    if values is None:
                        return torch.tensor(float("nan"))
                    return values[idx]

            return ColumnMetric()

        # valid
        metrics["balanced_accuracy_valid"] = make_column_metric(0, "valid")

        # classes
        for i, class_name in enumerate(class_cols):
            metrics[f"balanced_accuracy_{class_name}"] = make_column_metric(i + 1, class_name)

        # global metric
        class GlobalMetric:
            def reset(self):
                pass

            def update(self, y_pred, y_true):
                shared_metric.update(y_pred, y_true)

            def compute(self):
                values = shared_metric.compute()
                if values is None:
                    return torch.tensor(float("nan"))

                valid_values = [v for v in values if not torch.isnan(v)]

                if valid_values:
                    return torch.stack(valid_values).mean()
                else:
                    return torch.tensor(float("nan"))

        metrics["balanced_accuracy"] = GlobalMetric()

        return metrics

    train_metrics = TrainingRecorder(
        metric_functions=metrics_factory()
    )

    objective = Maximize("train", "balanced_accuracy", threshold=0.01)
    patience_objective = Minimize("train", "loss", threshold=0.001)

    trainer = Trainer(
        model=create_model(num_features, layer_sizes, num_outputs),
        loss_fn=nn.BCEWithLogitsLoss(),
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