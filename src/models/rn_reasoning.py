import logging
from pathlib import Path
from torch import nn
import torch

from core.datasets.csv_dataset import CSVDataset
from core.training import Trainer, TrainingRecorder
from core.training.checkpoint_triggers.best_metric import BestMetric
from core.training.stop_criteria import EarlyStop, GoalReached
from core.eval.metrics import Elapsed, metric_wrappers
from core.eval.objectives import Maximize, Minimize
import core.eval.metrics

from analysis_tools.random_reasoning_dataset import RandomReasoningDataset

logger = logging.getLogger("RN_Reasoning")

DEBUG_DIR = Path("reasoning_csvs")
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

"""
def create_model(layer_sizes: list[int], num_outputs: int) -> nn.Module:
    layers = []
    for size in layer_sizes:
        layers.append(nn.LazyLinear(size))
        layers.append(nn.ReLU())

    layers.append(nn.LazyLinear(num_outputs))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)
"""

def create_model(input_size: int, layer_sizes: list[int], num_outputs: int) -> nn.Module:
    layers = []

    in_features = input_size

    for size in layer_sizes:
        layers.append(nn.Linear(in_features, size))
        layers.append(nn.ReLU())
        in_features = size

    layers.append(nn.Linear(in_features, num_outputs))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)

class MaskedBCELoss(nn.Module):
    """
    - valid head: always trained
    - class heads: ONLY trained when valid == 1
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, y_pred, y_true):
        # y_pred, y_true shape: [B, 1 + num_classes]
        
        valid_pred = y_pred[:, -1]
        valid_true = y_true[:, -1]

        class_pred = y_pred[:, :-1]
        class_true = y_true[:, :-1]

        # valid loss (always)
        valid_loss = self.bce(valid_pred, valid_true)

        # class loss (masked)
        class_loss = self.bce(class_pred, class_true)

        # mask → only valid samples contribute
        mask = valid_true.unsqueeze(1)

        masked_class_loss = class_loss * mask

        # average only over active class terms
        class_count = mask.sum() * class_pred.shape[1]

        if class_count > 0:
            class_loss_mean = masked_class_loss.sum() / class_count
        else:
            #class_loss_mean = 0.0
            class_loss_mean = torch.tensor(
                0.0,
                device=y_pred.device
            )

        valid_loss_mean = valid_loss.mean()

        return valid_loss_mean + class_loss_mean

        '''
        valid_pred = y_pred[:, 0]
        valid_true = y_true[:, 0]

        class_pred = y_pred[:, 1:]
        class_true = y_true[:, 1:]

        # valid loss (always)
        valid_loss = self.bce(valid_pred, valid_true)

        # class loss (masked)
        class_loss = self.bce(class_pred, class_true)

        # mask → only valid samples contribute
        mask = valid_true.unsqueeze(1)  # [B,1]
        class_loss = class_loss * mask

        # mean over everything
        total_loss = torch.cat([valid_loss.unsqueeze(1), class_loss], dim=1)

        return total_loss.mean()
        '''


class EpochDatasetUpdater:
    def __init__(self, valid_path, feature_cols, class_cols, dataset_size, base_seed):
        self.valid_path = valid_path
        self.feature_cols = feature_cols
        self.class_cols = class_cols
        self.dataset_size = dataset_size
        self.base_seed = base_seed
        self.epoch = 0

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        path = DEBUG_DIR / "train.csv"

        epoch_seed = self.base_seed * 1000003 + self.epoch # large prime to ensure different seeds across epochs and large differences between seeds

        ds = RandomReasoningDataset(
            self.valid_path,
            self.feature_cols,
            self.class_cols,
            self.dataset_size,
            seed=epoch_seed
        )
        ds.to_csv(path)

        logger.info(f"[DATASET] Epoch {self.epoch} | Seed {epoch_seed}")

        self.epoch += 1
        return 0.0


def create_trainer(
    valid_path: str,
    feature_cols: list[str],
    class_cols: list[str],
    layer_sizes: list[int],
    dataset_size: int = 6400,
    batch_size: int = 64,
    patience: int = 20,
    base_seed: int = 42,
) -> Trainer:

    train_csv = DEBUG_DIR / "train.csv"
    
    initial_seed = base_seed * 1000003 + 0

    init_ds = RandomReasoningDataset(
        valid_path,
        feature_cols,
        class_cols,
        dataset_size,
        seed=initial_seed
    )

    init_ds.to_csv(train_csv)

    train_dataset = CSVDataset(
        path=train_csv,
        features=feature_cols,
        #target=["valid"] + class_cols
        target= class_cols + ["valid"]
    )

    metrics_per_class = {
        "balanced_accuracy": metric_wrappers.to_int(core.eval.metrics.BinaryBalancedAccuracy)
    }
    
    dataset_updater = EpochDatasetUpdater(
        valid_path,
        feature_cols,
        class_cols,
        dataset_size,
        base_seed
    )

    def metrics_factory():
        metrics = {
            "epoch_elapsed": Elapsed(),
            "dataset_update": dataset_updater, # this will regenerate the dataset at the end of each epoch
        }

        metric_wrappers.SelectCol.col_wise(
            train_dataset,
            metrics_per_class,
            reduction="min",   # gives a global "balanced_accuracy"
            out_dict=metrics
        )

        return metrics

    train_metrics = TrainingRecorder(
        metric_functions=metrics_factory()
    )

    objective = Maximize("train", "balanced_accuracy", threshold=0.01)
    patience_objective = Minimize("train", "loss", threshold=0.001)


    trainer = Trainer(
        model=create_model(38, layer_sizes, num_outputs=1 + len(class_cols)),
        loss_fn=MaskedBCELoss(),
        optimizer=torch.optim.Adam,
        training_set=train_dataset,
        batch_size=batch_size,
        metric_loggers=[train_metrics],
        objective=objective,
        stop_criteria=[
            EarlyStop(patience_objective, patience=patience),
            GoalReached(1.0)
        ],
        checkpoint_triggers=[BestMetric(objective)],
    )

    return trainer