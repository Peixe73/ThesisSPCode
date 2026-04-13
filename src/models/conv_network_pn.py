import logging
from typing import Optional, Sequence, Literal, assert_never

from core.training.checkpoint_triggers.best_metric import BestMetric
import torch
from torch import nn

from core.util import conv_out_shape
from core import datasets
from core.training import Trainer, TrainingRecorder
from core.eval.objectives import Maximize
from core.eval.metrics import Elapsed, metric_wrappers
import core.eval.metrics

logger = logging.getLogger(__name__)


def make_model(
        input_shape: Sequence[int],
        conv_layers: Sequence[int | tuple[Literal['pool'], int]],
        linear_layers: Sequence[int],
        num_outputs: int,
        hidden_activations: Literal['relu'] | tuple[Literal['leaky_relu'], float] = 'relu',
        dropout_last_layer : Optional[float] = None,
        kernel_size: int = 3
):
    logger.debug(f"Creating convolutional network...")
    same_padding = (kernel_size - 1) // 2
    layers: list[nn.Module] = []

    def hidden_activation():
        match hidden_activations:
            case 'relu':
                return nn.ReLU()
            case ('leaky_relu', alpha):
                return nn.LeakyReLU(alpha)
            case never:
                assert_never(never)

    in_channels = input_shape[0]
    in_shape = input_shape[1:]
    logger.debug(f"\t{in_channels} x {in_shape} -> ")
    for conv_layer in conv_layers:
        match conv_layer:
            case ('pool', pool_ksize):
                layers.append(nn.MaxPool2d(pool_ksize, padding=0))
                in_shape = conv_out_shape(in_shape, pool_ksize, padding=0, stride=pool_ksize)
                logger.debug(f"\tpool {pool_ksize}")
            case out_channels:
                assert isinstance(out_channels, int)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=same_padding))
                layers.append(hidden_activation())
                in_channels = out_channels
                logger.debug(f"\tconv")
        logger.debug(f"\t{in_channels} x {in_shape} -> ")

    last_conv_total_features = in_channels
    for dim_size in in_shape:
        last_conv_total_features *= dim_size
    in_features = last_conv_total_features
    layers.append(nn.Flatten())
    logger.debug(f"\tFlatten {in_features} ->")

    for out_features in linear_layers:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(hidden_activation())
        in_features = out_features
        logger.debug(f"\tLinear {out_features} ->")

    logger.debug(f"\tOutput {num_outputs}->")
    layers.append(nn.Linear(in_features, num_outputs))
    #If using BCEWithLogitsLoss, we should not apply sigmoid here, as it is included in the loss function.
    # If we apply it here, it will cause issues with the loss function and metrics.
    layers.append(nn.Sigmoid())
    if dropout_last_layer is not None:
        layers.append(nn.Dropout(dropout_last_layer))

    return nn.Sequential(*layers)


class GlobalBalancedAccuracy:
    def __init__(self):
        self.metric = core.eval.metrics.BinaryBalancedAccuracy()

    def reset(self):
        self.metric.reset()

    def update(self, y_pred, y_true):
        # apply threshold
        y_pred = (y_pred > 0.5).int()

        # flatten all concepts into one vector
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        self.metric.update(y_pred, y_true)

    def compute(self):
        return self.metric.compute()

# TRAINER WITH METRICS
def create_trainer(dataset_name: str, **kwargs) -> Trainer:

    dataset = datasets.get_dataset(dataset_name)
    input_shape = dataset.get_shape()[0]

    # Better loss for multi-label, supposedly
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()

    # METRICS
    metrics_per_class = {
        "balanced_accuracy": metric_wrappers.to_int(
            core.eval.metrics.BinaryBalancedAccuracy
        )
    }

    def metrics_factory():
        metrics = {
            "epoch_elapsed": Elapsed(),
            #"balanced_accuracy": GlobalBalancedAccuracy(),
        }

        # applies per concept 
        metric_wrappers.SelectCol.col_wise(
            dataset,
            metrics_per_class,
            reduction="min",  # global score
            out_dict=metrics
        )

        return metrics

    train_metrics = TrainingRecorder(metric_functions=metrics_factory())
    
    objective = Maximize("train", "balanced_accuracy", threshold=0.01)

    # TRAINER
    trainer = Trainer(
        model=make_model(input_shape, **kwargs),
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam,
        training_set=dataset,
        batch_size=64,
        metric_loggers=[train_metrics],
        objective=objective,
        checkpoint_triggers=[BestMetric(objective)],
    )

    return trainer