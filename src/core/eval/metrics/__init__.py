import logging
import torch
from typing import Any, Optional, Sequence
import torch
import inspect
from torcheval.metrics import functional as torch_metrics
from torcheval.metrics import Metric, BinaryConfusionMatrix
import torcheval
from torcheval.metrics.classification.confusion_matrix import TBinaryConfusionMatrix
from torch import Tensor

from core.datasets import SplitDataset
from .elapsed import Elapsed
from typing import Callable

MetricFunction = Callable[[torch.Tensor, torch.Tensor], float] | Metric
NamedMetricFunction = str | tuple[str, MetricFunction]

class DecoratedTorchMetric:

    def __init__(
            self, 
            metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            flatten_tensors : bool = False
        ) -> None:
        self.metric = metric
        self.flatten_tensors = flatten_tensors

    def __call__(self, y_pred, y_true, **kwargs):
        if self.flatten_tensors:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        return self.metric(y_pred, y_true).item()
    
    def __repr__(self):
        return f"DecoratedTorchMetric({self.metric})"

__all_metrics : dict[str, MetricFunction | type] = {
    'epoch_elapsed' : Elapsed
}


def get_metric(name : str) -> MetricFunction | None:
    if name in __all_metrics:
        metric = __all_metrics[name]
        if inspect.isclass(metric):
            metric = metric()
        return metric
    else:
        torch_metric = torch_metrics.__dict__.get(name)
        if torch_metric is not None:
            return DecoratedTorchMetric(torch_metric)
        else:
            return None

def metric_exists(name):
    return get_metric(name) is not None

def select_metrics(metrics : Sequence[NamedMetricFunction], dataset : Optional[SplitDataset] = None) -> dict[str, MetricFunction]:
    if dataset is not None:
        # TODO metrics logger uses this function but doesn't know the dataset
        raise NotImplementedError("Dataset specific metrics are not yet implemented")
    metric_functions = {}
    for metric in metrics:
        if isinstance(metric, str):
            metric_functions[metric] = get_metric(metric)
        else:
            name, metric_function = metric
            metric_functions[name] = metric_function
    return metric_functions

class BinaryBalancedAccuracy(BinaryConfusionMatrix):
    def __init__(self, threshold : float = 0.5):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__(threshold=threshold)

    def update(
        self, input: torch.Tensor, target: torch.Tensor
    ):
        super().update(input, target)
        return self

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        specificity = tn / (tn + fp)
        recall = tp / (tp + fn)
        result = (specificity + recall) / 2
        return result

class BinaryBalancedSpecificity(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def update(
            self, input: torch.Tensor, target: torch.Tensor
    ):
        super().update(input, target)
        return self

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        #fn = cm[1, 0]
        #tp = cm[1, 1]
        specificity = tn / (tn + fp)
        return specificity

class BinarySpecificity(BinaryConfusionMatrix):
    def __init__(self, threshold : float = 0.5):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__(threshold=threshold)

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        return tn / (tn + fp)
    
class BinaryPositiveRate(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        positives = tp + fp
        negatives = tn + fn
        total = positives + negatives
        return positives / total
    

class MulticlassAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)

    def update(self, input: torch.Tensor, target: torch.Tensor):
        preds = input.argmax(dim=1)

        # target is already class indices [B]
        self.correct += (preds == target).sum()
        self.total += target.size(0)

        return self

    def compute(self):
        return self.correct / self.total.clamp(min=1)

    def reset(self):
        self.correct.zero_()
        self.total.zero_()

    def merge_state(self, other):
        self.correct += other.correct
        self.total += other.total
        
"""
class MulticlassBalancedAccuracy(Metric):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.tp = torch.zeros(num_classes)
        self.fn = torch.zeros(num_classes)

    def update(self, input, target):
        preds = input.argmax(dim=1)

        for c in range(self.num_classes):
            self.tp[c] += ((preds == c) & (target == c)).sum()
            self.fn[c] += ((preds != c) & (target == c)).sum()

        return self

    def compute(self):
        recalls = self.tp / (self.tp + self.fn).clamp(min=1)
        return recalls.mean()

    def reset(self):
        self.tp.zero_()
        self.fn.zero_()
"""

class MulticlassBalancedAccuracy(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes)

    def update(self, input: torch.Tensor, target: torch.Tensor):
        preds = input.argmax(dim=1)

        # target already [B]
        for t, p in zip(target, preds):
            self.confusion[t.long(), p.long()] += 1

        return self

    def compute(self):
        recalls = []

        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fn = self.confusion[c].sum() - tp

            denom = tp + fn
            if denom > 0:
                recalls.append(tp / denom)

        return torch.stack(recalls).mean()

    def reset(self):
        self.confusion.zero_()

    def merge_state(self, other):
        self.confusion += other.confusion

from .pearson_correlation import PearsonCorrelationCoefficient

__all__=[
    'BinaryBalancedAccuracy',
    'BinarySpecificity',
    'BinaryPositiveRate',
    'MulticlassAccuracy',
    'MulticlassBalancedAccuracy',
    'PearsonCorrelationCoefficient',
    'Elapsed'
]