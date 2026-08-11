import logging
from pathlib import Path
from torch import nn
import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall#MulticlassBalancedAccuracy

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

# MODEL

class OntologyRN(nn.Module):
    """
    Two-part ontology RN:
    - pre: input -> categories
    - post: categories -> class + validity
    """
    def __init__(self, input_size, category_size, num_classes, pre_layers, post_layers):
        super().__init__()
        
        self.stage = "full"

        # INPUT -> MID (pre)
        pre = []
        in_dim = input_size
        for s in pre_layers:
            pre.append(nn.Linear(in_dim, s))
            pre.append(nn.ReLU())
            in_dim = s

        #self.pre = nn.Sequential(*pre) if len(pre_layers) > 0 else nn.Identity()
        pre.append(nn.Linear(in_dim, category_size))
        pre.append(nn.ReLU())

        self.pre = nn.Sequential(*pre)

        # MID -> OUTPUT (post)
        post = []
        in_dim = category_size
        for s in post_layers:
            post.append(nn.Linear(in_dim, s))
            post.append(nn.ReLU())
            in_dim = s

        post.append(nn.Linear(in_dim, num_classes + 1))  # +valid
        #post.append(nn.Sigmoid())

        self.post = nn.Sequential(*post)

    def forward(self, x):
        #x = self.pre(x)
        #return self.post(x)
        x = self.pre(x)

        if self.stage == "detach_pre":
            x = x.detach()

        return self.post(x)
    
class DirectRN(nn.Module):
    def __init__(self, input_size, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, num_outputs),
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def create_mlp(input_size, layer_sizes, num_outputs):
    layers = []
    in_dim = input_size

    for s in layer_sizes:
        layers.append(nn.Linear(in_dim, s))
        layers.append(nn.ReLU())
        in_dim = s

    layers.append(nn.Linear(in_dim, num_outputs))
    #layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)

'''

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
'''

class MaskedBCELoss(nn.Module):
    """
    - valid head: always trained
    - class heads: ONLY trained when valid == 1
    """
    def __init__(self):
        super().__init__()
        #self.bce = nn.BCELoss(reduction="none")
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

        masked = class_loss * mask

        # average only over active class terms
        denom  = mask.sum() * class_pred.shape[1]

        class_loss_mean = (
            masked.sum() / denom
            if denom > 0
            else torch.tensor(0.0, device=y_pred.device)
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
    def __init__(self, valid_path, feature_cols, class_cols, dataset_size, base_seed, concept_noise_mode="binary"):
        self.valid_path = valid_path
        self.feature_cols = feature_cols
        self.class_cols = class_cols
        self.dataset_size = dataset_size
        self.base_seed = base_seed
        self.concept_noise_mode = concept_noise_mode
        
        self.epoch = 0
        self.latest_dataset = None
        #print("BASE SEED:", base_seed)
        #print("TYPE:", type(base_seed))

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    """
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
    
    """
    def on_epoch_start(self, trainer: Trainer):
        path = DEBUG_DIR / "train.csv"

        #epoch_seed = self.base_seed * 1000003 + self.epoch
        epoch_seed = self.base_seed + self.epoch

        ds = RandomReasoningDataset(
            self.valid_path,
            self.feature_cols,
            self.class_cols,
            self.dataset_size,
            seed=epoch_seed,
            concept_noise_mode=self.concept_noise_mode
        )
        ds.to_csv(path)

        self.latest_dataset = CSVDataset(
            path=path,
            features=self.feature_cols,
            target=self.class_cols + ["invalid"],
            #target=["target"],
            stratify_col=None
        )

        # swap safely at epoch boundary
        trainer.training_set = self.latest_dataset

        logger.info(f"[DATASET] Epoch {self.epoch} | Seed {epoch_seed}")

        self.epoch += 1


def create_trainer(
    valid_path: str,
    feature_cols: list[str],
    class_cols: list[str],
    model_config: dict,
    #layer_sizes: list[int],
    dataset_size: int = 3200,
    batch_size: int = 64,
    patience: int = 20,
    base_seed: int = 42,
    training_mode: str = "standard",
    stage: str | None = None,               # used only for two_stage
    concept_noise_mode: str = "binary"
) -> Trainer:
    
    num_classes = len(class_cols) + 1

    #print("create_trainer base_seed =", base_seed, type(base_seed))
    
    train_csv = DEBUG_DIR / "train.csv"
    
    initial_seed = base_seed * 1000003 #+ 0

    init_ds = RandomReasoningDataset(
        valid_path,
        feature_cols,
        class_cols,
        dataset_size,
        seed=initial_seed,
        concept_noise_mode=concept_noise_mode
    )

    init_ds.to_csv(train_csv)

    train_dataset = CSVDataset(
        path=train_csv,
        features=feature_cols,
        #target=["valid"] + class_cols
        target= class_cols + ["invalid"],
        #target=["target"],
        stratify_col=None
    )
    
    metrics_per_class = {
        "balanced_accuracy": metric_wrappers.to_int(core.eval.metrics.BinaryBalancedAccuracy)
    }
    
    dataset_updater = EpochDatasetUpdater(
        valid_path,
        feature_cols,
        class_cols,
        dataset_size,
        base_seed,
        concept_noise_mode
    )

    def metrics_factory():
        metrics = {
            "epoch_elapsed": Elapsed(),
            #"dataset_update": dataset_updater, # this will regenerate the dataset at the end of each epoch
            #"balanced_accuracy": core.eval.metrics.MulticlassBalancedAccuracy(
            #num_classes=len(class_cols) + 1),
            #"accuracy": MulticlassAccuracy(num_classes=len(class_cols) + 1),
            #"f1": MulticlassF1Score(num_classes=len(class_cols) + 1, average="macro"),
            #"accuracy": core.eval.metrics.MulticlassAccuracy(),
            
            "accuracy": metric_wrappers.ToMulticlass(
                MulticlassAccuracy(
                    num_classes=num_classes
                )
            ),

            "recall": metric_wrappers.ToMulticlass(
                MulticlassRecall(
                    num_classes=num_classes
                )
            ),

            "f1": metric_wrappers.ToMulticlass(
                MulticlassF1Score(
                    num_classes=num_classes,
                    average="macro"
                )
            ),
        }
        
        metric_wrappers.SelectCol.col_wise(
            train_dataset,
            metrics_per_class,
            #reduction="min",   # gives a global "balanced_accuracy"
            out_dict=metrics
        )

        return metrics

    train_metrics = TrainingRecorder(
        metric_functions=metrics_factory()
    )
    
    # model selection
    num_outputs = len(class_cols) + 1

    #if isinstance(layer_sizes, dict):
    model_cfg = model_config
    
    if model_cfg["type"] == "ontology":
        model = OntologyRN(
            input_size=53,
            category_size=4,
            num_classes=len(class_cols),
            pre_layers=model_cfg["pre"],
            post_layers=model_cfg["post"]
        )
    elif model_cfg["type"] == "direct":
        model = DirectRN(input_size = 53,
                         num_outputs = num_outputs)
    else:
        model = create_mlp(53, model_cfg["layers"], num_outputs)

    # logic for two-stage training (if applicable)
    if training_mode == "two_stage" and isinstance(model, OntologyRN):

        if stage == "pretrain_post":
            logger.info("Stage A: POST training")

            model.stage = "detach_pre"

            for p in model.pre.parameters():
                p.requires_grad = False
            for p in model.post.parameters():
                p.requires_grad = True

        elif stage == "pretrain_pre":
            logger.info("Stage B: PRE training")

            model.stage = "full"

            for p in model.post.parameters():
                p.requires_grad = False
            for p in model.pre.parameters():
                p.requires_grad = True

    #objective = Maximize("train", "balanced_accuracy", threshold=0.01)
    objective = Maximize("train", "accuracy", threshold=0.01)
    patience_objective = Minimize("train", "loss", threshold=0.01)
    
    #num_classes = len(class_cols) + 1  # + invalid

    #weights = torch.ones(num_classes)

    # downweight invalid to avoid dominating the loss
    #weights[-1] = 1.0 / len(class_cols)

    #weights = weights.to(torch.get_default_device())


    trainer = Trainer(
        #model=create_model(38, layer_sizes, num_outputs=1 + len(class_cols)),
        model=model,
        #loss_fn=MaskedBCELoss(),
        #loss_fn=torch.nn.BCELoss(),
        #loss_fn=torch.nn.CrossEntropyLoss(weight=weights),
        loss_fn=torch.nn.CrossEntropyLoss(),
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
    
    trainer.epoch_start_hooks.append(dataset_updater.on_epoch_start)

    return trainer