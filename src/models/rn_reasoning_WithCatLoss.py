import logging
from pathlib import Path
import pandas as pd
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

ONTOLOGY_COLS = [
    "SignClass_A",
    "SignClass_B",
    "SignClass_C",
    "SignClass_D",
]

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
        
        self.pre_output = None

    def forward(self, x):
        #x = self.pre(x)
        #return self.post(x)
        x = self.pre(x)

        if self.stage == "detach_pre":
            x = x.detach()
            
        self.pre_output = x

        return self.post(x)
    
class OntologyRNWithSkip(nn.Module):
    """
    Ontology reasoning network with a skip connection.

    Input:
        53 original features

    Pre:
        53 -> ... -> 4 intermediate concepts

    Post:
        [53 original features + 4 concepts] -> ... -> outputs

    The original input is concatenated with the intermediate
    ontology representation before entering the post network.
    """

    def __init__(
        self,
        input_size,
        category_size,
        num_classes,
        pre_layers,
        post_layers,
    ):
        super().__init__()

        self.stage = "full"

        # INPUT -> INTERMEDIATE CONCEPTS
        pre = []
        in_dim = input_size

        for s in pre_layers:
            pre.append(nn.Linear(in_dim, s))
            pre.append(nn.ReLU())
            in_dim = s

        pre.append(nn.Linear(in_dim, category_size))
        pre.append(nn.ReLU())

        self.pre = nn.Sequential(*pre)

        # ORIGINAL INPUT + INTERMEDIATE CONCEPTS -> OUTPUT
        post = []

        # IMPORTANT:
        # The post network now receives:
        #
        #   input_size + category_size
        #
        # e.g. 53 + 4 = 57
        in_dim = input_size + category_size

        for s in post_layers:
            post.append(nn.Linear(in_dim, s))
            post.append(nn.ReLU())
            in_dim = s

        post.append(nn.Linear(in_dim, num_classes + 1))

        self.post = nn.Sequential(*post)

        self.pre_output = None

    def forward(self, x):

        # Keep the original input for the skip connection.
        original_x = x

        # 53 -> 4
        x = self.pre(x)

        if self.stage == "detach_pre":
            x = x.detach()

        self.pre_output = x

        # 53 + 4 -> POST
        post_input = torch.cat(
            [original_x, x],
            dim=1
        )

        return self.post(post_input)
    
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
        
class OntologyReasoningLoss(nn.Module):
    """
    Combined loss for OntologyRN.

    Final loss:
        CrossEntropyLoss on the final class/invalid prediction.

    Intermediate loss:
        Binary classification loss on the four intermediate
        SignClass_* predictions produced by model.pre().

    Total:
        final_loss + intermediate_weight * intermediate_loss
    """

    def __init__(self, model, intermediate_weight=1.0):
        super().__init__()

        self.model = model
        self.intermediate_weight = intermediate_weight
        self.final_loss = nn.CrossEntropyLoss()
        self.intermediate_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        # ---------------------------------------------------------
        # FINAL CLASS LOSS
        # ---------------------------------------------------------
        final_loss = self.final_loss(y_pred, y_true)

        # ---------------------------------------------------------
        # INTERMEDIATE SIGN CLASS LOSS
        # ---------------------------------------------------------
        #
        # OntologyRN.forward() stores the output of pre() here.
        #
        # Shape:
        #   [batch_size, 4]
        #
        # Corresponding targets are the first four columns of y_true:
        #   SignClass_A
        #   SignClass_B
        #   SignClass_C
        #   SignClass_D
        #
        intermediate_pred = self.model.pre_output

        if intermediate_pred is None:
            raise RuntimeError(
                "model.pre_output is None. "
                "OntologyRN.forward() must be called before the loss."
            )

        sign_class_size = intermediate_pred.shape[1]

        intermediate_true = y_true[:, :sign_class_size].float()

        # pre_output currently ends with ReLU(), so its values are
        # non-negative but are not probabilities. Convert them to
        # probabilities before BCELoss.
        intermediate_prob = torch.sigmoid(intermediate_pred)

        intermediate_loss = self.intermediate_loss(
            intermediate_prob,
            intermediate_true
        )

        return (
            (final_loss
            + self.intermediate_weight * intermediate_loss) / 2
        )
        
class IntermediateSignClassBalancedAccuracy:
    """
    Auxiliary metric for the four intermediate OntologyRN neurons.

    This metric deliberately does NOT use the Trainer's target tensor.

    Instead, at compute time it:
      1. Reads the currently generated train.csv.
      2. Gets the SignClass_* labels directly from that CSV.
      3. Runs the model's pre() network on the feature columns.
      4. Uses one intermediate neuron as the prediction.
      5. Computes binary balanced accuracy.

    This therefore has no effect on the training loss or the Trainer.
    """

    def __init__(
        self,
        model,
        dataset_path,
        feature_cols,
        category_col,
        neuron_idx,
        threshold=0.5,
    ):
        self.model = model
        self.dataset_path = dataset_path
        self.feature_cols = feature_cols
        self.category_col = category_col
        self.neuron_idx = neuron_idx
        self.threshold = threshold

    def update(self, *args, **kwargs):
        # Intentionally empty.
        #
        # The normal Trainer metrics receive the model's final output.
        # We don't need that output here because this metric evaluates
        # OntologyRN.pre() independently during compute().
        pass

    def compute(self):
        df = pd.read_csv(self.dataset_path)

        x = torch.tensor(
            df[self.feature_cols].values,
            dtype=torch.float32,
        )

        y_true = torch.tensor(
            df[self.category_col].values,
            dtype=torch.int64,
        )

        # Find the device on which the model currently lives.
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = x.device

        x = x.to(device)

        # Preserve current training state.
        was_training = self.model.training

        self.model.eval()

        with torch.no_grad():
            # IMPORTANT:
            # We intentionally call only the pre network.
            pre_output = self.model.pre(x)

            scores = pre_output[:, self.neuron_idx]

            # The generated category labels are binary.
            predictions = (scores >= self.threshold).to(torch.int64)

        # Restore the model's previous state.
        if was_training:
            self.model.train()

        y_true = y_true.to(device)
        predictions = predictions.to(device)

        # Binary balanced accuracy:
        #
        # sensitivity = TP / (TP + FN)
        # specificity = TN / (TN + FP)
        #
        # balanced_accuracy = (sensitivity + specificity) / 2
        #
        # Handle missing positive/negative classes safely.
        tp = ((predictions == 1) & (y_true == 1)).sum().float()
        fn = ((predictions == 0) & (y_true == 1)).sum().float()
        tn = ((predictions == 0) & (y_true == 0)).sum().float()
        fp = ((predictions == 1) & (y_true == 0)).sum().float()

        positive_total = tp + fn
        negative_total = tn + fp

        if positive_total > 0:
            sensitivity = tp / positive_total
        else:
            sensitivity = torch.tensor(0.0, device=device)

        if negative_total > 0:
            specificity = tn / negative_total
        else:
            specificity = torch.tensor(0.0, device=device)

        return ((sensitivity + specificity) / 2).item()

    def reset(self):
        # No accumulated state is used.
        pass


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
            target=self.class_cols + ["invalid"], #+ ["SignClass_A", "SignClass_B", "SignClass_C", "SignClass_D"],
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
        target= class_cols + ["invalid"], #+ ["SignClass_A", "SignClass_B", "SignClass_C", "SignClass_D"],
        #target=["target"],
        stratify_col=None
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
    if model_cfg["type"] == "ontology_x":
            model = OntologyRN(
                input_size=53,
                category_size=20,
                num_classes=len(class_cols),
                pre_layers=model_cfg["pre"],
                post_layers=model_cfg["post"]
            )
    elif model_cfg["type"] == "ontology_skip":
        model = OntologyRNWithSkip(
            input_size=53,
            category_size=4,
            num_classes=len(class_cols),
            pre_layers=model_cfg["pre"],
            post_layers=model_cfg["post"]
        )
    elif model_cfg["type"] == "ontology_skip_x":
            model = OntologyRNWithSkip(
                input_size=53,
                category_size=20,
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
    if training_mode == "two_stage" and isinstance(model, (OntologyRN, OntologyRNWithSkip)):

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
        
        if isinstance(model, (OntologyRN, OntologyRNWithSkip)):
            sign_classes = [
                "SignClass_A",
                "SignClass_B",
                "SignClass_C",
                "SignClass_D",
            ]

            for neuron_idx, category_col in enumerate(sign_classes):
                metrics[f"{category_col}_balanced_accuracy"] = (
                    IntermediateSignClassBalancedAccuracy(
                        model=model,
                        dataset_path=train_csv,
                        feature_cols=feature_cols,
                        category_col=category_col,
                        neuron_idx=neuron_idx,
                    )
                )

        return metrics

    train_metrics = TrainingRecorder(
        metric_functions=metrics_factory()
    )
    
    

    #objective = Maximize("train", "balanced_accuracy", threshold=0.01)
    objective = Maximize("train", "accuracy", threshold=0.01)
    patience_objective = Minimize("train", "loss", threshold=0.01)
    
    #num_classes = len(class_cols) + 1  # + invalid

    #weights = torch.ones(num_classes)

    # downweight invalid to avoid dominating the loss
    #weights[-1] = 1.0 / len(class_cols)

    #weights = weights.to(torch.get_default_device())
    if isinstance(model, (OntologyRN, OntologyRNWithSkip)):
        loss_fn = OntologyReasoningLoss(model, intermediate_weight=1.0)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        #model=create_model(38, layer_sizes, num_outputs=1 + len(class_cols)),
        model=model,
        #loss_fn=MaskedBCELoss(),
        #loss_fn=torch.nn.BCELoss(),
        #loss_fn=torch.nn.CrossEntropyLoss(weight=weights),
        loss_fn=loss_fn,
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