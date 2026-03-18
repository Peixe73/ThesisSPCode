import torch
import torch.nn as nn
import torch.optim as optim
import logging

from analysis_tools.signature_utils import load_valid_signatures
from analysis_tools.random_batch_generator import RandomBalancedBatchGenerator

logger = logging.getLogger(__name__)


# =========================
# MODEL
# =========================

class ReasoningNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =========================
# STUDY
# =========================

class Study:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------
    # BUILD
    # ---------------------
    def build(self):

        logger.info("Building reasoning study...")
        logger.info("Using device: %s", self.device)

        # Load valid signatures
        valid_set, valid_list = load_valid_signatures(
            self.cfg["valid_path"],
            self.cfg["feature_cols"]
        )

        logger.info("Valid signatures loaded: %d", len(valid_list))

        # Batch generator
        self.batch_gen = RandomBalancedBatchGenerator(
            valid_list,
            valid_set,
            num_features=len(self.cfg["feature_cols"]),
            batch_size=self.cfg["batch_size"],
            device=self.device
        )

        # Model
        self.model = ReasoningNet(len(self.cfg["feature_cols"])).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg["lr"]
        )

        self.criterion = nn.BCELoss()

        logger.info("Study build complete.")

    # ---------------------
    # STEP (one iteration)
    # ---------------------
    def step(self, step_idx):

        self.model.train()

        x, y = self.batch_gen.get_batch()

        pred = self.model(x)
        loss = self.criterion(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item()
        }

    # ---------------------
    # EPOCH LOOP
    # ---------------------
    def run(self):

        logger.info("Starting training...")

        for epoch in range(self.cfg["epochs"]):

            losses = []

            for step_idx in range(self.cfg["steps_per_epoch"]):
                metrics = self.step(step_idx)
                losses.append(metrics["loss"])

            avg_loss = sum(losses) / len(losses)

            logger.info(
                "Epoch %d | Loss: %.6f",
                epoch,
                avg_loss
            )

            # Early stopping
            if avg_loss < self.cfg["early_stop_loss"]:
                logger.info("Converged at epoch %d", epoch)
                break

        logger.info("Training finished.")


# =========================
# ENTRYPOINT (REQUIRED)
# =========================

def main():

    cfg = {
        "valid_path": "data/gtsrb_ontology_Valid_Final.csv",

        "feature_cols": [
            "Circular_Shape", "Diamond_Shape", "Triangular_Shape", "Octagonal_Shape",
            "Red_Ground", "White_Ground", "Yellow_Ground", "Blue",
            "Border", "Black_Border", "Red_Border", "White_Border",
            "Bar", "Black_Bar", "White_Bar",
            "Symbol", "Black_Symbol", "White_Symbol",

            "Symbol_Stop",
            "Symbol_NoEntryGoods", "Symbol_Overtaking", "Symbol_OvertakingGoods",
            "Symbol_Speed20", "Symbol_Speed30", "Symbol_Speed50",
            "Symbol_Speed60", "Symbol_Speed70", "Symbol_Speed80",
            "Symbol_Speed100", "Symbol_Speed120",

            "D1a1", "D1a4", "D1a5", "D1a6", "D1a7",
            "D2a1", "D2a2", "D3",
        ],

        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 1000,
        "steps_per_epoch": 10,
        "early_stop_loss": 0.01,
    }

    study = Study(cfg)
    study.build()
    study.run()


if __name__ == "__main__":
    main()