import torch
import torch.nn as nn
import torch.optim as optim
import logging

from analysis_tools.signature_utils import load_valid_signatures
from analysis_tools.random_batch_generator import RandomBalancedBatchGenerator

logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================

VALID_PATH = "data/gtsrb_ontology_Valid_Final.csv"

FEATURE_COLS = [
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
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
# TRAINING
# =========================

def main():

    logger.info("Starting reasoning network training...")
    logger.info("Using device: %s", DEVICE)

    # Load valid signatures
    logger.info("Loading valid signatures from %s", VALID_PATH)
    valid_set, valid_list = load_valid_signatures(
        VALID_PATH, FEATURE_COLS
    )

    logger.info("Loaded %d valid signatures", len(valid_list))

    # Batch generator
    batch_gen = RandomBalancedBatchGenerator(
        valid_list,
        valid_set,
        num_features=len(FEATURE_COLS),
        batch_size=64,
        device=DEVICE
    )

    logger.info("Initialized batch generator (batch_size=64)")

    # Model
    model = ReasoningNet(len(FEATURE_COLS)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    logger.info("Model initialized")

    # Training loop
    for epoch in range(1000):

        losses = []

        for batch_idx in range(10):
            x, y = batch_gen.get_batch()

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

        logger.info(
            "Epoch %d | Loss: %.6f",
            epoch,
            avg_loss
        )

        # Optional early stopping
        if avg_loss < 0.01:
            logger.info("Training converged at epoch %d", epoch)
            break

    logger.info("Training finished.")


if __name__ == "__main__":
    main()