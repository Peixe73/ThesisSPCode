import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from core.datasets.csv_dataset import CSVDataset


CSV_PATH = "data/gtsrb_ontology_Valid_Final.csv"

FEATURE_COLS = [ ... ] 


def create_model(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim),
        nn.Sigmoid()
    )


def main():
    print("Loading dataset...")

    dataset = CSVDataset(
        path=CSV_PATH,
        features=FEATURE_COLS,
        target=FEATURE_COLS
    )

    # extract tensors directly
    X = torch.tensor(dataset.features, dtype=torch.float32)
    Y = torch.tensor(dataset.targets, dtype=torch.float32)

    # tiny subset for overfitting
    X = X[:32]
    Y = Y[:32]

    loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=32,
        shuffle=False  # avoid your CUDA generator bug
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(len(FEATURE_COLS), len(FEATURE_COLS)).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Starting overfit test...\n")

    for epoch in range(1, 101):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss:.6f}")

    print("\nFinal evaluation:")

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            print("Prediction mean:", preds.mean().item())
            print("Prediction std:", preds.std().item())

            preds = (preds > 0.5).float()
            acc = (preds == y).float().mean()

            print("Accuracy:", acc.item())


if __name__ == "__main__":
    main()