import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from core.datasets.csv_dataset import CSVDataset


# -----------------------
# WRAPPER
# -----------------------
class TorchCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dataset):
        self.ds = csv_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds.get_item(idx)
        return x.float(), y.float()


# -----------------------
# CONFIG
# -----------------------
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
    base_dataset = CSVDataset(
        path=CSV_PATH,
        features=FEATURE_COLS,
        target=FEATURE_COLS
    )

    dataset = TorchCSVDataset(base_dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(len(FEATURE_COLS), len(FEATURE_COLS)).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 51):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += loss_fn(preds, y).item()

        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if epoch % 10 == 0:
            with torch.no_grad():
                x, _ = next(iter(val_loader))
                preds = model(x.to(device))

                print("Prediction mean:", preds.mean().item())
                print("Prediction std:", preds.std().item())


if __name__ == "__main__":
    main()