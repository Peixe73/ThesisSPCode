import random
import torch
import pandas as pd


class BalancedBatchGenerator:
    def __init__(
        self,
        valid_list,
        valid_set,
        full_dataset_path,
        feature_cols,
        batch_size=64,
        device="cpu"
    ):
        self.valid_list = valid_list
        self.valid_set = valid_set
        self.batch_size = batch_size
        self.feature_cols = feature_cols
        self.device = device

        # Load full dataset once
        self.full_df = pd.read_csv(full_dataset_path)
        self.num_rows = len(self.full_df)

    def _get_valid_sample(self):
        sig = random.choice(self.valid_list)
        return sig, 1

    def _get_invalid_sample(self):
        while True:
            idx = random.randint(0, self.num_rows - 1)
            row = self.full_df.iloc[idx]

            sig = tuple(int(row[c]) for c in self.feature_cols)

            if sig not in self.valid_set:
                return sig, 0

    def get_batch(self):
        half = self.batch_size // 2

        batch_x = []
        batch_y = []

        # valid samples
        for _ in range(half):
            x, y = self._get_valid_sample()
            batch_x.append(x)
            batch_y.append(y)

        # invalid samples
        for _ in range(half):
            x, y = self._get_invalid_sample()
            batch_x.append(x)
            batch_y.append(y)

        # shuffle
        combined = list(zip(batch_x, batch_y))
        random.shuffle(combined)
        batch_x, batch_y = zip(*combined)

        return (
            torch.tensor(batch_x, dtype=torch.float32).to(self.device),
            torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1).to(self.device),
        )