import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RandomReasoningDataset(Dataset):
    def __init__(self, valid_path: str, feature_cols: list[str], dataset_size: int = 20000):
        self.valid_df = pd.read_csv(valid_path)
        self.feature_cols = feature_cols
        self.dataset_size = dataset_size
        self.num_outputs = 1  # binary classification

        # store valid signatures for fast lookup
        self.rows_set = {
            tuple(row) for row in self.valid_df[self.feature_cols].values
        }

        self.data, self.labels = self._generate_dataset()

        logger.info(
            "RandomReasoningDataset initialized with %d samples",
            len(self.data)
        )

    def _generate_dataset(self):
        data = []
        labels = []

        n_valid = len(self.valid_df)
        logger.info("%d valid signatures loaded", n_valid)

        for _ in range(self.dataset_size):
            if np.random.rand() < 0.5:
                # valid sample
                idx = np.random.randint(0, n_valid)
                row = self.valid_df.iloc[idx][self.feature_cols].values.astype(np.float32)
                label = 1
            else:
                # invalid sample
                while True:
                    row = np.random.randint(0, 2, size=len(self.feature_cols)).astype(np.float32)
                    if tuple(row) not in self.rows_set:
                        break
                label = 0

            data.append(row)
            labels.append(label)

        # IMPORTANT: keep dataset on CPU only
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        return (
            torch.from_numpy(data).float(),                 # CPU tensor
            torch.from_numpy(labels).unsqueeze(1).float()   # CPU tensor
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # NEVER move to CUDA here
        return self.data[idx], self.labels[idx]