import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RandomReasoningDataset(Dataset):
    def __init__(
        self,
        valid_path: str,
        feature_cols: list[str],
        dataset_size: int = 640,
        debug: bool = False
    ):
        self.valid_df = pd.read_csv(valid_path)
        self.feature_cols = feature_cols
        self.dataset_size = dataset_size
        self.debug = debug

        self.num_outputs = 1

        self.valid_rows = self.valid_df[self.feature_cols].values.astype(np.float32)
        self.rows_set = {tuple(row) for row in self.valid_rows}

        logger.info("Loaded %d valid signatures", len(self.valid_rows))

    def _sample_valid(self):
        idx = np.random.randint(0, len(self.valid_rows))
        return self.valid_rows[idx]

    def _sample_invalid(self):
        if np.random.rand() < 0.5:
            while True:
                row = np.random.randint(0, 2, size=len(self.feature_cols)).astype(np.float32)
                if tuple(row) not in self.rows_set:
                    return row
        else:
            row = self._sample_valid().copy()
            idx = np.random.randint(0, len(row))
            row[idx] = 1 - row[idx]

            if tuple(row) not in self.rows_set:
                return row

            return self._sample_invalid()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx % 2 == 0:
            row = self._sample_valid()
            label = 1.0
        else:
            row = self._sample_invalid()
            label = 0.0

        x = torch.from_numpy(row).float()
        y = torch.tensor([label], dtype=torch.float32)

        return x, y