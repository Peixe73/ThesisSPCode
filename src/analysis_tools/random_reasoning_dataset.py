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
        class_cols: list[str],
        dataset_size: int = 640,
        seed: int | None = None,
    ):
        self.valid_df = pd.read_csv(valid_path)

        self.feature_cols = feature_cols
        self.class_cols = class_cols
        self.dataset_size = dataset_size

        if seed is not None:
            np.random.seed(seed)

        # Extract arrays
        self.valid_features = self.valid_df[self.feature_cols].values.astype(np.float32)
        self.valid_classes = self.valid_df[self.class_cols].values.astype(np.float32)

        self.rows_set = {tuple(row) for row in self.valid_features}

        self.num_outputs = 1 + len(self.class_cols)  # valid + classes

        logger.info("Loaded %d valid signatures", len(self.valid_features))

    def _sample_valid(self):
        idx = np.random.randint(0, len(self.valid_features))
        return self.valid_features[idx], self.valid_classes[idx]

    def _sample_invalid(self):
        if np.random.rand() < 0.5:
            while True:
                row = np.random.randint(0, 2, size=len(self.feature_cols)).astype(np.float32)
                if tuple(row) not in self.rows_set:
                    break
        else:
            row, _ = self._sample_valid()
            row = row.copy()
            idx = np.random.randint(0, len(row))
            row[idx] = 1 - row[idx]

            if tuple(row) in self.rows_set:
                return self._sample_invalid()

        # invalid → no class
        class_labels = np.zeros(len(self.class_cols), dtype=np.float32)

        return row, class_labels

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx % 2 == 0:
            features, class_labels = self._sample_valid()
            valid = 1.0
        else:
            features, class_labels = self._sample_invalid()
            valid = 0.0

        y = np.concatenate([[valid], class_labels]).astype(np.float32)

        x = torch.from_numpy(features)
        y = torch.from_numpy(y)

        return x, y