from pyexpat import features

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

SIGN_CLASS_A = {
    "A19a",
    "A32",
    "A1a",
    "A1b",
    "A1c",
    "A7a",
    "A9",
    "A4b2",
    "A16",
    "A17a",
    "A33",
    "A13",
    "A14",
    "A34",
    "A15b",
}

SIGN_CLASS_B = {
    "B1",
    "B2a",
    "B3",
}

SIGN_CLASS_C = {
    "C1a",
    "C2",
    "C3e3",
    "C13aa",
    "C13bb",
    "C14_20",
    "C14_30",
    "C14_50",
    "C14_60",
    "C14_70",
    "C14_80",
    "C14_100",
    "C14_120",
    "C17a",
    "C17b_80",
    "C17c",
    "C17d",
}

SIGN_CLASS_D = {
    "D1a1",
    "D1a4",
    "D1a5",
    "D1a6",
    "D1a7",
    "D2a1",
    "D2a2",
    "D3",
}

SIGN_CLASS_TO_CATEGORY = {}

for name in SIGN_CLASS_A:
    SIGN_CLASS_TO_CATEGORY[name] = 0

for name in SIGN_CLASS_B:
    SIGN_CLASS_TO_CATEGORY[name] = 1

for name in SIGN_CLASS_C:
    SIGN_CLASS_TO_CATEGORY[name] = 2

for name in SIGN_CLASS_D:
    SIGN_CLASS_TO_CATEGORY[name] = 3

class RandomReasoningDataset(Dataset):
    def __init__(
        self,
        valid_path: str,
        feature_cols: list[str],
        class_cols: list[str],
        dataset_size: int = 640,
        seed: int | None = None,
        concept_noise_mode: str = "binary",
    ):
        self.valid_df = pd.read_csv(valid_path)
        self.feature_cols = feature_cols
        self.class_cols = class_cols
        self.dataset_size = dataset_size

        self.rng = np.random.default_rng(seed)
        self.concept_noise_mode = concept_noise_mode
            

        # Extract arrays
        self.valid_features = self.valid_df[self.feature_cols].values.astype(np.float32)
        self.valid_classes = self.valid_df[self.class_cols].values.astype(np.float32)
        self.sign_classes = np.zeros(
            (len(self.valid_df), 4),
            dtype=np.float32
        )

        for i, class_row in enumerate(self.valid_classes):
            class_indices = np.flatnonzero(class_row)

            if len(class_indices) != 1:
                raise ValueError(
                    f"Expected exactly one valid class, got indices {class_indices}"
                )

            class_name = self.class_cols[class_indices[0]]

            category = SIGN_CLASS_TO_CATEGORY.get(class_name)

            if category is None:
                raise ValueError(
                    f"Class {class_name!r} does not belong to A/B/C/D"
                )

            self.sign_classes[i, category] = 1.0
        self.rows_set = {tuple(row) for row in self.valid_features}
        self.num_outputs = 1 + len(self.class_cols)  # valid + classes

        logger.info("Loaded %d valid signatures", len(self.valid_features))
    
    def _soften_features(self, features):
        features = features.copy()

        if self.concept_noise_mode == "binary":
            return features

        # Uniform split at 0.5
        elif self.concept_noise_mode == "uniform":
            #return self.rng.uniform(0, 1, size=features.shape)
            return np.where(
                features > 0.5,
                self.rng.uniform(0.5, 1.0, size=features.shape),
                self.rng.uniform(0.0, 0.5, size=features.shape)
            ).astype(np.float32)

        # Concentrated near 0 and 1
        elif self.concept_noise_mode == "extremes":
            #return self.rng.beta(0.5, 0.5, size=features.shape)
            highs = self.rng.beta(5, 1, size=features.shape)
            lows  = self.rng.beta(1, 5, size=features.shape)

            return np.where(
                features > 0.5,
                highs,
                lows
            ).astype(np.float32)

        # Concentrated near 0.5
        elif self.concept_noise_mode == "middle":
            #return self.rng.beta(10, 10, size=features.shape)
            
            # symmetric concentration around 0.5
            
            noise = self.rng.beta(5, 5, size=features.shape)

            # map directly into [0,1] centered at 0.5
            return noise.astype(np.float32)
        """
        elif self.concept_noise_mode == "middle":
            highs = 0.5 + 0.5 * self.rng.beta(5, 5, size=features.shape)
            lows  = 0.5 * self.rng.beta(5, 5, size=features.shape)

            return np.where(
                features > 0.5,
                highs,
                lows
            ).astype(np.float32)
        """

        raise ValueError(
            f"Unknown concept_noise_mode: {self.concept_noise_mode}"
        )

    """
    def _sample_valid(self):
        idx = self.rng.integers(0, len(self.valid_features))
        return self.valid_features[idx], self.valid_classes[idx]
    """
    def _sample_valid(self):
        idx = self.rng.integers(0, len(self.valid_features))

        features = self.valid_features[idx]
        features = self._soften_features(features)

        #return features, self.valid_classes[idx]
        return (
            features,
            self.valid_classes[idx],
            self.sign_classes[idx],
        )
        #class_index = int(np.argmax(self.valid_classes[idx]))

        #return features, class_index

    """
    def _sample_invalid(self):
        if self.rng.random() < 0.5:
            while True:
                row = self.rng.integers(0, 2, size=len(self.feature_cols)).astype(np.float32)
                if tuple(row) not in self.rows_set:
                    break
        else:
            row, _ = self._sample_valid()
            row = row.copy()
            idx = self.rng.integers(0, len(row))
            row[idx] = 1 - row[idx]
            if tuple(row) in self.rows_set:
                return self._sample_invalid()

        # invalid → no class
        class_labels = np.zeros(len(self.class_cols), dtype=np.float32)
        return row, class_labels
    """
    
    def _sample_invalid(self):
        # 50%: fully random invalid
        if self.rng.random() < 0.5:
            while True:
                row = self.rng.integers(
                    0, 2, size=len(self.feature_cols)
                ).astype(np.float32)

                if tuple(row) not in self.rows_set:
                    break
            
            """
            class_labels = np.zeros(len(self.class_cols), dtype=np.float32)
            row = self._soften_features(row)
            return row, class_labels
            """
            
            class_labels = np.zeros(len(self.class_cols), dtype=np.float32)
            sign_classes = np.zeros(4, dtype=np.float32)

            row = self._soften_features(row)

            return row, class_labels, sign_classes
            #invalid_class = len(self.class_cols)
            #return row, invalid_class

        # 50% corrupted valid sample
        idx = self.rng.integers(0, len(self.valid_features))
        row = self.valid_features[idx].copy()

        k = self.rng.integers(1, 6)
        feature_indices = self.rng.choice(len(row), size=k, replace=False)

        for i in feature_indices:
            row[i] = 1.0 - row[i]
        
        if tuple(row) in self.rows_set:
                return self._sample_invalid()
            
        row = self._soften_features(row)
        #return row, len(self.class_cols)

        """
        class_labels = np.zeros(len(self.class_cols), dtype=np.float32)
        return row, class_labels
        """
        class_labels = np.zeros(len(self.class_cols), dtype=np.float32)
        sign_classes = np.zeros(4, dtype=np.float32)

        return row, class_labels, sign_classes

    def __len__(self):
        return self.dataset_size

    """
    def __getitem__(self, idx):
        if self.rng.random() < 0.5:
            features, target = self._sample_valid()
        else:
            features, target = self._sample_invalid()

        x = torch.from_numpy(features)
        y = torch.tensor(target, dtype=torch.long)

        return x, y
    """

    def __getitem__(self, idx):
        if self.rng.random() < 0.5:
            features, class_labels, sign_classes = self._sample_valid()
            invalid = 0.0
        else:
            features, class_labels, sign_classes = self._sample_invalid()
            invalid = 1.0

        #y = np.concatenate([[valid], class_labels]).astype(np.float32)
        y = np.concatenate([class_labels, [invalid]]).astype(np.float32)
        x = torch.from_numpy(features)
        y = torch.from_numpy(y)
        sign_classes = torch.from_numpy(sign_classes)
        """
        if self.return_sign_classes:
            sign_classes = torch.from_numpy(sign_classes)
            return x, y, sign_classes
        """

        #return x, y
        return x, y, sign_classes
    
    def to_csv(self, path: str):
        rows = []

        for i in range(len(self)):
            x, y, sign_classes = self[i]

            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            sign_classes = sign_classes.detach().cpu().numpy()

            entry = {
                self.feature_cols[j]: float(x[j])
                for j in range(len(self.feature_cols))
            }

            # classes first (commented version is inverse)
            for j, col in enumerate(self.class_cols):
                entry[col] = float(y[j])

            # valid last
            entry["invalid"] = float(y[-1])
            
            entry["SignClass_A"] = float(sign_classes[0])
            entry["SignClass_B"] = float(sign_classes[1])
            entry["SignClass_C"] = float(sign_classes[2])
            entry["SignClass_D"] = float(sign_classes[3])
            
            
            #entry["target"] = int(y)

            rows.append(entry)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        pd.DataFrame(rows).to_csv(path, index=False)

        return path

    '''
    def to_csv(self, path: str):
        """Generate a CSV containing the full dataset."""
        rows = []
        for i in range(len(self)):
            x, y = self[i]
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            entry = {self.feature_cols[j]: float(x[j]) for j in range(len(self.feature_cols))}
            entry["valid"] = float(y[0])
            for j, col in enumerate(self.class_cols):
                entry[col] = float(y[j + 1])
            rows.append(entry)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path
    '''