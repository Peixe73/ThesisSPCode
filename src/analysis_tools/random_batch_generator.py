import random
import torch
import logging

logger = logging.getLogger(__name__)


class RandomBalancedBatchGenerator:
    def __init__(
        self,
        valid_list,
        valid_set,
        num_features,
        batch_size=64,
        device="cpu"
    ):
        self.valid_list = valid_list
        self.valid_set = valid_set
        self.num_features = num_features
        self.batch_size = batch_size
        self.device = device

        logger.info(
            "BatchGenerator initialized | features=%d | batch_size=%d",
            num_features,
            batch_size
        )

    def _random_signature(self):
        return tuple(random.randint(0, 1) for _ in range(self.num_features))

    def _get_valid_sample(self):
        return random.choice(self.valid_list), 1

    def _get_invalid_sample(self):
        while True:
            sig = self._random_signature()
            if sig not in self.valid_set:
                return sig, 0

    def get_batch(self):
        half = self.batch_size // 2

        batch_x = []
        batch_y = []

        # valid
        for _ in range(half):
            x, y = self._get_valid_sample()
            batch_x.append(x)
            batch_y.append(y)

        # invalid
        for _ in range(half):
            x, y = self._get_invalid_sample()
            batch_x.append(x)
            batch_y.append(y)

        combined = list(zip(batch_x, batch_y))
        random.shuffle(combined)
        batch_x, batch_y = zip(*combined)

        return (
            torch.tensor(batch_x, dtype=torch.float32).to(self.device),
            torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1).to(self.device),
        )