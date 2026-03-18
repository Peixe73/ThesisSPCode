import pandas as pd
import logging

logger = logging.getLogger(__name__)


def make_signature(row, feature_cols):
    return tuple(int(row[c]) for c in feature_cols)


def load_valid_signatures(path, feature_cols):
    logger.info("Reading CSV: %s", path)

    df = pd.read_csv(path)

    valid_set = set()
    valid_list = []

    for _, row in df.iterrows():
        sig = make_signature(row, feature_cols)
        valid_set.add(sig)
        valid_list.append(sig)

    logger.info(
        "Loaded %d valid signatures (unique=%d)",
        len(valid_list),
        len(valid_set)
    )

    return valid_set, valid_list