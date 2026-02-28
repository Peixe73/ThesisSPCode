from core.datasets import CSVImageDataset, register_datasets
from pathlib import Path
from collections import defaultdict
import numpy as np

PATH = Path("data/gtsrb_dataset")

SEED = 42
IMAGE_COLUMN = 'Filename'

DTYPES = defaultdict(lambda: np.int32, {
    'Filename': str,
    'ClassId': np.int32
})

# Function to convert CSV row to image path
def gtsrb_name_getter(row):
    # row['Filename'] already has relative path inside CSV
    return row[IMAGE_COLUMN]

register_datasets(
    gtsrb_train = CSVImageDataset(
        csv_path = PATH.joinpath('train/Images/GT-final_train.csv'),  # may need to merge train CSVs
        images_path = PATH.joinpath('train/Images'),
        image_columns = [
            (IMAGE_COLUMN, gtsrb_name_getter),
        ],
        target = ['ClassId'],
        features = [IMAGE_COLUMN],
        random_state = SEED
    ),
    gtsrb_test = CSVImageDataset(
        csv_path = PATH.joinpath('test/GT-final_test.csv'),
        images_path = PATH.joinpath('test/Images'),
        image_columns = [
            (IMAGE_COLUMN, gtsrb_name_getter),
        ],
        target = ['ClassId'],
        features = [IMAGE_COLUMN],
        random_state = SEED
    )
)