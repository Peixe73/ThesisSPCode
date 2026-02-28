from core.datasets import CSVImageDataset, register_datasets
from pathlib import Path
from collections import defaultdict
import numpy as np

PATH = Path("data/gtsrb_dataset")
SEED = 42

IMAGE_COLUMN = "Filename"

# ----- CLASS -----
CLASSES = ["ClassId"]

# ----- CONCEPTS -----
CONCEPTS = [
    "Bar", "Black_Bar", "White_Bar",
    "Border", "Black_Border", "Red_Border",
    "Red_Ground", "White_Ground", "Yellow_Ground",
    "End_Prohibition", "Start_Prohibition",
    "Circular_Shape", "Diamond_Shape", "Triangular_Shape", "Octagonal_Shape",
    "Symbol", "Black_Symbol", "White_Symbol",
    "Symbol_NoEntryGoods", "Symbol_Overtaking", "Symbol_OvertakingGoods",
    "Symbol_Speed20", "Symbol_Speed30", "Symbol_Speed50", "Symbol_Speed60",
    "Symbol_Speed70", "Symbol_Speed80", "Symbol_Speed100", "Symbol_Speed120",
    "Symbol_Stop",
    "Blue", "White_Border", "Post",
    "D1a1", "D1a4", "D1a5", "D1a6", "D1a7",
    "D2a1", "D2a2", "D3"
]

DTYPES = defaultdict(lambda: np.int32, {
    "Filename": str,
    "SignClass": str,
    "Track": str
})

register_datasets(

    # ----- CLASSES ONLY -----
    gtsrb = CSVImageDataset(
        csv_path=PATH.joinpath("gtsrb.csv"),
        #images_path=PATH.joinpath("train"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, None)],
        target=CLASSES,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED
    ),

    # ----- CLASSES + CONCEPTS -----
    gtsrb_with_concepts = CSVImageDataset(
        csv_path=PATH.joinpath("gtsrb.csv"),
        #images_path=PATH.joinpath("train"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, None)],
        target=CLASSES + CONCEPTS,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED
    ),

    # ----- CONCEPTS ONLY -----
    gtsrb_concepts_only = CSVImageDataset(
        csv_path=PATH.joinpath("gtsrb.csv"),
        #images_path=PATH.joinpath("train"),~
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, None)],
        target=CONCEPTS,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED
    ),
)