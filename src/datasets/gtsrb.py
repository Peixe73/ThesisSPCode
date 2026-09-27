from core.datasets import CSVImageDataset, register_datasets
from pathlib import Path
from collections import defaultdict
#from analysis_tools.gtsrb_utils import CLASSID_BINARY, SIGNCLASS_BINARY
import numpy as np

from analysis_tools.gtsrb_utils import CATEGORIES_COLS, CLASS_COLS, CONCEPTS
from core.datasets.csv_img_dataset_gtsrb import CSVImageDatasetGTSRB

PATH = Path("data/gtsrb_dataset")
SEED = 42

IMAGE_COLUMN = "Filename"

SPLIT_COLUMN = "train_data"

CLASSES = ["ClassId"]

ONTOLOGY_COLS = [
    "SignClass_A",
    "SignClass_B",
    "SignClass_C",
    "SignClass_D",
]

""" Old Version of Concepts
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
"""

DTYPES = defaultdict(lambda: np.int32, {
    "Filename": str,
    "SignClass": str,
    "Track": str
})

# Helper to get the filename from CSV row or already-extracted string
def path_getter(row):
    if isinstance(row, str):
        return row
    return row[IMAGE_COLUMN]

def train_filter(row):
    return row["train_data"] == 1

def test_filter(row):
    return row["train_data"] == 0

register_datasets(

    # ----- CLASSES ONLY -----
    gtsrb = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_Final_Version.csv"),
        #images_path=PATH.joinpath("train"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=CATEGORIES_COLS,#CLASSES,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED,
        splits=(0.9,0.1),
        use_test_set=False,
    ),

    # ----- CLASSES + CONCEPTS -----
    gtsrb_with_concepts = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_Final_Version.csv"),
        #images_path=PATH.joinpath("train"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=CATEGORIES_COLS + CONCEPTS, #CLASSES + ALL_CONCEPTS, SIGNCLASS_BINARY +
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED,
        splits=(0.9,0.1),
        use_test_set=False,
    ),
    
    # ----- CLASSES + CONCEPTS + CATEGORIES -----
    gtsrb_with_concepts_and_categories = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_Final_Version.csv"),
        #images_path=PATH.joinpath("train"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=CATEGORIES_COLS + CONCEPTS + CLASS_COLS, #CLASSES + ALL_CONCEPTS, SIGNCLASS_BINARY +
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED,
        splits=(0.9,0.1),
        use_test_set=False,
    ),


    # ----- CONCEPTS ONLY -----
    gtsrb_concepts_only = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_Final_Version.csv"),
        #images_path=PATH.joinpath("train"),~
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=CONCEPTS,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED,
        splits=(0.9,0.1),
        use_test_set=False,
    ),
    
    # ----- CATEGORIES ONLY -----
    gtsrb_categories_only = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_Final_Version.csv"),
        #images_path=PATH.joinpath("train"),~
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=CLASS_COLS,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED,
        splits=(0.9,0.1),
        use_test_set=False,
    ),
)

'''

register_datasets(

    # ----- CLASSES ONLY -----
    gtsrb = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_filtered_bin.csv"),
        #images_path=PATH.joinpath("train"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=CLASS_COLS,#CLASSES,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED
    ),

    # ----- CLASSES + CONCEPTS -----
    gtsrb_with_concepts = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_filtered_bin.csv"),
        #images_path=PATH.joinpath("train"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=SIGNCLASS_BINARY + CLASS_COLS + CONCEPTS, #CLASSES + ALL_CONCEPTS,
        features=[IMAGE_COLUMN],
        splits=(0.9,0.1),
        filter=train_filter,
        #dtypes=DTYPES,
        random_state=SEED
    ),
    
    gtsrb_with_concepts_test = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_filtered_bin.csv"),
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=SIGNCLASS_BINARY + CLASS_COLS + CONCEPTS, #CLASSES + ALL_CONCEPTS,
        features=[IMAGE_COLUMN],
        splits=(1.0, 0.0),
        filter=test_filter,
        random_state=SEED
    ),

    # ----- CONCEPTS ONLY -----
    gtsrb_concepts_only = CSVImageDatasetGTSRB(
        csv_path=PATH.joinpath("gtsrb_concepts_filtered_bin.csv"),
        #images_path=PATH.joinpath("train"),~
        images_path=PATH,
        image_columns=[(IMAGE_COLUMN, path_getter)],
        target=CONCEPTS,
        features=[IMAGE_COLUMN],
        #dtypes=DTYPES,
        random_state=SEED
    ),
)'''