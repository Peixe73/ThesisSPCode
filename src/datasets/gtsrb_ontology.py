from pathlib import Path
import logging
import pandas as pd

from core.datasets import register_datasets
from core.datasets.csv_dataset import CSVDataset

logger = logging.getLogger(__name__)

PATH = Path("data/gtsrb_ontology_Final.csv")
SPLIT = (1.0, 0.0)


# VISUAL PRIMITIVES 

VISUAL_PRIMITIVES = [

    # Shapes
    "Circular_Shape",
    "Diamond_Shape",
    "Triangular_Shape",
    "Octagonal_Shape",

    # Ground colors
    "Red_Ground",
    "White_Ground",
    "Yellow_Ground",
    "Blue",

    # Borders
    "Border",
    "Black_Border",
    "Red_Border",
    "White_Border",

    # Bars
    "Bar",
    "Black_Bar",
    "White_Bar",

    # Symbol presence and color
    "Symbol",
    "Black_Symbol",
    "White_Symbol",

    # Symbol types
    "Symbol_Stop",

    "Symbol_NoEntryGoods",
    "Symbol_Overtaking",
    "Symbol_OvertakingGoods",

    "Symbol_Speed20",
    "Symbol_Speed30",
    "Symbol_Speed50",
    "Symbol_Speed60",
    "Symbol_Speed70",
    "Symbol_Speed80",
    "Symbol_Speed100",
    "Symbol_Speed120",

    "D1a1",
    "D1a4",
    "D1a5",
    "D1a6",
    "D1a7",
    "D2a1",
    "D2a2",
    "D3",
]


# SIGN CATEGORIES

SIGN_CATEGORIES = [
    "SignClass_A",
    "SignClass_B",
    "SignClass_C",
    "SignClass_D",
]


# FINAL CLASSES (LEVEL 2)

CLASSES = [

    "ClassId_1",
    "ClassId_2",
    "ClassId_3",
    "ClassId_4",
    "ClassId_5",
    "ClassId_6",
    "ClassId_7",
    "ClassId_8",
    "ClassId_9",

    "ClassId_10",
    "ClassId_11",

    "ClassId_12",

    "ClassId_13",
    "ClassId_14",
    "ClassId_15",

    "ClassId_16",
    "ClassId_17",
    "ClassId_18",

    "ClassId_33",

    "ClassId_34",
    "ClassId_35",
    "ClassId_36",
    "ClassId_37",
    "ClassId_38",
    "ClassId_39",
    "ClassId_40",
    "ClassId_41",

    "ClassId_42",
    "ClassId_43",
]


# CLASS → CATEGORY MAPPING

CLASS_TO_CATEGORY = {

    # C — prohibition / restriction
    "ClassId_1": "SignClass_C",
    "ClassId_2": "SignClass_C",
    "ClassId_3": "SignClass_C",
    "ClassId_4": "SignClass_C",
    "ClassId_5": "SignClass_C",
    "ClassId_6": "SignClass_C",
    "ClassId_7": "SignClass_C",
    "ClassId_8": "SignClass_C",
    "ClassId_9": "SignClass_C",
    "ClassId_10": "SignClass_C",
    "ClassId_11": "SignClass_C",
    "ClassId_16": "SignClass_C",
    "ClassId_17": "SignClass_C",
    "ClassId_18": "SignClass_C",
    "ClassId_33": "SignClass_C",
    "ClassId_42": "SignClass_C",
    "ClassId_43": "SignClass_C",

    # A — danger
    "ClassId_12": "SignClass_A",

    # B — priority
    "ClassId_13": "SignClass_B",
    "ClassId_14": "SignClass_B",
    "ClassId_15": "SignClass_B",

    # D — mandatory
    "ClassId_34": "SignClass_D",
    "ClassId_35": "SignClass_D",
    "ClassId_36": "SignClass_D",
    "ClassId_37": "SignClass_D",
    "ClassId_38": "SignClass_D",
    "ClassId_39": "SignClass_D",
    "ClassId_40": "SignClass_D",
    "ClassId_41": "SignClass_D",
}


# ADD CATEGORY COLUMNS

"""

def _ensure_category_columns():

    df = pd.read_csv(PATH)

    if all(c in df.columns for c in SIGN_CATEGORIES):
        return

    logger.info("Deriving A/B/C/D category columns from ClassId columns")

    for cat in SIGN_CATEGORIES:
        df[cat] = 0

    for cls, cat in CLASS_TO_CATEGORY.items():

        if cls in df.columns:
            df.loc[df[cls] == 1, cat] = 1

    df.to_csv(PATH, index=False)


_ensure_category_columns()
"""


# DATASET DEFINITIONS

visual_to_category = CSVDataset(
    path=PATH,
    target=SIGN_CATEGORIES,
    features=VISUAL_PRIMITIVES,
    splits=SPLIT
)

category_to_class = CSVDataset(
    path=PATH,
    target=CLASSES + ["valid"],
    features=SIGN_CATEGORIES,
    splits=SPLIT
)

visual_to_class = CSVDataset(
    path=PATH,
    target=CLASSES + ["valid"],
    features=VISUAL_PRIMITIVES,
    splits=SPLIT
)


# ==========================================================
# REGISTER DATASETS
# ==========================================================

register_datasets(
    gtsrb_ontology=visual_to_class,
    gtsrb_ontology_lvl1=visual_to_category,
    gtsrb_ontology_lvl2=category_to_class
)


"""
from pathlib import Path

from core.datasets import register_datasets, RandomDataset, SplitDataset
import logging

from core.datasets.csv_dataset import CSVDataset
from core.datasets import dataset_wrappers

logger = logging.getLogger(__name__)

BASIC_CONCEPTS = [
    'PassengerCar',
    'FreightWagon',
    'EmptyWagon',
    'LongWagon',
    'ReinforcedCar',
    'LongPassengerCar',
    'AtLeast2PassengerCars',
    'AtLeast2FreightWagons',
    'AtLeast3Wagons',
    'AtLeast2LongWagons'
]

INTERMEDIARY_CONCEPTS = [
    'WarTrain',
    'PassengerTrain',
    'FreightTrain',
    'RuralTrain',
    'MixedTrain',
    'LongTrain',
    'EmptyTrain',
    'LongFreightTrain'
]

CLASSES = [
    'TypeA',
    'TypeB',
    'TypeC',
    'Other',
    'valid'
]
PATH = Path('data/xtrains_ontology.csv')


COMPLETE_SPLIT = (1.0, 0.0)


basics_to_intermediary = CSVDataset(
    path = PATH,
    target = INTERMEDIARY_CONCEPTS,
    features = BASIC_CONCEPTS,
    splits=COMPLETE_SPLIT
)

intermediary_to_classes = CSVDataset(
    path = PATH,
    target = CLASSES,
    features = INTERMEDIARY_CONCEPTS,
    splits=COMPLETE_SPLIT
)

basics_to_classes = CSVDataset(
    path = PATH,
    target = CLASSES,
    features = BASIC_CONCEPTS,
    splits=COMPLETE_SPLIT
)

register_datasets(
    xtrains_ontology = basics_to_classes,
    xtrains_ontology_lvl1 = basics_to_intermediary,
    xtrains_ontology_lvl2 = intermediary_to_classes
)
"""