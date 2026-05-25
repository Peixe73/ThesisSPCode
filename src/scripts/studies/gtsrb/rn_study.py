import logging
from pathlib import Path
from core.studies import StudyManager
from core.storage_management.study_file_manager import StudyFileManager
from core.training.trainer import TrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RN_Study")

STUDY_NAME = "gtsrb_rn"
DATASET_PATH = Path("data/gtsrb_ontology_Valid_Final.csv")

FEATURE_COLS = [
    "Circular_Shape","Diamond_Shape","Triangular_Shape","Octagonal_Shape",
    "Red_Ground","White_Ground","Yellow_Ground","Blue",
    "Border","Black_Border","Red_Border","White_Border",
    "Bar","Black_Bar","White_Bar",
    "Symbol","Black_Symbol","White_Symbol",
    "Symbol_Stop","Symbol_NoEntryGoods","Symbol_Overtaking",
    "Symbol_OvertakingGoods","Symbol_Speed20","Symbol_Speed30",
    "Symbol_Speed50","Symbol_Speed60","Symbol_Speed70",
    "Symbol_Speed80","Symbol_Speed100","Symbol_Speed120",
    "D1a1","D1a4","D1a5","D1a6","D1a7","D2a1","D2a2","D3",
]

CLASS_COLS = [
    "ClassId_1","ClassId_2","ClassId_3","ClassId_4","ClassId_5",
    "ClassId_6","ClassId_7","ClassId_8","ClassId_9","ClassId_10",
    "ClassId_11","ClassId_12","ClassId_13","ClassId_14","ClassId_15",
    "ClassId_16","ClassId_17","ClassId_18","ClassId_33","ClassId_34",
    "ClassId_35","ClassId_36","ClassId_37","ClassId_38","ClassId_39",
    "ClassId_40","ClassId_41","ClassId_42","ClassId_43",
]

"""
CONFIGS = [
    # ONTOLOGY-ALIGNED BASELINE

    (
        "ONTO_BASE",
        {
            "architecture": "ontology",
            "pre_layers": [],
            "post_layers": [],
        }
    ),

    # ONTOLOGY-ALIGNED EXPANDED

    (
        "ONTO_32",
        {
            "architecture": "ontology",
            "pre_layers": [32],
            "post_layers": [32],
        }
    ),

    (
        "ONTO_64",
        {
            "architecture": "ontology",
            "pre_layers": [64],
            "post_layers": [64],
        }
    ),

    (
        "ONTO_64x2",
        {
            "architecture": "ontology",
            "pre_layers": [64, 64],
            "post_layers": [64, 64],
        }
    ),

    (
        "ONTO_128",
        {
            "architecture": "ontology",
            "pre_layers": [128],
            "post_layers": [128],
        }
    ),

    # CONTROL NETWORKS

    (
        "CTRL_LINEAR",
        {
            "architecture": "control",
            "layers": [],
        }
    ),

    (
        "CTRL_16",
        {
            "architecture": "control",
            "layers": [16],
        }
    ),

    (
        "CTRL_32",
        {
            "architecture": "control",
            "layers": [32],
        }
    ),

    (
        "CTRL_64",
        {
            "architecture": "control",
            "layers": [64],
        }
    ),

    (
        "CTRL_64x2",
        {
            "architecture": "control",
            "layers": [64, 64],
        }
    ),
    
    (
        "CTRL_128",
        {
            "architecture": "control",
            "layers": [128],
        }
    ),
    
]
"""

CONFIGS=[
    ('L16', [16]),
    ('L32', [32]),
    ('L64', [64]),
    ('L16x2', [16, 16]),
    ('L16L32', [16, 32]),
    ('L32x2', [32, 32]),
    ('L32L64', [32, 64]),
    ('L64x2', [64, 64]),
    ('L16x3', [16, 16, 16]),
    ('L16L32x2', [16, 32, 32]),
    ('L32x3', [32, 32, 32]),
    ('L16L32L64', [16, 32, 64]),
    ('L64x3', [64, 64, 64]),
]


"""
def make_config(layer_sizes):
    return {
        "build_script": "rn_reasoning",
        "build_args": [],
        "build_kwargs": {
            "valid_path": str(DATASET_PATH),
            "feature_cols": FEATURE_COLS,
            "class_cols": CLASS_COLS,
            "dataset_size": 1200,
            "batch_size": 64,
            "layer_sizes": layer_sizes,
            "base_seed": 42,
        }
    }
"""
    
def make_config(layer_sizes):
    return {
        "valid_path": str(DATASET_PATH),
        "feature_cols": FEATURE_COLS,
        "class_cols": CLASS_COLS,
        "dataset_size": 1200,
        "batch_size": 64,
        "layer_sizes": layer_sizes,
        "base_seed": 42,
    }
"""
def make_config(layer_sizes):
    return TrainerConfig(
        build_script = "rn_reasoning",
        build_args = [],
        build_kwargs={
            "valid_path": str(DATASET_PATH),
            "feature_cols": FEATURE_COLS,
            "class_cols": CLASS_COLS,
            "dataset_size": 1200,
            "batch_size": 64,
            "layer_sizes": layer_sizes,
            "base_seed": 42,
        }
    )
"""
"""
def create_trainer_config(layer_sizes):
    return TrainerConfig(
        build_script="rn_reasoning",
        build_args=[],
        build_kwargs={
            "valid_path": str(DATASET_PATH),
            "feature_cols": FEATURE_COLS,
            "class_cols": CLASS_COLS,
            "layer_sizes": layer_sizes,
            "dataset_size": 1200,
            "batch_size": 64,
        }
    )
"""

def main():
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(file_manager, max_epochs=100)

    configs = [
        (name, [], make_config(layers))
        for name, layers in CONFIGS
    ]

    logger.info("Starting RN study...")
    study_manager.run_with_script("rn_reasoning", configs)
    logger.info("Done.")


if __name__ == "__main__":
    main()