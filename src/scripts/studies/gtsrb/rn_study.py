import logging
from core.studies import StudyManager
from core.storage_management.study_file_manager import StudyFileManager
from core.training.trainer import TrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RN_Study")

STUDY_NAME = "gtsrb_rn"
DATASET_PATH = "data/gtsrb_ontology_Valid_Final.csv"

FEATURE_COLS = [
    "Circular_Shape","Diamond_Shape","Triangular_Shape","Octagonal_Shape",
    "Red_Ground","White_Ground","Yellow_Ground","Blue",
    "Border","Black_Border","Red_Border","White_Border",
    "Bar","Black_Bar","White_Bar","Symbol","Black_Symbol","White_Symbol",
    "Symbol_Stop", "Symbol_NoEntryGoods", "Symbol_Overtaking", "Symbol_OvertakingGoods",
    "Symbol_Speed20", "Symbol_Speed30", "Symbol_Speed50",
    "Symbol_Speed60", "Symbol_Speed70", "Symbol_Speed80",
    "Symbol_Speed100", "Symbol_Speed120",
    "D1a1", "D1a4", "D1a5", "D1a6", "D1a7",
    "D2a1", "D2a2", "D3",
]

CLASS_COLS = [
    "ClassId_1","ClassId_2","ClassId_3","ClassId_4","ClassId_5",
    "ClassId_6","ClassId_7","ClassId_8","ClassId_9","ClassId_10",
    "ClassId_11","ClassId_12","ClassId_13","ClassId_14","ClassId_15",
    "ClassId_16","ClassId_17","ClassId_18","ClassId_33","ClassId_34",
    "ClassId_35","ClassId_36","ClassId_37","ClassId_38","ClassId_39",
    "ClassId_40","ClassId_41","ClassId_42","ClassId_43",
]

INTERMEDIARY_CATEGORYS = [
    "SignClass_A", "SignClass_B", "SignClass_C", "SignClass_D"
]

LAYER_CONFIGS = [[16], [32], [64], [16,16], [32,32]]


def create_trainer_config(layer_sizes):
    return TrainerConfig(
        build_script="rn_reasoning",
        build_kwargs={
            "valid_path": DATASET_PATH,
            "feature_cols": FEATURE_COLS,
            "class_cols": CLASS_COLS,
            "layer_sizes": layer_sizes,
            "dataset_size": 640,
            "batch_size": 64,
        }
    )


def main():
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(file_manager, max_epochs=100)

    def config_generator():
        for layers in LAYER_CONFIGS:
            yield f"L{layers}", create_trainer_config(layers)

    logger.info("Starting RN study...")
    study_manager.run(config_generator())
    logger.info("Done.")


if __name__ == "__main__":
    main()