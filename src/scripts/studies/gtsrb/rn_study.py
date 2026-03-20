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

LAYER_CONFIGS = [[16], [32], [64], [16,16], [32,32]]


def create_trainer_config(layer_sizes):
    return TrainerConfig(
        build_script="rn_reasoning",
        build_kwargs={
            "valid_path": DATASET_PATH,
            "feature_cols": FEATURE_COLS,
            "layer_sizes": layer_sizes,
            "dataset_size": 20000,
            "batch_size": 64,
            "classid_rules": "gtsrb_v1",
        }
    )


def main():
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(file_manager, max_epochs=50)

    def config_generator():
        for layers in LAYER_CONFIGS:
            yield f"L{layers}", create_trainer_config(layers)

    logger.info("Starting RN study with multiple layer configurations...")
    study_manager.run(config_generator())
    logger.info("RN study completed.")


if __name__ == "__main__":
    main()