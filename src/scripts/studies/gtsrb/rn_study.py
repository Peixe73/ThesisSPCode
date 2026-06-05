import logging
from pathlib import Path
from core.studies import StudyManager
from core.storage_management.study_file_manager import StudyFileManager
from analysis_tools.gtsrb_utils import CLASS_COLS, CONCEPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RN_Study")

STUDY_NAME = "gtsrb_rn"
DATASET_PATH = Path("data/gtsrb_ontology_Valid_Final.csv")



CONFIGS = [
    # ONTOLOGY MODELS
    ("ONTO_BASE",   {"type": "ontology", "pre": [], "post": []}),
    
    ("ONTO_32",     {"type": "ontology", "pre": [32], "post": [32]}),
    ("ONTO_64",     {"type": "ontology", "pre": [64], "post": [64]}),
    ("ONTO_64x2",   {"type": "ontology", "pre": [64, 64], "post": [64, 64]}),
    ("ONTO_128",    {"type": "ontology", "pre": [128], "post": [128]}),
    
    # Direct Baseline (38->30) skips the category bottleneck
    ("DIRECT_BASE", {"type": "direct"}),

    # CONTROL MLPs
    ("CTRL_0",      {"type": "mlp", "layers": []}),
    ("CTRL_16",     {"type": "mlp", "layers": [16]}),
    ("CTRL_32",     {"type": "mlp", "layers": [32]}),
    ("CTRL_64",     {"type": "mlp", "layers": [64]}),
    ("CTRL_64x2",   {"type": "mlp", "layers": [64, 64]}),
    ("CTRL_128",    {"type": "mlp", "layers": [128]}),
]


def make_config(cfg):
    return {
        "valid_path": str(DATASET_PATH),
        "feature_cols": CONCEPTS,
        "class_cols": CLASS_COLS,
        "dataset_size": 6400,
        "batch_size": 64,
        "model_config": cfg,
        "base_seed": 42,
    }


def main():
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(file_manager, max_epochs=1000)

    configs = [(name, [], make_config(cfg)) for name, cfg in CONFIGS]

    logger.info("Starting RN study...")
    study_manager.run_with_script("rn_reasoning", configs)
    logger.info("Done.")


if __name__ == "__main__":
    main()