import logging
from pathlib import Path
from core.studies import StudyManager
from core.storage_management.study_file_manager import StudyFileManager
from analysis_tools.gtsrb_utils import CLASS_COLS, CONCEPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RN_Study")

STUDY_NAME = "gtsrb_rn_Multiple_ConceptN"
DATASET_PATH = Path("data/gtsrb_ontology_Valid_Final.csv")



CONFIGS = [
    # ONTOLOGY MODELS
    
    ("ONTO_BASE",   {"type": "ontology", "pre": [], "post": []}),
        
    #("ONTO_32",     {"type": "ontology", "pre": [32], "post": [32]}),
    #("ONTO_64",     {"type": "ontology", "pre": [64], "post": [64]}),
    #("ONTO_2L64",   {"type": "ontology", "pre": [64, 16], "post": [16, 64]}),
    #("ONTO_2L32",    {"type": "ontology", "pre": [32, 12], "post": [12, 32]}),
    
    #("DIRECT_BASE", {"type": "direct"}),

    # CONTROL MLPs
    #("CTRL_0",      {"type": "mlp", "layers": []}),
    #("CTRL_16",     {"type": "mlp", "layers": [16]}),
    #("CTRL_32",     {"type": "mlp", "layers": [32]}),
    #("CTRL_64",     {"type": "mlp", "layers": [64]}),
    #("CTRL_128",    {"type": "mlp", "layers": [128]}),
    #("CTRL_2L",   {"type": "mlp", "layers": [64, 128]}),
    #("CTRL_3L",   {"type": "mlp", "layers": [64, 128, 64]}),
]

NOISE_MODES = [
    "binary",
    "uniform",
    "extremes",
    "middle"
]

BASE_SEED = 42
NUM_RUNS = 5

"""
def make_config(cfg):
    return {
        "valid_path": str(DATASET_PATH),
        "feature_cols": CONCEPTS,
        "class_cols": CLASS_COLS,
        "dataset_size": 6400,
        "batch_size": 64,
        "model_config": cfg,
        "base_seed": base_seed,
    }
"""
def make_config(cfg, seed):
    return {
        "valid_path": str(DATASET_PATH),
        "feature_cols": CONCEPTS,
        "class_cols": CLASS_COLS,
        "dataset_size": 6400,
        "batch_size": 64,
        "model_config": cfg,
        "base_seed": seed,
    }

def main():
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(file_manager, max_epochs=1000)

    #configs = [(name, [], make_config(cfg)) for name, cfg in CONFIGS]
    """
    configs = []

    for noise_mode in NOISE_MODES:
        for name, cfg in CONFIGS:

            cfg_name = f"{name}_{noise_mode}"

            configs.append((
                cfg_name,
                [],
                make_config(cfg) | {
                    "concept_noise_mode": noise_mode
                }
            ))
    """
    
    configs = []

    for run in range(NUM_RUNS):
        seed = BASE_SEED + run * 1000

        for noise_mode in NOISE_MODES:
            for name, cfg in CONFIGS:

                cfg_name = f"{name}_{noise_mode}_run{run}"

                configs.append((
                    cfg_name,
                    [],
                    make_config(cfg, seed) | {
                        "concept_noise_mode": noise_mode,
                        #"run": run,          
                    }
                ))

    logger.info("Starting RN study...")
    study_manager.run_with_script("rn_reasoning", configs)
    logger.info("Done.")


if __name__ == "__main__":
    main()