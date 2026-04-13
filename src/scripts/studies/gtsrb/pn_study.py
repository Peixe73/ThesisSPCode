import logging
from core.studies import StudyManager
from core.storage_management.study_file_manager import StudyFileManager
from core.training.trainer import TrainerConfig

from scripts.studies.gtsrb.hn_1 import ENTRY_CONCEPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PN_Study")

STUDY_NAME = "gtsrb_pn"
DATASET_NAME = "gtsrb_concepts_only" 

CONVOLUTIONS = [

    ('C1', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2
    )),
    ('C2', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2 +
        [128, ('pool', 2)] * 2
    )),
]

"""
CONVOLUTIONS = [

    ('C1', (
        [32, ('pool', 2)] +
        [64, ('pool', 2)] +
        [128, ('pool', 2)]
    )),
    
    ('C2', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] +
        [128, ('pool', 2)]
    )),

    ('C3', (
        [32, 32, ('pool', 2)] +
        [64, 64, ('pool', 2)] +
        [128, ('pool', 2)]
    )),
    
    ('C4', (
        [32, 32, ('pool', 2)] +
        [64, 64, ('pool', 2)] +
        [128, ('pool', 2)]
    )),

    ('C5', (
        [32, 32, ('pool', 2)] +
        [64, 64, ('pool', 2)] +
        [128, 128, ('pool', 2)] +
        [256, ('pool', 2)]
    )),
]
"""

"""
LINEAR_CONFIGS = [
    ('', []),

    ('_L64', [64]),
    ('_L128', [128]),
    ('_L256', [256]),

    ('_2L', [128, 64]),
    ('_2L_256', [256, 128]),

    ('_3L', [256, 128, 64]),
]
"""

LINEAR_CONFIGS = [
    ('', []),
    ('_L16', [16]),
    ('_L32', [32]),
    ('_L64', [64]),
    ('_L128', [128]),
    ('_2L', [64, 32]),
    ('_3L', [64, 32, 16]),
    ('_4L', [128, 64, 32, 16])
]

def create_trainer_config(conv_layers, linear_layers):
    return TrainerConfig(
        build_script="conv_network_pn",
        build_kwargs={
            "dataset_name": DATASET_NAME,
            "conv_layers": conv_layers,
            "linear_layers": linear_layers,
            "num_outputs": len(ENTRY_CONCEPTS),
            "hidden_activations": ('leaky_relu', 0.1)
        }
    )

def main():
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(file_manager, max_epochs=30)

    def config_generator():
        for name_c, conv in CONVOLUTIONS:
            for name_l, linear in LINEAR_CONFIGS:
                name = name_c + name_l
                yield name, create_trainer_config(conv, linear)

    logger.info("Starting PN study...")
    study_manager.run(config_generator())
    logger.info("Done.")

if __name__ == "__main__":
    main()