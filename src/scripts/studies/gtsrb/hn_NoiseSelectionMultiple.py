from typing import TYPE_CHECKING

from core.init import DO_SCRIPT_IMPORTS
if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager

from analysis_tools.gtsrb_utils import CLASS_COLS, CONCEPTS
STUDY_NAME=f"gtsrb_{__name__.split('.')[-1]}"

DATASET_NAME = "gtsrb"

BASE_SEED = 42
NUM_RUNS = 5

RN_WITHOUT_WEIGHTS = {
    "build_script": "rn_reasoning",
    "build_args": [],
    "build_kwargs": {
        "valid_path": "data/gtsrb_ontology_Valid_Final.csv",
        "feature_cols": CONCEPTS,
        "class_cols": CLASS_COLS,
        "dataset_size": 6400,
        "batch_size": 64,
        "base_seed": 42,

        "model_config": {
            "type": "ontology",
            "pre": [32],
            "post": [32]
        }
    }
}

RN_ARCHITECTURES = [
    # ONTOLOGY MODELS
    ("ONTO_BASE",   {"type": "ontology", "pre": [], "post": []}),
    
    #("ONTO_32",     {"type": "ontology", "pre": [32], "post": [32]}),
    #("ONTO_64",     {"type": "ontology", "pre": [64], "post": [64]}),
    #("ONTO_2L",   {"type": "ontology", "pre": [64, 16], "post": [16, 64]}),
    #("ONTO_128",    {"type": "ontology", "pre": [128], "post": [128]}),
    
    #("DIRECT_BASE", {"type": "direct"}),

    # CONTROL MLPs
    #("CTRL_0",      {"type": "mlp", "layers": []}),
    #("CTRL_16",     {"type": "mlp", "layers": [16]}),
    #("CTRL_32",     {"type": "mlp", "layers": [32]}),
    #("CTRL_64",     {"type": "mlp", "layers": [64]}),
    #("CTRL_2L",   {"type": "mlp", "layers": [64, 128]}),
    #("CTRL_3L",   {"type": "mlp", "layers": [32, 64, 128]}),
    #("CTRL_128",    {"type": "mlp", "layers": [128]}),
]

def make_untrained_rn_config(model_config, seed, noise_mode="binary"):
    return {
        "build_script": "rn_reasoning",
        "build_args": [],
        "build_kwargs": {
            "valid_path": "data/gtsrb_ontology_Valid_Final.csv",
            "feature_cols": CONCEPTS,
            "class_cols": CLASS_COLS,
            "dataset_size": 6400,
            "batch_size": 64,
            "base_seed": seed,
            "model_config": model_config,
            "concept_noise_mode": noise_mode,
        }
    }

def make_pretrained_rn_config(model_name, noise_mode="binary"):
    return {
        "model_name": f"{model_name}_{noise_mode}",
        "model_path": "storage/studies/gtsrb_rn_test",
    }

def make_pn_config(kwargs):
    return {
        "build_script" : "conv_network",
        "build_args" : [],
        "build_kwargs" : {
            'dataset_name': DATASET_NAME,
            'num_outputs' : len(CONCEPTS),
            'hidden_activations' : ('leaky_relu', 0.1)
        } | kwargs
    }

def make_config(rn_config, pn_kwargs, extra_kwargs):
    kwargs = {
        "dataset_name": DATASET_NAME,
        "concept_dataset_name": "gtsrb_concepts_only",
        "concepts": CONCEPTS,
        "pre_trained_learning_rate" : 0.001,
        "untrained_learning_rate" : 0.001,
        "reasoning_network_config" : rn_config,
        "perception_network_config" : make_pn_config(pn_kwargs),
        **extra_kwargs
    }
    return kwargs

# noinspection DuplicatedCode
CONVOLUTIONS = [
    ('C2', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2 +
        [128, ('pool', 2)] * 2
    )),
]

""",
    ('C2', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2 +
        [128, ('pool', 2)] * 2
    )),"""

LINEAR_CONFIGS = [
    #('', []),
    #('_L16', [16]),
    #('_L32', [32]),
    #('_L64', [64]),
    ('_L128', [128]),
    #('_2L', [64, 32]),
    #('_3L', [64, 32, 16]),
    #('_4L', [128, 64, 32, 16])
]

NOISE_MODES = [
    "binary",
    "uniform",
    "extremes",
    "middle"
]


def make_configs():
    configs = []

    for run in range(NUM_RUNS):
        seed = BASE_SEED + run * 1000

        for rn_name, rn_model_cfg in RN_ARCHITECTURES:

            for name_linear, linear in LINEAR_CONFIGS:

                for name_conv, conv in CONVOLUTIONS:

                    pn_name = name_conv + name_linear

                    pn_kwargs = {
                        "conv_layers": conv,
                        "linear_layers": linear
                    }
                    """
                    configs.append((
                        #f"{pn_name}_{rn_name}_{noise_mode}_untRN",
                        f"{pn_name}_{rn_name}_untRN",
                        [],
                        make_config(
                            make_untrained_rn_config(rn_model_cfg),
                            #make_untrained_rn_config(rn_model_cfg, noise_mode),
                            pn_kwargs,
                            {
                                "rn_learning_rate": 0.001,
                            }
                        )
                    ))
                    """
                    for noise_mode in NOISE_MODES:

                        configs.append((
                            f"{pn_name}_{rn_name}_{noise_mode}_preRN_run{run}",
                            [],
                            make_config(
                                make_pretrained_rn_config(rn_name, noise_mode),
                                pn_kwargs,
                                {
                                    "rn_learning_rate": 0.001,
                                }
                            )
                        ))

    return configs

def main():
    # Load study manager
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(
        file_manager,
        max_epochs=1000
    )

    configs = make_configs()
    study_manager.run_with_script('build_hybrid_network', configs)
