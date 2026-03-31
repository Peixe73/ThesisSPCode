from typing import TYPE_CHECKING

from core.init import DO_SCRIPT_IMPORTS
if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager

STUDY_NAME=f"gtsrb{__name__.split('.')[-1]}"

DATASET_NAME = "gtsrb"

ENTRY_CONCEPTS = [
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


RN_WITH_WEIGHTS = {
    "model_name" : "L[32]",
    "model_path" : "storage/studies/gtsrb_rn"
}


"""
RN_WITHOUT_WEIGHTS = {
    "build_script" : "linear_rn",
    "build_args" : [],
    "build_kwargs" : {
        "dataset_name" : "gtsrb_ontology",
        "layer_sizes" : [32],
        "num_outputs" : 30
    }
}
"""

RN_WITHOUT_WEIGHTS = {
    "build_script" : "rn_reasoning",
    "build_args" : [],
    "build_kwargs" : {
        "valid_path" : "data/gtsrb_ontology_Valid_Final.csv",
        "feature_cols" : ENTRY_CONCEPTS,
        "class_cols" : CLASS_COLS,
        "dataset_size" : 6400,
        "batch_size" : 64,
        "base_seed" : 42,
        "layer_sizes" : [32],
    }
}


def make_pn_config(kwargs):
    return {
        "build_script" : "conv_network",
        "build_args" : [],
        "build_kwargs" : {
            'dataset_name': DATASET_NAME,
            'num_outputs' : len(ENTRY_CONCEPTS),
            'hidden_activations' : ('leaky_relu', 0.1)
        } | kwargs
    }

def make_config(rn_config, pn_kwargs, extra_kwargs):
    kwargs = {
        "dataset_name": DATASET_NAME,
        "concept_dataset_name": "gtsrb_concepts_only",
        "concepts": ENTRY_CONCEPTS,
        "pre_trained_learning_rate" : 0.001,
        "untrained_learning_rate" : 0.001,
        "reasoning_network_config" : rn_config,
        "perception_network_config" : make_pn_config(pn_kwargs),
        **extra_kwargs
    }
    return kwargs

# noinspection DuplicatedCode
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

def make_configs():
    configs = []
    for name_linear, linear in LINEAR_CONFIGS:
        for name_conv, conv in CONVOLUTIONS:
            name = name_conv + name_linear
            pn_kwargs = {
                "conv_layers": conv,
                "linear_layers": linear
            }
            configs.append((name + '_untRN', [],
                            make_config(
                                RN_WITHOUT_WEIGHTS,
                                pn_kwargs,
                                {'skip_pn_eval': True,
                                 'rn_learning_rate': 0.001,
                                 'activation': 'relu'}
                            )))
            configs.append((name, [],
                            make_config(
                                RN_WITH_WEIGHTS,
                                pn_kwargs,
                                {}
                            )))
    return configs

def main():
    # Load study manager
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(
        file_manager,
        max_epochs=20
    )

    configs = make_configs()
    study_manager.run_with_script('build_hybrid_network', configs)
