from typing import TYPE_CHECKING

from core.init import DO_SCRIPT_IMPORTS

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager


STUDY_NAME = "gtsrb_pn_only"

DATASET_NAME = "gtsrb_concepts_only"


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


def make_config():
    return {
        "build_script": "conv_network",
        "build_args": [],
        "build_kwargs": {
            "dataset_name": DATASET_NAME,
            "num_outputs": len(ENTRY_CONCEPTS),

            "conv_layers": [
                32, 32, ("pool", 2),
                64, ("pool", 2),
                64, ("pool", 2),
            ],
            "linear_layers": [64],

            "hidden_activations": ("leaky_relu", 0.1),
        }
    }


def main():
    file_manager = StudyFileManager(STUDY_NAME)

    study_manager = StudyManager(
        file_manager,
        max_epochs=20
    )

    configs = [
        ("pn_only", [], make_config())
    ]

    study_manager.run_with_script(
        "build_hybrid_network",
        configs
    )


if __name__ == "__main__":
    main()