#Old Version
'''
import sys
from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS
if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.datasets import SplitDataset, get_dataset
    from core.training import Trainer
    from core.storage_management import ModelFileManager

TRAINER_PREFIX = 'from_trainer:'

def main():
    dataset_name : str = sys.argv[0]
    dataset : SplitDataset
    if dataset_name.startswith(TRAINER_PREFIX):
        dataset_name = dataset_name[len(TRAINER_PREFIX):]
        with ModelFileManager(path=dataset_name) as file_manager:
            config = file_manager.load_config()
            trainer = Trainer.from_config(config)
            training_set = trainer.training_set

            if isinstance(training_set, SplitDataset):
                dataset = training_set
            elif hasattr(training_set, 'dataset'):
                dataset = training_set.dataset #type: ignore
            else:
                raise ValueError(f"Cannot get dataset from {sys.argv[0]}")
    else:
        dataset = get_dataset(dataset_name)

        
    raise NotImplementedError()
    '''

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path

from core.init import DO_SCRIPT_IMPORTS

from core.init.options_parsing import option, positional

from analysis_tools.datasets import analyze_dataset

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.datasets import SplitDataset, get_dataset
    from core.training import Trainer
    from core.storage_management import ModelFileManager


TRAINER_PREFIX = "from_trainer:"


@dataclass
class Options:
    dataset: str = field(
        metadata=positional(str, help_="Dataset name or 'from_trainer:<model_path>'")
    )


def run(options: Options):
    dataset_name = options.dataset
    dataset: "SplitDataset"

    # Case 1: Load dataset from trainer config
    if dataset_name.startswith(TRAINER_PREFIX):
        model_path = dataset_name[len(TRAINER_PREFIX):]

        #model_path = global_options.models_path / model_path

        with ModelFileManager(path=model_path) as file_manager:
            config = file_manager.load_config()
            trainer = Trainer.from_config(config)
            training_set = trainer.training_set

            if isinstance(training_set, SplitDataset):
                dataset = training_set
            elif hasattr(training_set, "dataset"):
                dataset = training_set.dataset  # type: ignore
            else:
                raise ValueError(
                    f"Cannot extract dataset from trainer at {model_path}"
                )

    # Case 2: Load dataset normally
    else:
        dataset = get_dataset(dataset_name)

    #raise NotImplementedError("Dataset analysis not implemented yet")
    analyze_dataset(dataset, Path('analysis_results'), dataset_description=dataset_name, class_names=dataset.class_names)

def main(options: Options):
    return run(options)