from . import (
    xtrains_ontology,
    xtrains,
    gtsrb,
    gtsrb_ontology
)

import logging
from core.datasets import _dataset_registry
dataset_names = '\n'.join(_dataset_registry.keys())
logging.getLogger(__name__).debug(f"Registered datasets:\n{dataset_names}")