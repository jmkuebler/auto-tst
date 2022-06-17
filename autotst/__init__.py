# flake8: noqa

import importlib_metadata

__version__ = importlib_metadata.version("autotst")

from .autotst_types import Weights, Predictions, Labels, Samples, Dataset
from .test import AutoTST
from .model import AutoGluonTabularPredictor
from .functions import (
    permutations_p_value,
    get_weights,
    fit_witness,
    p_value_evaluate,
    p_value,
    interpret,
)
from .splitted_sets import SplittedSets
