import typing
import nptyping as npt

ListFloats = npt.NDArray[(1,), npt.Float64]
"""One dimentional array of floats, used for weights and predictions"""

Weights = ListFloats
Predictions = ListFloats

Labels = npt.NDArray[(1,), npt.UInt]
"""One dimentional array of ints, used for labels (with values 0 and 1)"""

Samples = npt.NDArray[(typing.Any, ...), typing.Any]
"""Numpy array of any shape and type, used for the distribution's samples"""

Dataset = npt.NDArray[(typing.Any, ...), typing.Any]
"""Numpy array of any shape and type, used for datasets"""
