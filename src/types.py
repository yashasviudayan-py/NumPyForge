"""Shared type aliases for NumPyForge.

The project uses explicit aliases to make array shape contracts easier to read
in estimator signatures and tests. Runtime shape validation lives in
`src.validation`; these aliases document the expected numerical dtypes.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike as NumpyArrayLike
from numpy.typing import NDArray

ArrayLike: TypeAlias = NDArray[Any]
BoolArray: TypeAlias = NDArray[np.bool_]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
RawArrayLike: TypeAlias = NumpyArrayLike

FeatureMatrix: TypeAlias = FloatArray
TargetVector: TypeAlias = ArrayLike
ParameterArray: TypeAlias = FloatArray | IntArray
ParameterValue: TypeAlias = ParameterArray | float | int | bool
ParameterState: TypeAlias = dict[str, ParameterValue]
PathLikeString: TypeAlias = str | PathLike[str]
RandomState: TypeAlias = int | np.random.Generator | None
