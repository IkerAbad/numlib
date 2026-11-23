"""Conversions at the NumPy boundary.

The algorithms in numlib operate on plain Python lists of floats. NumPy is
used only at the boundary: to accept array-like inputs, to hand arrays back to
the caller, and in the benchmark code that compares against LAPACK.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numlib._types import Matrix, Vector


def as_matrix(a: ArrayLike) -> Matrix:
    """Copy an array-like object into a list of lists of floats."""
    array = np.asarray(a, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("matrix must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix entries must be finite")
    return [[float(value) for value in row] for row in array]


def as_vector(v: ArrayLike) -> Vector:
    """Copy an array-like object into a list of floats."""
    array = np.asarray(v, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("expected a one-dimensional array")
    if array.size == 0:
        raise ValueError("vector must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("vector entries must be finite")
    return [float(value) for value in array]


def as_square(a: ArrayLike) -> Matrix:
    """Copy an array-like object into a square list of lists of floats."""
    out = as_matrix(a)
    n = len(out)
    if any(len(row) != n for row in out):
        raise ValueError("matrix must be square")
    return out


def to_array(a: Matrix) -> NDArray[np.float64]:
    """Convert a list of lists into a NumPy array."""
    return np.array(a, dtype=np.float64)
