"""Triangular solves used by the LU and least-squares code."""

from __future__ import annotations

from numlib._types import Matrix, Vector
from numlib.exceptions import SingularMatrixError


def solve_lower_unit(lower: Matrix, rhs: Vector) -> Vector:
    """Solve ``L y = b`` for a unit lower triangular ``L``."""
    n = len(lower)
    y = [0.0] * n
    for i in range(n):
        acc = rhs[i]
        for j in range(i):
            acc -= lower[i][j] * y[j]
        y[i] = acc
    return y


def solve_upper(upper: Matrix, rhs: Vector) -> Vector:
    """Solve ``U x = b`` for an upper triangular ``U``."""
    n = len(upper)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = rhs[i]
        for j in range(i + 1, n):
            acc -= upper[i][j] * x[j]
        pivot = upper[i][i]
        if pivot == 0.0:
            raise SingularMatrixError("zero pivot in back substitution")
        x[i] = acc / pivot
    return x
