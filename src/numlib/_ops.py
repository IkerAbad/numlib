"""Elementary dense-matrix operations used across the package."""

from __future__ import annotations

import math

from numlib._types import Matrix, Vector


def zeros(rows: int, cols: int) -> Matrix:
    """Return a ``rows x cols`` matrix of zeros."""
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    """Return the ``n x n`` identity matrix."""
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def transpose(a: Matrix) -> Matrix:
    """Return the transpose of ``a``."""
    return [list(row) for row in zip(*a, strict=True)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two dense matrices with plain Python loops."""
    if not a or not b:
        raise ValueError("matrices must be non-empty")
    if len(a[0]) != len(b):
        raise ValueError(f"shape mismatch: {len(a)}x{len(a[0])} times {len(b)}x{len(b[0])}")
    b_columns = transpose(b)
    return [
        [sum(x * y for x, y in zip(row, column, strict=True)) for column in b_columns] for row in a
    ]


def matvec(a: Matrix, x: Vector) -> Vector:
    """Multiply a dense matrix by a vector."""
    if not a or len(a[0]) != len(x):
        raise ValueError("matrix and vector have incompatible shapes")
    return [sum(ai * xi for ai, xi in zip(row, x, strict=True)) for row in a]


def dot(x: Vector, y: Vector) -> float:
    """Return the Euclidean inner product of two vectors."""
    return sum(a * b for a, b in zip(x, y, strict=True))


def norm2(x: Vector) -> float:
    """Return the Euclidean norm, computed without intermediate overflow."""
    return math.hypot(*x)


def norm_inf_vector(x: Vector) -> float:
    """Return the infinity norm of a vector."""
    return max((abs(value) for value in x), default=0.0)


def norm_inf_matrix(a: Matrix) -> float:
    """Return the induced infinity norm (maximum absolute row sum)."""
    return max((sum(abs(value) for value in row) for row in a), default=0.0)


def frobenius(a: Matrix) -> float:
    """Return the Frobenius norm of a matrix."""
    return math.hypot(*(value for row in a for value in row))


def permutation_matrix(perm: list[int]) -> Matrix:
    """Build the row-permutation matrix with ones at ``(i, perm[i])``."""
    n = len(perm)
    out = zeros(n, n)
    for i, column in enumerate(perm):
        out[i][column] = 1.0
    return out


def column_permutation_matrix(perm: list[int]) -> Matrix:
    """Build the column-permutation matrix with ones at ``(perm[j], j)``."""
    n = len(perm)
    out = zeros(n, n)
    for j, row in enumerate(perm):
        out[row][j] = 1.0
    return out
