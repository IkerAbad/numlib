"""Error and conditioning analysis helpers."""

from __future__ import annotations

import math

from numpy.typing import ArrayLike

from numlib._convert import as_matrix, as_vector
from numlib._ops import (
    identity,
    matmul,
    matvec,
    norm_inf_matrix,
    norm_inf_vector,
    transpose,
)
from numlib.svd import svd


def condition_number(a: ArrayLike) -> float:
    """Return the 2-norm condition number, estimated from our SVD."""
    singular_values = svd(a).S
    largest = singular_values[0]
    smallest = singular_values[-1]
    if smallest == 0.0:
        return math.inf
    return largest / smallest


def loss_of_orthogonality(q: ArrayLike) -> float:
    """Return ``||Q^T Q - I||_F``, a measure of departure from orthonormality."""
    matrix = as_matrix(q)
    n = len(matrix[0])
    gram = matmul(transpose(matrix), matrix)
    target = identity(n)
    total = 0.0
    for i in range(n):
        for j in range(n):
            difference = gram[i][j] - target[i][j]
            total += difference * difference
    return math.sqrt(total)


def backward_error(a: ArrayLike, x: ArrayLike, b: ArrayLike) -> float:
    """Normwise relative backward error of a square system.

    ``eta = ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf)``.
    """
    matrix = as_matrix(a)
    solution = as_vector(x)
    rhs = as_vector(b)
    if len(matrix) != len(solution):
        raise ValueError("A and x have incompatible shapes")
    residual = [rhs[i] - value for i, value in enumerate(matvec(matrix, solution))]
    denominator = norm_inf_matrix(matrix) * norm_inf_vector(solution) + norm_inf_vector(rhs)
    if denominator == 0.0:
        return 0.0
    return norm_inf_vector(residual) / denominator


def forward_error(x: ArrayLike, reference: ArrayLike) -> float:
    """Return ``||x - reference||_inf / ||reference||_inf``."""
    actual = as_vector(x)
    expected = as_vector(reference)
    if len(actual) != len(expected):
        raise ValueError("vectors have different lengths")
    difference = [a - e for a, e in zip(actual, expected, strict=True)]
    denominator = norm_inf_vector(expected)
    if denominator == 0.0:
        return math.inf
    return norm_inf_vector(difference) / denominator
