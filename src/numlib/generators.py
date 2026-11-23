"""Test matrices with controlled spectral properties."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from numlib._types import Matrix
from numlib.qr import qr_householder


def hilbert(n: int) -> Matrix:
    """Hilbert matrix ``H_ij = 1 / (i + j + 1)``, symmetric positive definite."""
    if n < 1:
        raise ValueError("n must be positive")
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def vandermonde(x: ArrayLike, degree: int | None = None) -> Matrix:
    """Vandermonde matrix with columns ``x^0, ..., x^(degree-1)``."""
    nodes = [float(value) for value in np.asarray(x, dtype=np.float64).ravel()]
    if not nodes:
        raise ValueError("at least one node is required")
    if degree is None:
        degree = len(nodes)
    if degree < 1:
        raise ValueError("degree must be positive")
    return [[node**power for power in range(degree)] for node in nodes]


def random_gaussian(m: int, n: int, *, seed: int | None = None) -> Matrix:
    """Return an ``m x n`` matrix with standard normal entries."""
    if m < 1 or n < 1:
        raise ValueError("dimensions must be positive")
    rng = np.random.default_rng(seed)
    return [[float(value) for value in row] for row in rng.standard_normal((m, n))]


def with_condition(m: int, n: int, condition: float, *, seed: int | None = None) -> Matrix:
    """Return a random ``m x n`` matrix with a prescribed condition number.

    The matrix is ``Q diag(s) V^T`` with ``Q`` and ``V`` orthogonal and the
    singular values geometrically spaced from 1 down to ``1 / condition``.
    """
    if m < n:
        raise ValueError("m must be at least n")
    if condition < 1.0:
        raise ValueError("condition number must be at least 1")

    rng = np.random.default_rng(seed)
    gaussian_q = [[float(value) for value in row] for row in rng.standard_normal((m, n))]
    q = qr_householder(gaussian_q).Q
    gaussian_v = [[float(value) for value in row] for row in rng.standard_normal((n, n))]
    v = qr_householder(gaussian_v).Q

    if n == 1:
        singular_values = [1.0]
    else:
        exponent = math.log10(condition)
        singular_values = [10.0 ** (-exponent * k / (n - 1)) for k in range(n)]

    return [
        [sum(q[i][k] * singular_values[k] * v[j][k] for k in range(n)) for j in range(n)]
        for i in range(m)
    ]


def wilkinson(n: int) -> Matrix:
    """Symmetric tridiagonal Wilkinson matrix with clustered eigenvalues."""
    if n < 1:
        raise ValueError("n must be positive")
    center = (n - 1) / 2.0
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = abs(i - center)
        if i + 1 < n:
            matrix[i][i + 1] = 1.0
            matrix[i + 1][i] = 1.0
    return matrix
