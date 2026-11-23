"""QR factorizations.

Thin QR is returned: for an ``m x n`` matrix with ``m >= n``, ``Q`` is
``m x n`` with orthonormal columns and ``R`` is ``n x n`` upper triangular.

Two methods are implemented:

* Householder reflections, backward stable and the default choice;
* modified Gram-Schmidt (with optional reorthogonalization), cheaper but
  more sensitive to the condition number of the input. Classical
  Gram-Schmidt is also available for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike

from numlib._convert import as_matrix
from numlib._ops import dot, norm2, zeros
from numlib._types import Matrix
from numlib.exceptions import RankDeficientError

Method = Literal["householder", "modified-gram-schmidt", "classical-gram-schmidt"]


@dataclass(frozen=True)
class QR:
    """Thin QR factorization ``A = Q R``."""

    Q: Matrix
    R: Matrix
    method: Method


def qr_householder(a: ArrayLike) -> QR:
    """Compute a thin QR factorization with Householder reflections."""
    matrix = as_matrix(a)
    m = len(matrix)
    n = len(matrix[0])
    if m < n:
        raise ValueError("Householder QR requires at least as many rows as columns")

    r = [row[:] for row in matrix]
    reflectors: list[tuple[int, list[float], float]] = []

    for k in range(n):
        x = [r[i][k] for i in range(k, m)]
        norm_x = norm2(x)
        if norm_x == 0.0:
            continue
        alpha = -norm_x if x[0] >= 0.0 else norm_x
        v = x
        v[0] -= alpha
        v_norm_sq = dot(v, v)
        if v_norm_sq == 0.0:
            r[k][k] = alpha
            for i in range(k + 1, m):
                r[i][k] = 0.0
            continue
        beta = 2.0 / v_norm_sq
        reflectors.append((k, v, beta))

        for j in range(k, n):
            s = 0.0
            for i in range(k, m):
                s += v[i - k] * r[i][j]
            for i in range(k, m):
                r[i][j] -= beta * v[i - k] * s
        r[k][k] = alpha
        for i in range(k + 1, m):
            r[i][k] = 0.0

    # Thin Q: the first n columns of the product of the reflectors.
    q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(m)]
    for k, v, beta in reversed(reflectors):
        for j in range(n):
            s = 0.0
            for i in range(k, m):
                s += v[i - k] * q[i][j]
            for i in range(k, m):
                q[i][j] -= beta * v[i - k] * s

    r_thin = [row[:n] for row in r[:n]]
    return QR(Q=q, R=r_thin, method="householder")


def qr_gram_schmidt(
    a: ArrayLike,
    *,
    modified: bool = True,
    reorthogonalize: bool = False,
) -> QR:
    """Compute a thin QR factorization by (modified) Gram-Schmidt.

    Reorthogonalization applies a second MGS pass and restores orthogonality
    for moderately ill-conditioned matrices.
    """
    matrix = as_matrix(a)
    m = len(matrix)
    n = len(matrix[0])
    if m < n:
        raise ValueError("Gram-Schmidt QR requires at least as many rows as columns")
    if reorthogonalize and not modified:
        raise ValueError("reorthogonalization only applies to modified Gram-Schmidt")

    q_columns: list[list[float]] = []
    r = zeros(n, n)

    for j in range(n):
        v = [matrix[i][j] for i in range(m)]
        if modified:
            for i in range(j):
                qi = q_columns[i]
                coefficient = dot(qi, v)
                r[i][j] = coefficient
                for k in range(m):
                    v[k] -= coefficient * qi[k]
            if reorthogonalize:
                for i in range(j):
                    qi = q_columns[i]
                    coefficient = dot(qi, v)
                    r[i][j] += coefficient
                    for k in range(m):
                        v[k] -= coefficient * qi[k]
        else:
            coefficients = [dot(q_columns[i], v) for i in range(j)]
            for i in range(j):
                r[i][j] = coefficients[i]
                coefficient = coefficients[i]
                qi = q_columns[i]
                for k in range(m):
                    v[k] -= coefficient * qi[k]

        column_norm = norm2(v)
        r[j][j] = column_norm
        if column_norm == 0.0:
            raise RankDeficientError("column is linearly dependent")
        q_columns.append([value / column_norm for value in v])

    q = [[q_columns[j][i] for j in range(n)] for i in range(m)]
    method: Method = "modified-gram-schmidt" if modified else "classical-gram-schmidt"
    return QR(Q=q, R=r, method=method)
