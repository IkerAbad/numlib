"""Linear least squares by Householder QR and by the normal equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike

from numlib._convert import as_matrix, as_vector
from numlib._ops import dot, matmul, matvec, norm2, transpose
from numlib._triangular import solve_upper
from numlib._types import Vector
from numlib.analysis import backward_error
from numlib.lu import solve
from numlib.qr import qr_householder

Method = Literal["qr", "normal-equations"]


@dataclass(frozen=True)
class LeastSquares:
    """Least-squares solution and diagnostics.

    ``backward_error`` refers to the square system that the method actually
    solves: ``R x = Q^T b`` for QR and ``A^T A x = A^T b`` for the normal
    equations.
    """

    x: Vector
    residual_norm: float
    backward_error: float
    method: Method


def lstsq_qr(a: ArrayLike, b: ArrayLike) -> LeastSquares:
    """Solve ``min ||A x - b||_2`` with a Householder QR factorization."""
    matrix = as_matrix(a)
    rhs = as_vector(b)
    m = len(matrix)
    n = len(matrix[0])
    if m < n:
        raise ValueError("least squares requires at least as many rows as columns")
    if len(rhs) != m:
        raise ValueError("right-hand side has the wrong length")

    qr = qr_householder(matrix)
    projected = [dot([qr.Q[i][j] for i in range(m)], rhs) for j in range(n)]
    x = solve_upper(qr.R, projected)
    residual = [rhs[i] - value for i, value in enumerate(matvec(matrix, x))]
    return LeastSquares(
        x=x,
        residual_norm=norm2(residual),
        backward_error=backward_error(qr.R, x, projected),
        method="qr",
    )


def lstsq_normal(a: ArrayLike, b: ArrayLike) -> LeastSquares:
    """Solve ``min ||A x - b||_2`` through the normal equations."""
    matrix = as_matrix(a)
    rhs = as_vector(b)
    m = len(matrix)
    n = len(matrix[0])
    if m < n:
        raise ValueError("least squares requires at least as many rows as columns")
    if len(rhs) != m:
        raise ValueError("right-hand side has the wrong length")

    transpose_a = transpose(matrix)
    gram = matmul(transpose_a, matrix)
    right_hand_side = matvec(transpose_a, rhs)
    x = solve(gram, right_hand_side)
    residual = [rhs[i] - value for i, value in enumerate(matvec(matrix, x))]
    return LeastSquares(
        x=x,
        residual_norm=norm2(residual),
        backward_error=backward_error(gram, x, right_hand_side),
        method="normal-equations",
    )
