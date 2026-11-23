"""LU decomposition by Gaussian elimination.

Two pivoting strategies are provided:

* partial pivoting selects the largest entry in the current column, which is
  enough for backward stability in practice;
* complete (full) pivoting searches the whole remaining submatrix and also
  permutes columns, which keeps the growth factor small at the cost of an
  O(n^2) search per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike

from numlib._convert import as_square, as_vector
from numlib._ops import (
    column_permutation_matrix,
    matvec,
    permutation_matrix,
    zeros,
)
from numlib._triangular import solve_lower_unit, solve_upper
from numlib._types import Matrix, Vector
from numlib.exceptions import SingularMatrixError

Pivoting = Literal["partial", "full"]


@dataclass(frozen=True)
class LU:
    """Result of :func:`lu`: ``P A Q = L U``.

    ``Q`` is ``None`` when only row pivoting is used, in which case the
    relation is ``P A = L U``.
    """

    L: Matrix
    U: Matrix
    P: Matrix
    Q: Matrix | None
    growth_factor: float

    def solve(self, b: Vector) -> Vector:
        """Solve ``A x = b`` using the stored factorization."""
        n = len(self.U)
        if len(b) != n:
            raise ValueError("right-hand side has the wrong length")
        permuted = matvec(self.P, b)
        z = solve_lower_unit(self.L, permuted)
        w = solve_upper(self.U, z)
        return w if self.Q is None else matvec(self.Q, w)


def lu(a: ArrayLike, *, pivoting: Pivoting = "partial") -> LU:
    """Factor ``A`` as ``P A Q = L U`` (``P A = L U`` with partial pivoting)."""
    matrix = as_square(a)
    n = len(matrix)
    if pivoting not in ("partial", "full"):
        raise ValueError(f"unknown pivoting strategy: {pivoting!r}")

    u = [row[:] for row in matrix]
    lower = zeros(n, n)
    row_perm = list(range(n))
    col_perm = list(range(n))
    max_abs_a = max(abs(value) for row in matrix for value in row)

    for k in range(n):
        if pivoting == "partial":
            p = max(range(k, n), key=lambda i: abs(u[i][k]))
            q = k
        else:
            p, q = max(
                ((i, j) for i in range(k, n) for j in range(k, n)),
                key=lambda pair: abs(u[pair[0]][pair[1]]),
            )
        if u[p][q] == 0.0:
            raise SingularMatrixError("matrix is singular to working precision")

        if q != k:
            for i in range(n):
                u[i][k], u[i][q] = u[i][q], u[i][k]
            col_perm[k], col_perm[q] = col_perm[q], col_perm[k]
        if p != k:
            u[k], u[p] = u[p], u[k]
            lower[k], lower[p] = lower[p], lower[k]
            row_perm[k], row_perm[p] = row_perm[p], row_perm[k]

        pivot = u[k][k]
        for i in range(k + 1, n):
            factor = u[i][k] / pivot
            lower[i][k] = factor
            if factor != 0.0:
                row_i = u[i]
                row_k = u[k]
                for j in range(k + 1, n):
                    row_i[j] -= factor * row_k[j]
            u[i][k] = 0.0

    for i in range(n):
        lower[i][i] = 1.0

    max_abs_u = max(abs(value) for row in u for value in row)
    growth = max_abs_u / max_abs_a if max_abs_a != 0.0 else 0.0
    return LU(
        L=lower,
        U=u,
        P=permutation_matrix(row_perm),
        Q=column_permutation_matrix(col_perm) if pivoting == "full" else None,
        growth_factor=growth,
    )


def solve(a: ArrayLike, b: ArrayLike, *, pivoting: Pivoting = "partial") -> Vector:
    """Solve the square system ``A x = b`` by Gaussian elimination."""
    matrix = as_square(a)
    rhs = as_vector(b)
    if len(rhs) != len(matrix):
        raise ValueError("right-hand side has the wrong length")
    return lu(matrix, pivoting=pivoting).solve(rhs)
