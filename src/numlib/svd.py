"""Singular value decomposition by one-sided Jacobi rotations.

The columns of the working matrix are rotated until they are mutually
orthogonal. The singular values are the resulting column norms. One-sided
Jacobi is slower than the bidiagonalization used by LAPACK, but it has high
relative accuracy for tiny singular values and needs no QR iteration.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from numpy.typing import ArrayLike

from numlib._convert import as_matrix
from numlib._ops import identity, norm2, transpose, zeros
from numlib._types import Matrix, Vector


@dataclass(frozen=True)
class SVD:
    """Thin SVD ``A = U diag(S) V^T``.

    With ``k = min(m, n)``, ``U`` is ``m x k``, ``S`` has length ``k`` and
    ``V`` is ``n x k``.
    """

    U: Matrix
    S: Vector
    V: Matrix


def svd(a: ArrayLike, *, tol: float = 1e-15, max_sweeps: int = 60) -> SVD:
    """Compute the thin SVD of ``a`` with one-sided Jacobi rotations."""
    matrix = as_matrix(a)
    m = len(matrix)
    n = len(matrix[0])
    if m < n:
        factor = _svd_tall(transpose(matrix), tol, max_sweeps)
        # A = (A^T)^T = (U S V^T)^T = V S U^T
        return SVD(U=factor.V, S=factor.S, V=factor.U)
    return _svd_tall(matrix, tol, max_sweeps)


def _svd_tall(matrix: Matrix, tol: float, max_sweeps: int) -> SVD:
    m = len(matrix)
    n = len(matrix[0])
    b = [row[:] for row in matrix]
    v = identity(n)

    for _ in range(max_sweeps):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = 0.0
                beta = 0.0
                gamma = 0.0
                for i in range(m):
                    bp = b[i][p]
                    bq = b[i][q]
                    alpha += bp * bp
                    beta += bq * bq
                    gamma += bp * bq
                if alpha == 0.0 or beta == 0.0:
                    continue
                if abs(gamma) <= tol * math.sqrt(alpha * beta):
                    continue
                changed = True

                zeta = (beta - alpha) / (2.0 * gamma)
                if zeta == 0.0:
                    tangent = 1.0
                else:
                    tangent = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = cosine * tangent

                for i in range(m):
                    bp = b[i][p]
                    bq = b[i][q]
                    b[i][p] = cosine * bp - sine * bq
                    b[i][q] = sine * bp + cosine * bq
                for i in range(n):
                    vp = v[i][p]
                    vq = v[i][q]
                    v[i][p] = cosine * vp - sine * vq
                    v[i][q] = sine * vp + cosine * vq

        if not changed:
            break

    sigma = [norm2([b[i][j] for i in range(m)]) for j in range(n)]
    order = sorted(range(n), key=lambda j: sigma[j], reverse=True)
    sigma = [sigma[j] for j in order]
    b = [[b[i][j] for j in order] for i in range(m)]
    v = [[v[i][j] for j in order] for i in range(n)]

    # Singular values below this level carry no reliable direction in double
    # precision. Treating them as zero lets us complete U to an orthonormal
    # basis without changing the reconstruction at working accuracy.
    threshold = sigma[0] * max(m, n) * sys.float_info.epsilon
    for j in range(n):
        if sigma[j] <= threshold:
            sigma[j] = 0.0

    u = zeros(m, n)
    for j in range(n):
        if sigma[j] > 0.0:
            inverse = 1.0 / sigma[j]
            for i in range(m):
                u[i][j] = b[i][j] * inverse

    _complete_orthonormal_basis(u, sigma, m, n)
    return SVD(U=u, S=sigma, V=v)


def _complete_orthonormal_basis(u: Matrix, sigma: Vector, m: int, n: int) -> None:
    """Fill columns with zero singular values with orthonormal vectors."""
    start = next((j for j, value in enumerate(sigma) if value == 0.0), n)
    if start == n:
        return

    for j in range(start, n):
        best: list[float] | None = None
        best_norm = 0.0
        for basis_index in range(m):
            candidate = [1.0 if i == basis_index else 0.0 for i in range(m)]
            for _ in range(2):
                for column in range(j):
                    coefficient = sum(u[i][column] * candidate[i] for i in range(m))
                    for i in range(m):
                        candidate[i] -= coefficient * u[i][column]
            candidate_norm = norm2(candidate)
            if candidate_norm > best_norm:
                best_norm = candidate_norm
                best = candidate
        if best is None or best_norm <= 1e-12:
            continue
        inverse = 1.0 / best_norm
        for i in range(m):
            u[i][j] = best[i] * inverse
