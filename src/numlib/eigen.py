"""Eigenvalues of real matrices by the QR algorithm.

The matrix is first reduced to upper Hessenberg form by Householder
similarity transformations. The active trailing block is then iterated with
the Wilkinson double shift (the two eigenvalues of its trailing 2x2 block),
implemented implicitly with 3x3 Householder reflectors that chase the bulge
down the subdiagonal. A 1x1 block yields one real eigenvalue; a 2x2 block is
solved analytically, which also produces complex conjugate pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from numpy.typing import ArrayLike

from numlib._convert import as_square
from numlib._ops import dot, identity, norm2
from numlib._types import Matrix
from numlib.exceptions import ConvergenceError


@dataclass(frozen=True)
class Eigen:
    """Schur-like decomposition ``A = Q T Q^T``.

    ``T`` is quasi upper triangular: entries below the first subdiagonal are
    zero, and 2x2 diagonal blocks carry complex conjugate pairs.
    """

    values: list[complex]
    Q: Matrix
    T: Matrix


def eigenvalues(a: ArrayLike, *, tol: float = 1e-14, max_iterations: int = 1000) -> list[complex]:
    """Return the eigenvalues of a real square matrix."""
    return eigen_qr(a, tol=tol, max_iterations=max_iterations).values


def eigen_qr(a: ArrayLike, *, tol: float = 1e-14, max_iterations: int = 1000) -> Eigen:
    """Compute eigenvalues together with the accumulated Schur vectors."""
    h, q = hessenberg(a)
    n = len(h)
    values: list[complex] = []
    hi = n - 1
    iterations = 0

    while hi >= 0:
        # Deflate negligible subdiagonal entries.
        for i in range(hi, 0, -1):
            scale = abs(h[i][i]) + abs(h[i - 1][i - 1])
            if scale == 0.0:
                scale = max(abs(value) for row in h for value in row)
            if scale == 0.0 or abs(h[i][i - 1]) <= tol * scale:
                h[i][i - 1] = 0.0

        lo = hi
        while lo > 0 and h[lo][lo - 1] != 0.0:
            lo -= 1

        if lo == hi:
            values.append(complex(h[hi][hi], 0.0))
            hi -= 1
            iterations = 0
            continue

        if lo == hi - 1:
            values.extend(_eigenvalues_2x2(h[lo][lo], h[lo][hi], h[hi][lo], h[hi][hi]))
            hi -= 2
            iterations = 0
            continue

        iterations += 1
        if iterations > max_iterations:
            raise ConvergenceError(f"QR iteration did not converge for block {lo}..{hi}")

        _double_shift_step(h, q, lo, hi, n)

        # Numerical noise below the Hessenberg profile is not meaningful.
        for i in range(1, n):
            for j in range(i - 1):
                h[i][j] = 0.0

    values.sort(key=lambda value: (value.real, value.imag))
    return Eigen(values=values, Q=q, T=h)


def hessenberg(a: ArrayLike) -> tuple[Matrix, Matrix]:
    """Reduce ``A`` to upper Hessenberg form: ``A = Q H Q^T``."""
    matrix = as_square(a)
    n = len(matrix)
    h = [row[:] for row in matrix]
    q = identity(n)

    for k in range(n - 2):
        x = [h[i][k] for i in range(k + 1, n)]
        norm_x = norm2(x)
        if norm_x == 0.0:
            continue
        alpha = -norm_x if x[0] >= 0.0 else norm_x
        v = x
        v[0] -= alpha
        v_norm_sq = dot(v, v)
        if v_norm_sq == 0.0:
            continue
        beta = 2.0 / v_norm_sq

        for j in range(k, n):
            s = 0.0
            for i in range(k + 1, n):
                s += v[i - k - 1] * h[i][j]
            for i in range(k + 1, n):
                h[i][j] -= beta * v[i - k - 1] * s

        for i in range(n):
            s = 0.0
            for j in range(k + 1, n):
                s += h[i][j] * v[j - k - 1]
            for j in range(k + 1, n):
                h[i][j] -= beta * s * v[j - k - 1]

        for i in range(n):
            s = 0.0
            for j in range(k + 1, n):
                s += q[i][j] * v[j - k - 1]
            for j in range(k + 1, n):
                q[i][j] -= beta * s * v[j - k - 1]

        h[k + 1][k] = alpha
        for i in range(k + 2, n):
            h[i][k] = 0.0

    return h, q


def _double_shift_step(h: Matrix, q: Matrix, lo: int, hi: int, n: int) -> None:
    """Implicit Francis double-shift step on the block ``lo..hi``."""
    s = h[hi - 1][hi - 1] + h[hi][hi]
    t = h[hi - 1][hi - 1] * h[hi][hi] - h[hi - 1][hi] * h[hi][hi - 1]

    x = h[lo][lo] * h[lo][lo] + h[lo][lo + 1] * h[lo + 1][lo] - s * h[lo][lo] + t
    y = h[lo + 1][lo] * (h[lo][lo] + h[lo + 1][lo + 1] - s)
    z = h[lo + 1][lo] * h[lo + 2][lo + 1]

    for k in range(lo, hi - 1):
        norm = math.hypot(x, y, z)
        if norm != 0.0:
            alpha = -norm if x >= 0.0 else norm
            v = (x - alpha, y, z)
            beta = 2.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            start = lo if k == lo else k - 1
            _apply_reflector(h, q, (k, k + 1, k + 2), start, v, beta, n)
            if k > lo:
                h[k + 1][k - 1] = 0.0
                h[k + 2][k - 1] = 0.0
        if k < hi - 2:
            x, y, z = h[k + 1][k], h[k + 2][k], h[k + 3][k]

    # The last 3x3 reflector leaves a bulge at (hi, hi-2); a 2x2 reflector
    # acting on rows hi-1 and hi removes it and restores Hessenberg form.
    x = h[hi - 1][hi - 2]
    y = h[hi][hi - 2]
    norm = math.hypot(x, y)
    if norm != 0.0:
        alpha = -norm if x >= 0.0 else norm
        last_vector = (x - alpha, y)
        beta = 2.0 / (last_vector[0] * last_vector[0] + last_vector[1] * last_vector[1])
        _apply_reflector(h, q, (hi - 1, hi), hi - 2, last_vector, beta, n)
        h[hi][hi - 2] = 0.0


def _apply_reflector(
    h: Matrix,
    q: Matrix,
    rows: tuple[int, ...],
    col_start: int,
    v: tuple[float, ...],
    beta: float,
    n: int,
) -> None:
    """Apply ``P = I - beta v v^T`` to ``H`` on both sides and to ``Q``."""
    size = len(rows)
    for j in range(col_start, n):
        dot_product = sum(v[i] * h[rows[i]][j] for i in range(size))
        for i in range(size):
            h[rows[i]][j] -= beta * v[i] * dot_product

    end = min(rows[-1] + 1, n - 1)
    for i in range(end + 1):
        dot_product = sum(h[i][rows[j]] * v[j] for j in range(size))
        for j in range(size):
            h[i][rows[j]] -= beta * dot_product * v[j]

    for i in range(n):
        dot_product = sum(q[i][rows[j]] * v[j] for j in range(size))
        for j in range(size):
            q[i][rows[j]] -= beta * dot_product * v[j]


def _eigenvalues_2x2(a: float, b: float, c: float, d: float) -> list[complex]:
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace * trace - 4.0 * determinant
    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        return [
            complex(0.5 * (trace + root), 0.0),
            complex(0.5 * (trace - root), 0.0),
        ]
    root = math.sqrt(-discriminant)
    return [
        complex(0.5 * trace, 0.5 * root),
        complex(0.5 * trace, -0.5 * root),
    ]
