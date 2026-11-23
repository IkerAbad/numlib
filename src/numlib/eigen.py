"""Eigenvalues and eigenvectors of real matrices by the QR algorithm.

The matrix is first reduced to upper Hessenberg form by Householder
similarity transformations. The active trailing block is then iterated with
the Wilkinson double shift (the two eigenvalues of its trailing 2x2 block),
implemented implicitly with 3x3 Householder reflectors that chase the bulge
down the subdiagonal. A 1x1 block yields one real eigenvalue; a 2x2 block is
solved analytically, which also produces complex conjugate pairs.

The orthogonal matrix ``Q`` accumulates every Householder transformation used
by the reduction and by the QR steps. Eigenvectors of the real Schur form
``T`` are obtained by inverse iteration: the shifted system
``(T - lambda I) y = v`` is solved by block back substitution on the
quasi-triangular form, with a tiny diagonal regularization when the shift
makes a block exactly singular. The eigenvector of the original matrix is
``x = Q y``. The cost is O(n^3) for the QR iteration plus O(n^3) for the
inverse iterations when the quasi-triangular solves are used, and O(n^2)
storage.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from numpy.typing import ArrayLike

from numlib._convert import as_square
from numlib._ops import dot, identity, matmul, norm2
from numlib._types import ComplexMatrix, Matrix
from numlib.exceptions import ConvergenceError
from numlib.qr import qr_householder


@dataclass(frozen=True)
class Eigen:
    """Schur decomposition and eigenvectors of a real matrix.

    ``A = Q T Q^T`` with ``T`` quasi upper triangular. Column ``i`` of
    ``vectors`` satisfies ``A v_i = values_i v_i``. ``residuals`` contains the
    relative Schur residual of each eigenvector.

    For a defective eigenvalue the algebraic multiplicity exceeds the
    geometric one. In that case the returned columns cannot form a basis; the
    residuals still measure how well each column satisfies the eigenvector
    equation.
    """

    values: list[complex]
    vectors: ComplexMatrix
    residuals: list[float]
    Q: Matrix
    T: Matrix


def eigenvalues(a: ArrayLike, *, tol: float = 1e-14, max_iterations: int = 1000) -> list[complex]:
    """Return the eigenvalues of a real square matrix."""
    return eigen_qr(
        a,
        tol=tol,
        max_iterations=max_iterations,
        compute_vectors=False,
    ).values


def eigenvectors(
    a: ArrayLike,
    *,
    tol: float = 1e-14,
    max_iterations: int = 1000,
    vector_iterations: int = 20,
    vector_tol: float = 1e-12,
) -> tuple[list[complex], ComplexMatrix]:
    """Return the eigenvalues and right eigenvectors of a real square matrix."""
    result = eigen_qr(
        a,
        tol=tol,
        max_iterations=max_iterations,
        vector_iterations=vector_iterations,
        vector_tol=vector_tol,
    )
    return result.values, result.vectors


def eigen_qr(
    a: ArrayLike,
    *,
    tol: float = 1e-14,
    max_iterations: int = 1000,
    compute_vectors: bool = True,
    vector_iterations: int = 20,
    vector_tol: float = 1e-12,
) -> Eigen:
    """Compute eigenvalues, Schur vectors and optionally eigenvectors."""
    h, q = hessenberg(a)
    n = len(h)
    blocks: list[tuple[complex, int]] = []
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
            blocks.append((complex(h[hi][hi], 0.0), hi))
            hi -= 1
            iterations = 0
            continue

        if lo == hi - 1:
            pair = _eigenvalues_2x2(h[lo][lo], h[lo][hi], h[hi][lo], h[hi][hi])
            blocks.append((pair[0], lo))
            blocks.append((pair[1], lo + 1))
            hi -= 2
            iterations = 0
            continue

        iterations += 1
        if iterations > max_iterations:
            raise ConvergenceError(f"QR iteration did not converge for block {lo}..{hi}")

        a = h[hi - 1][hi - 1]
        b = h[hi - 1][hi]
        c = h[hi][hi - 1]
        d = h[hi][hi]
        discriminant = 0.25 * (a - d) * (a - d) + b * c
        if discriminant >= 0.0:
            _single_shift_step(h, q, lo, hi, _wilkinson_shift(h, hi), n)
        else:
            _double_shift_step(h, q, lo, hi, n)

        # Numerical noise below the Hessenberg profile is not meaningful.
        for i in range(1, n):
            for j in range(i - 1):
                h[i][j] = 0.0

    blocks.sort(key=lambda item: (item[0].real, item[0].imag, item[1]))
    values = [item[0] for item in blocks]

    if compute_vectors:
        vectors, residuals = _eigenvectors_from_schur(
            h,
            q,
            blocks,
            max_iterations=vector_iterations,
            tol=vector_tol,
        )
    else:
        vectors = []
        residuals = []

    return Eigen(
        values=values,
        vectors=vectors,
        residuals=residuals,
        Q=q,
        T=h,
    )


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


def _single_shift_step(h: Matrix, q: Matrix, lo: int, hi: int, mu: float, n: int) -> None:
    """Explicit single-shift QR step with the Wilkinson shift ``mu``."""
    size = hi - lo + 1
    block = [
        [h[lo + i][lo + j] - (mu if i == j else 0.0) for j in range(size)] for i in range(size)
    ]
    factor = qr_householder(block)
    product = matmul(factor.R, factor.Q)

    for i in range(size):
        for j in range(size):
            h[lo + i][lo + j] = product[i][j] + (mu if i == j else 0.0)

    for j in range(hi + 1, n):
        column = [h[lo + k][j] for k in range(size)]
        for i in range(size):
            h[lo + i][j] = sum(factor.Q[k][i] * column[k] for k in range(size))

    for i in range(lo):
        row = [h[i][lo + k] for k in range(size)]
        for j in range(size):
            h[i][lo + j] = sum(row[k] * factor.Q[k][j] for k in range(size))

    for i in range(n):
        row = [q[i][lo + k] for k in range(size)]
        for j in range(size):
            q[i][lo + j] = sum(row[k] * factor.Q[k][j] for k in range(size))


def _wilkinson_shift(h: Matrix, hi: int) -> float:
    """Return the eigenvalue of the trailing 2x2 block closest to ``h[hi][hi]``."""
    a = h[hi - 1][hi - 1]
    b = h[hi - 1][hi]
    c = h[hi][hi - 1]
    d = h[hi][hi]
    delta = 0.5 * (a - d)
    discriminant = delta * delta + b * c
    root = math.sqrt(max(discriminant, 0.0))
    denominator = root if delta == 0.0 else delta + math.copysign(root, delta)
    if denominator == 0.0:
        return d
    return d - b * c / denominator


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


def _eigenvectors_from_schur(
    t: Matrix,
    q: Matrix,
    blocks: list[tuple[complex, int]],
    *,
    max_iterations: int,
    tol: float,
) -> tuple[ComplexMatrix, list[float]]:
    """Compute eigenvectors of ``T`` and map them back with ``Q``."""
    n = len(t)
    columns: ComplexMatrix = []
    residuals: list[float] = []
    for value, start in blocks:
        vector, residual = _inverse_iteration(
            t,
            value,
            start,
            max_iterations=max_iterations,
            tol=tol,
        )
        mapped = _matvec_real_complex(q, vector)
        columns.append(_normalize_complex(mapped))
        residuals.append(residual)
    vectors = [[columns[j][i] for j in range(n)] for i in range(n)]
    return vectors, residuals


def _inverse_iteration(
    t: Matrix,
    value: complex,
    start: int,
    *,
    max_iterations: int,
    tol: float,
) -> tuple[list[complex], float]:
    """Run inverse iteration on ``T - value I`` starting from ``e_start``."""
    n = len(t)
    matrix_norm = _frobenius(t)
    delta = 4.0 * sys.float_info.epsilon * max(1.0, matrix_norm)
    vector = [0j] * n
    vector[start] = 1.0

    best = vector
    best_residual = _relative_residual(t, value, vector, matrix_norm)
    if best_residual <= tol:
        return _normalize_complex(best), best_residual

    for _ in range(max_iterations):
        solved = _solve_quasi_triangular(t, value, vector, delta)
        solved_norm = _norm2_complex(solved)
        if solved_norm == 0.0:
            break
        vector = [entry / solved_norm for entry in solved]
        residual = _relative_residual(t, value, vector, matrix_norm)
        if residual < best_residual:
            best = vector
            best_residual = residual
        if residual <= tol:
            break

    return _normalize_complex(best), best_residual


def _solve_quasi_triangular(
    t: Matrix,
    value: complex,
    rhs: list[complex],
    delta: float,
) -> list[complex]:
    """Solve ``(T - value I) y = rhs`` by block back substitution."""
    n = len(t)
    y = [0j] * n
    blocks = _schur_blocks(t)
    for start, end in reversed(blocks):
        if start == end:
            acc = rhs[start]
            for j in range(start + 1, n):
                acc -= complex(t[start][j]) * y[j]
            denominator = complex(t[start][start], 0.0) - value
            if abs(denominator) < delta:
                denominator = complex(delta, 0.0)
            y[start] = acc / denominator
        else:
            first, second = start, end
            right_0 = rhs[first]
            right_1 = rhs[second]
            for j in range(second + 1, n):
                right_0 -= complex(t[first][j]) * y[j]
                right_1 -= complex(t[second][j]) * y[j]

            a = complex(t[first][first], 0.0) - value
            b = complex(t[first][second], 0.0)
            c = complex(t[second][first], 0.0)
            d = complex(t[second][second], 0.0) - value
            determinant = a * d - b * c
            scale = abs(a) + abs(b) + abs(c) + abs(d)
            if abs(determinant) <= delta * max(1.0, scale):
                a += delta
                d += delta
                determinant = a * d - b * c
                if determinant == 0.0:
                    determinant = complex(delta, 0.0)
            y[first] = (right_0 * d - b * right_1) / determinant
            y[second] = (a * right_1 - c * right_0) / determinant
    return y


def _schur_blocks(t: Matrix) -> list[tuple[int, int]]:
    """Return the 1x1 and 2x2 diagonal blocks of the quasi-triangular ``T``."""
    n = len(t)
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i + 1 < n and t[i + 1][i] != 0.0:
            blocks.append((i, i + 1))
            i += 2
        else:
            blocks.append((i, i))
            i += 1
    return blocks


def _matvec_real_complex(a: Matrix, x: list[complex]) -> list[complex]:
    """Multiply a real matrix by a complex vector."""
    n = len(a)
    return [sum(a[i][j] * x[j] for j in range(n)) for i in range(n)]


def _normalize_complex(vector: list[complex]) -> list[complex]:
    """Normalize in the 2-norm and rotate the largest component to real."""
    norm = _norm2_complex(vector)
    if norm == 0.0:
        raise ConvergenceError("cannot normalize a zero eigenvector")
    normalized = [entry / norm for entry in vector]
    pivot = max(range(len(normalized)), key=lambda i: abs(normalized[i]))
    phase = normalized[pivot] / abs(normalized[pivot])
    if phase == 0.0:
        return normalized
    return [entry / phase for entry in normalized]


def _norm2_complex(vector: list[complex]) -> float:
    return math.hypot(*(abs(entry) for entry in vector))


def _frobenius(matrix: Matrix) -> float:
    return math.hypot(*(value for row in matrix for value in row))


def _relative_residual(
    t: Matrix,
    value: complex,
    vector: list[complex],
    matrix_norm: float,
) -> float:
    n = len(t)
    residual = [sum(t[i][j] * vector[j] for j in range(n)) - value * vector[i] for i in range(n)]
    scale = matrix_norm + abs(value)
    if scale == 0.0:
        return 0.0
    return _norm2_complex(residual) / scale
