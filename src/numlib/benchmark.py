"""Benchmarks against NumPy's LAPACK-backed routines.

All timings are wall-clock measurements with ``time.perf_counter``. The NumPy
routines call the LAPACK implementation shipped with the installed BLAS. The
point is not to beat LAPACK (it is compiled and blocked), but to measure how
far a readable pure-Python implementation is from it in accuracy and time.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray

from numlib._convert import to_array
from numlib._ops import matvec
from numlib.analysis import (
    backward_error,
    condition_number,
    forward_error,
    loss_of_orthogonality,
)
from numlib.eigen import eigen_qr
from numlib.exceptions import SingularMatrixError
from numlib.generators import random_gaussian, with_condition
from numlib.lstsq import LeastSquares, lstsq_normal, lstsq_qr
from numlib.lu import lu
from numlib.qr import QR, qr_gram_schmidt, qr_householder
from numlib.svd import svd

CONDITIONS: tuple[float, ...] = (1.0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12)
SOLVE_SIZES: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40)
DECOMPOSITION_SIZES: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16)


@dataclass(frozen=True)
class ErrorRecord:
    """Accuracy and timing of one method on one problem."""

    method: str
    n: int
    condition: float
    forward_error: float
    backward_error: float
    seconds: float


@dataclass(frozen=True)
class OrthogonalityRecord:
    """Loss of orthogonality of a QR factorization."""

    method: str
    n: int
    condition: float
    loss: float
    seconds: float


@dataclass(frozen=True)
class ScalingRecord:
    """Wall-clock scaling of one method with the problem size."""

    method: str
    n: int
    seconds: float


def _time_call(call: Callable[[], object], repeats: int = 1) -> float:
    """Return the best of ``repeats`` wall-clock measurements."""
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best


def _ones(n: int) -> list[float]:
    return [1.0] * n


def _lu_solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    return lu(matrix).solve(rhs)


def _lapack_solve(matrix: NDArray[np.float64], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.linalg.solve(matrix, rhs)


def _lapack_lstsq(matrix: NDArray[np.float64], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _lapack_lstsq_full(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, NDArray[np.float64]]:
    solution, residuals, rank, singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
    return solution, residuals, int(rank), singular_values


def _lapack_svd(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return np.linalg.svd(matrix)


def _lapack_eig(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    return np.linalg.eig(matrix)


def solve_error_vs_condition(
    *,
    n: int = 16,
    conditions: Sequence[float] = CONDITIONS,
    seed: int = 0,
) -> list[ErrorRecord]:
    """Compare LU solve against LAPACK over a range of condition numbers."""
    records: list[ErrorRecord] = []
    for index, target in enumerate(conditions):
        matrix = with_condition(n, n, target, seed=seed + index)
        x_true = _ones(n)
        rhs = matvec(matrix, x_true)
        condition = condition_number(matrix)

        x_ours = _lu_solve(matrix, rhs)
        seconds_ours = _time_call(partial(_lu_solve, matrix, rhs), repeats=3)

        array_a = to_array(matrix)
        array_b = np.array(rhs, dtype=np.float64)
        x_lapack = _lapack_solve(array_a, array_b)
        seconds_lapack = _time_call(partial(_lapack_solve, array_a, array_b), repeats=20)

        records.append(
            ErrorRecord(
                "LU (ours)",
                n,
                condition,
                forward_error(x_ours, x_true),
                backward_error(matrix, x_ours, rhs),
                seconds_ours,
            )
        )
        records.append(
            ErrorRecord(
                "LAPACK solve",
                n,
                condition,
                forward_error(x_lapack, x_true),
                backward_error(matrix, x_lapack, rhs),
                seconds_lapack,
            )
        )
    return records


def lstsq_error_vs_condition(
    *,
    m: int = 24,
    n: int = 12,
    conditions: Sequence[float] = CONDITIONS,
    seed: int = 0,
) -> list[ErrorRecord]:
    """Compare QR and normal-equation least squares over condition numbers."""
    records: list[ErrorRecord] = []
    for index, target in enumerate(conditions):
        matrix = with_condition(m, n, target, seed=seed + 100 + index)
        x_true = _ones(n)
        rhs = matvec(matrix, x_true)
        condition = condition_number(matrix)

        qr_result = lstsq_qr(matrix, rhs)
        qr_cost = _time_call(partial(lstsq_qr, matrix, rhs), repeats=2)
        normal_cost = _time_call(partial(lstsq_normal, matrix, rhs), repeats=2)
        try:
            normal_result: LeastSquares | None = lstsq_normal(matrix, rhs)
        except SingularMatrixError:
            normal_result = None
        normal_error = math.inf if normal_result is None else forward_error(normal_result.x, x_true)

        array_a = to_array(matrix)
        array_b = np.array(rhs, dtype=np.float64)
        x_lapack = _lapack_lstsq(array_a, array_b)
        lapack_cost = _time_call(partial(_lapack_lstsq, array_a, array_b), repeats=20)

        records.append(
            ErrorRecord(
                "QR (ours)",
                n,
                condition,
                forward_error(qr_result.x, x_true),
                qr_result.backward_error,
                qr_cost,
            )
        )
        records.append(
            ErrorRecord(
                "Normal equations",
                n,
                condition,
                normal_error,
                math.nan,
                normal_cost,
            )
        )
        records.append(
            ErrorRecord(
                "LAPACK lstsq",
                n,
                condition,
                forward_error(x_lapack, x_true),
                0.0,
                lapack_cost,
            )
        )
    return records


def orthogonality_vs_condition(
    *,
    n: int = 20,
    conditions: Sequence[float] = CONDITIONS,
    seed: int = 0,
) -> list[OrthogonalityRecord]:
    """Measure ``||Q^T Q - I||`` for four QR strategies."""
    records: list[OrthogonalityRecord] = []
    for index, target in enumerate(conditions):
        matrix = with_condition(n, n, target, seed=seed + 200 + index)
        condition = condition_number(matrix)
        methods: tuple[tuple[str, Callable[[], QR]], ...] = (
            ("Householder", partial(qr_householder, matrix)),
            ("MGS", partial(qr_gram_schmidt, matrix)),
            (
                "MGS twice",
                partial(qr_gram_schmidt, matrix, reorthogonalize=True),
            ),
            (
                "Classical GS",
                partial(qr_gram_schmidt, matrix, modified=False),
            ),
        )
        for method, run in methods:
            factor = run()
            records.append(
                OrthogonalityRecord(
                    method,
                    n,
                    condition,
                    loss_of_orthogonality(factor.Q),
                    _time_call(run, repeats=2),
                )
            )
    return records


def solve_error_vs_size(*, sizes: Sequence[int] = SOLVE_SIZES, seed: int = 0) -> list[ErrorRecord]:
    """Forward error of LU solve for well-conditioned matrices of growing size."""
    records: list[ErrorRecord] = []
    for index, n in enumerate(sizes):
        matrix = with_condition(n, n, 100.0, seed=seed + 300 + index)
        x_true = _ones(n)
        rhs = matvec(matrix, x_true)

        x_ours = _lu_solve(matrix, rhs)
        seconds_ours = _time_call(partial(_lu_solve, matrix, rhs), repeats=3)

        array_a = to_array(matrix)
        array_b = np.array(rhs, dtype=np.float64)
        x_lapack = _lapack_solve(array_a, array_b)
        seconds_lapack = _time_call(partial(_lapack_solve, array_a, array_b), repeats=10)

        records.append(
            ErrorRecord(
                "LU (ours)",
                n,
                100.0,
                forward_error(x_ours, x_true),
                backward_error(matrix, x_ours, rhs),
                seconds_ours,
            )
        )
        records.append(
            ErrorRecord(
                "LAPACK solve",
                n,
                100.0,
                forward_error(x_lapack, x_true),
                0.0,
                seconds_lapack,
            )
        )
    return records


def lstsq_error_vs_size(
    *, sizes: Sequence[int] = (5, 10, 15, 20, 25, 30), seed: int = 0
) -> list[ErrorRecord]:
    """Forward error of QR and normal-equation least squares for growing sizes."""
    records: list[ErrorRecord] = []
    for index, n in enumerate(sizes):
        matrix = with_condition(2 * n, n, 100.0, seed=seed + 400 + index)
        x_true = _ones(n)
        rhs = matvec(matrix, x_true)

        qr_result = lstsq_qr(matrix, rhs)
        qr_cost = _time_call(partial(lstsq_qr, matrix, rhs), repeats=2)
        normal_result = lstsq_normal(matrix, rhs)
        normal_cost = _time_call(partial(lstsq_normal, matrix, rhs), repeats=2)

        array_a = to_array(matrix)
        array_b = np.array(rhs, dtype=np.float64)
        x_lapack = _lapack_lstsq(array_a, array_b)
        lapack_cost = _time_call(partial(_lapack_lstsq, array_a, array_b), repeats=10)

        records.append(
            ErrorRecord(
                "QR (ours)",
                n,
                100.0,
                forward_error(qr_result.x, x_true),
                qr_result.backward_error,
                qr_cost,
            )
        )
        records.append(
            ErrorRecord(
                "Normal equations",
                n,
                100.0,
                forward_error(normal_result.x, x_true),
                normal_result.backward_error,
                normal_cost,
            )
        )
        records.append(
            ErrorRecord(
                "LAPACK lstsq",
                n,
                100.0,
                forward_error(x_lapack, x_true),
                0.0,
                lapack_cost,
            )
        )
    return records


def decomposition_error_vs_size(
    *, sizes: Sequence[int] = DECOMPOSITION_SIZES, seed: int = 0
) -> list[ErrorRecord]:
    """Accuracy of the SVD and of the symmetric QR eigensolver."""
    records: list[ErrorRecord] = []
    for index, n in enumerate(sizes):
        matrix = random_gaussian(n, n, seed=seed + 500 + index)
        array = to_array(matrix)
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            continue

        factor = svd(matrix)
        svd_cost = _time_call(partial(svd, matrix), repeats=2)
        reference = np.linalg.svd(array, compute_uv=False)
        singular_error = float(
            np.linalg.norm(np.array(factor.S) - reference) / np.linalg.norm(reference)
        )
        u = np.array(factor.U)
        v = np.array(factor.V)
        reconstruction = u @ np.diag(np.array(factor.S)) @ v.T
        singular_residual = float(np.linalg.norm(reconstruction - array) / norm)
        records.append(
            ErrorRecord(
                "SVD (ours)",
                n,
                1.0,
                singular_error,
                singular_residual,
                svd_cost,
            )
        )

        symmetric = [[0.5 * (matrix[i][j] + matrix[j][i]) for j in range(n)] for i in range(n)]
        symmetric_array = to_array(symmetric)
        symmetric_norm = float(np.linalg.norm(symmetric_array))
        result = eigen_qr(symmetric)
        eigen_cost = _time_call(partial(eigen_qr, symmetric), repeats=2)
        reference_values = np.linalg.eigvalsh(symmetric_array)
        computed = np.sort(np.array([value.real for value in result.values]))
        eigen_error = float(
            np.linalg.norm(computed - reference_values)
            / max(float(np.linalg.norm(reference_values)), 1e-300)
        )
        q = np.array(result.Q)
        t = np.array(result.T)
        eigen_residual = float(np.linalg.norm(q @ t @ q.T - symmetric_array) / symmetric_norm)
        records.append(
            ErrorRecord(
                "QR eigenvalues (ours)",
                n,
                1.0,
                eigen_error,
                eigen_residual,
                eigen_cost,
            )
        )
    return records


def solve_time_vs_size(*, sizes: Sequence[int] = SOLVE_SIZES, seed: int = 0) -> list[ScalingRecord]:
    """Wall-clock time of LU and QR least squares against LAPACK."""
    records: list[ScalingRecord] = []
    for index, n in enumerate(sizes):
        matrix = random_gaussian(n, n, seed=seed + 600 + index)
        rhs = matvec(matrix, _ones(n))
        records.append(
            ScalingRecord(
                "LU solve (ours)",
                n,
                _time_call(partial(_lu_solve, matrix, rhs), repeats=2),
            )
        )
        array_a = to_array(matrix)
        array_b = np.array(rhs, dtype=np.float64)
        records.append(
            ScalingRecord(
                "LAPACK solve",
                n,
                _time_call(partial(_lapack_solve, array_a, array_b), repeats=10),
            )
        )

        tall = random_gaussian(2 * n, n, seed=seed + 700 + index)
        tall_rhs = matvec(tall, _ones(n))
        records.append(
            ScalingRecord(
                "QR least squares (ours)",
                n,
                _time_call(partial(lstsq_qr, tall, tall_rhs), repeats=2),
            )
        )
        tall_array = to_array(tall)
        tall_b = np.array(tall_rhs, dtype=np.float64)
        records.append(
            ScalingRecord(
                "LAPACK lstsq",
                n,
                _time_call(
                    partial(_lapack_lstsq_full, tall_array, tall_b),
                    repeats=10,
                ),
            )
        )
    return records


def decomposition_time_vs_size(
    *, sizes: Sequence[int] = DECOMPOSITION_SIZES, seed: int = 0
) -> list[ScalingRecord]:
    """Wall-clock time of SVD and symmetric eigenvalues against LAPACK."""
    records: list[ScalingRecord] = []
    for index, n in enumerate(sizes):
        matrix = random_gaussian(n, n, seed=seed + 800 + index)
        array = to_array(matrix)
        records.append(
            ScalingRecord(
                "SVD (ours)",
                n,
                _time_call(partial(svd, matrix), repeats=2),
            )
        )
        records.append(
            ScalingRecord(
                "LAPACK SVD",
                n,
                _time_call(partial(_lapack_svd, array), repeats=10),
            )
        )

        symmetric = [[0.5 * (matrix[i][j] + matrix[j][i]) for j in range(n)] for i in range(n)]
        symmetric_array = to_array(symmetric)
        records.append(
            ScalingRecord(
                "QR eigenvalues (ours)",
                n,
                _time_call(partial(eigen_qr, symmetric), repeats=2),
            )
        )
        records.append(
            ScalingRecord(
                "LAPACK eig",
                n,
                _time_call(partial(_lapack_eig, symmetric_array), repeats=10),
            )
        )
    return records
