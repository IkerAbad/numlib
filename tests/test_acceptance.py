import numpy as np
import pytest

from numlib import (
    backward_error,
    eigen_qr,
    forward_error,
    lstsq_qr,
    lu,
    qr_householder,
    random_gaussian,
    svd,
    with_condition,
)
from numlib._ops import matvec

EPS = float(np.finfo(np.float64).eps)


def test_lu_is_within_two_orders_of_lapack() -> None:
    for seed in (0, 1, 2):
        matrix = with_condition(20, 20, 10.0, seed=seed)
        expected = [1.0] * 20
        rhs = matvec(matrix, expected)
        ours = lu(matrix).solve(rhs)
        lapack = np.linalg.solve(np.array(matrix), np.array(rhs))
        ours_error = forward_error(ours, expected)
        lapack_error = forward_error(lapack, expected)
        assert ours_error <= 100.0 * max(lapack_error, 1e-15)
        assert ours_error < 1e-12
        assert backward_error(matrix, ours, rhs) < 1e-14


def test_qr_least_squares_is_within_two_orders_of_lapack() -> None:
    for seed in (0, 1, 2):
        matrix = with_condition(30, 15, 10.0, seed=seed + 10)
        expected = [float(index + 1) for index in range(15)]
        rhs = matvec(matrix, expected)
        ours = lstsq_qr(matrix, rhs)
        lapack = np.linalg.lstsq(np.array(matrix), np.array(rhs), rcond=None)[0]
        ours_error = forward_error(ours.x, expected)
        lapack_error = forward_error(lapack, expected)
        assert ours_error <= 100.0 * max(lapack_error, 1e-15)
        assert ours_error < 1e-11
        assert ours.backward_error < 1e-14


def test_svd_is_within_two_orders_of_lapack() -> None:
    for seed in (0, 1):
        matrix = random_gaussian(12, 12, seed=seed + 20)
        array = np.array(matrix)
        factor = svd(matrix)
        u = np.array(factor.U)
        singular = np.array(factor.S)
        v = np.array(factor.V)
        ours_residual = float(
            np.linalg.norm(u @ np.diag(singular) @ v.T - array) / np.linalg.norm(array)
        )
        lapack_u, lapack_s, lapack_vt = np.linalg.svd(array)
        lapack_residual = float(
            np.linalg.norm(lapack_u @ np.diag(lapack_s) @ lapack_vt - array) / np.linalg.norm(array)
        )
        assert ours_residual <= 100.0 * max(lapack_residual, 1e-15)
        difference = np.linalg.norm(singular - lapack_s) / np.linalg.norm(lapack_s)
        assert difference < 1e-13


def test_eigenvalues_are_within_two_orders_of_lapack() -> None:
    rng = np.random.default_rng(30)
    matrix = rng.standard_normal((14, 14))
    matrix = 0.5 * (matrix + matrix.T)
    result = eigen_qr(matrix)
    computed = np.sort(np.array([value.real for value in result.values]))
    reference = np.linalg.eigvalsh(matrix)
    scale = max(float(np.linalg.norm(reference)), 1.0)
    error = float(np.linalg.norm(computed - reference) / scale)
    assert error < 100.0 * EPS
    q = np.array(result.Q)
    t = np.array(result.T)
    residual = np.linalg.norm(q @ t @ q.T - matrix) / np.linalg.norm(matrix)
    assert residual < 1e-13


def test_qr_factorization_backward_error() -> None:
    rng = np.random.default_rng(40)
    matrix = rng.standard_normal((30, 12))
    factor = qr_householder(matrix)
    residual = np.linalg.norm(np.array(factor.Q) @ np.array(factor.R) - matrix) / np.linalg.norm(
        matrix
    )
    assert residual < 1e-14


def test_svd_backward_error() -> None:
    matrix = random_gaussian(15, 15, seed=50)
    factor = svd(matrix)
    residual = np.linalg.norm(
        np.array(factor.U) @ np.diag(np.array(factor.S)) @ np.array(factor.V).T - np.array(matrix)
    ) / np.linalg.norm(matrix)
    assert residual < 1e-14


def test_rejects_nan() -> None:
    with pytest.raises(ValueError):
        lu([[1.0, float("nan")], [0.0, 1.0]])
