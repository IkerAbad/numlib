import numpy as np
import pytest

from numlib import (
    SingularMatrixError,
    lstsq_normal,
    lstsq_qr,
    with_condition,
)


def test_well_conditioned_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((20, 6))
    expected = rng.standard_normal(6)
    rhs = matrix @ expected
    lapack = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    qr_result = lstsq_qr(matrix, rhs)
    normal_result = lstsq_normal(matrix, rhs)
    np.testing.assert_allclose(qr_result.x, lapack, atol=1e-10)
    np.testing.assert_allclose(normal_result.x, lapack, atol=1e-10)
    assert qr_result.method == "qr"
    assert normal_result.method == "normal-equations"


def test_residual_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((15, 4))
    rhs = rng.standard_normal(15)
    expected_x = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    expected_residual = np.linalg.norm(matrix @ expected_x - rhs)
    qr_result = lstsq_qr(matrix, rhs)
    normal_result = lstsq_normal(matrix, rhs)
    assert qr_result.residual_norm == pytest.approx(expected_residual, rel=1e-8)
    assert normal_result.residual_norm == pytest.approx(expected_residual, rel=1e-8)


def test_normal_equations_lose_accuracy() -> None:
    matrix = with_condition(30, 15, 1e8, seed=2)
    expected = [1.0] * 15
    rhs = [sum(matrix[i][j] * expected[j] for j in range(15)) for i in range(30)]
    qr_error = max(abs(value - 1.0) for value in lstsq_qr(matrix, rhs).x)
    normal_error = max(abs(value - 1.0) for value in lstsq_normal(matrix, rhs).x)
    assert qr_error < 1e-6
    assert normal_error > 10.0 * qr_error


def test_rank_deficient_normal_equations_raise() -> None:
    matrix = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    rhs = [1.0, 2.0, 3.0]
    with pytest.raises(SingularMatrixError):
        lstsq_normal(matrix, rhs)


def test_wrong_shapes_raise() -> None:
    with pytest.raises(ValueError):
        lstsq_qr([[1.0, 2.0, 3.0]], [1.0])
    with pytest.raises(ValueError):
        lstsq_qr([[1.0], [2.0]], [1.0, 2.0, 3.0])
