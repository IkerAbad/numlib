import math

import numpy as np
import pytest

from numlib import (
    backward_error,
    condition_number,
    forward_error,
    hilbert,
    loss_of_orthogonality,
    lu,
    qr_householder,
)


def test_condition_number_identity() -> None:
    assert condition_number(np.eye(5)) == pytest.approx(1.0)


def test_condition_number_diagonal() -> None:
    assert condition_number([[1.0, 0.0], [0.0, 1e6]]) == pytest.approx(1e6, rel=1e-12)


def test_condition_number_hilbert_matches_numpy() -> None:
    matrix = hilbert(8)
    assert condition_number(matrix) == pytest.approx(float(np.linalg.cond(matrix)), rel=1e-6)


def test_condition_number_singular_is_infinite() -> None:
    assert math.isinf(condition_number([[1.0, 2.0], [2.0, 4.0]]))


def test_backward_error_of_exact_solution_is_zero() -> None:
    matrix = [[2.0, 0.0], [0.0, 4.0]]
    rhs = [2.0, 8.0]
    assert backward_error(matrix, [1.0, 2.0], rhs) == 0.0


def test_backward_error_of_lu_solution_is_small() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((10, 10)) + 10 * np.eye(10)
    rhs = rng.standard_normal(10)
    solution = lu(matrix).solve(rhs.tolist())
    assert backward_error(matrix, solution, rhs) < 1e-14


def test_forward_error() -> None:
    assert forward_error([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert forward_error([1.5, 2.0], [1.0, 2.0]) == pytest.approx(0.25)
    assert math.isinf(forward_error([1.0], [0.0]))


def test_loss_of_orthogonality_identity() -> None:
    assert loss_of_orthogonality(np.eye(6)) == pytest.approx(0.0, abs=1e-15)


def test_loss_of_orthogonality_householder_is_small() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((12, 5))
    factor = qr_householder(matrix)
    assert loss_of_orthogonality(factor.Q) < 1e-12
