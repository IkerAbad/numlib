import numpy as np
import pytest

from numlib import SingularMatrixError, lu, solve
from numlib._ops import matmul


def test_partial_pivoting_factorization() -> None:
    matrix = [[4.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 4.0]]
    factor = lu(matrix)
    assert factor.Q is None
    np.testing.assert_allclose(matmul(factor.P, matrix), matmul(factor.L, factor.U), atol=1e-14)


def test_full_pivoting_factorization() -> None:
    matrix = [[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [4.0, 1.0, 0.0]]
    factor = lu(matrix, pivoting="full")
    assert factor.Q is not None
    left = matmul(matmul(factor.P, matrix), factor.Q)
    np.testing.assert_allclose(left, matmul(factor.L, factor.U), atol=1e-14)
    assert factor.growth_factor >= 1.0


def test_factor_shape_and_triangularity() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((6, 6))
    factor = lu(matrix)
    n = len(matrix)
    for i in range(n):
        assert factor.L[i][i] == pytest.approx(1.0)
        for j in range(i + 1, n):
            assert factor.L[i][j] == 0.0
            assert factor.U[j][i] == 0.0


@pytest.mark.parametrize("pivoting", ["partial", "full"])
def test_solve_known_solution(pivoting: str) -> None:
    matrix = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
    expected = [8.0, -11.0, -3.0]
    rhs = [sum(matrix[i][j] * expected[j] for j in range(3)) for i in range(3)]
    np.testing.assert_allclose(solve(matrix, rhs, pivoting=pivoting), expected, atol=1e-12)


def test_solve_random_systems() -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 10):
        matrix = rng.standard_normal((n, n)) + n * np.eye(n)
        expected = rng.standard_normal(n)
        rhs = matrix @ expected
        computed = solve(matrix, rhs)
        np.testing.assert_allclose(computed, expected, rtol=1e-10, atol=1e-12)


def test_singular_matrix_raises() -> None:
    with pytest.raises(SingularMatrixError):
        lu([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(SingularMatrixError):
        lu([[0.0, 0.0], [0.0, 0.0]])


def test_wrong_shapes_raise() -> None:
    with pytest.raises(ValueError):
        lu([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0, 3.0])


def test_growth_factor_is_at_least_one() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((8, 8))
    assert lu(matrix).growth_factor >= 1.0
