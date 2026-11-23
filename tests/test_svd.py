import numpy as np
import pytest

from numlib import loss_of_orthogonality, svd


def _reconstruct(factor: object) -> np.ndarray:
    from numlib.svd import SVD

    assert isinstance(factor, SVD)
    u = np.array(factor.U)
    s = np.array(factor.S)
    v = np.array(factor.V)
    return u @ np.diag(s) @ v.T


@pytest.mark.parametrize("shape", [(1, 1), (5, 3), (4, 4), (3, 5), (1, 4), (6, 2)])
def test_reconstruction_and_orthogonality(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    matrix = rng.standard_normal(shape)
    factor = svd(matrix)
    np.testing.assert_allclose(_reconstruct(factor), matrix, atol=1e-13, rtol=1e-13)
    assert loss_of_orthogonality(factor.U) < 1e-12
    assert loss_of_orthogonality(factor.V) < 1e-12
    singular = np.array(factor.S)
    assert np.all(singular >= 0.0)
    assert np.all(np.diff(singular) <= 0.0)


def test_singular_values_match_numpy() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((9, 5))
    factor = svd(matrix)
    expected = np.linalg.svd(matrix, compute_uv=False)
    np.testing.assert_allclose(factor.S, expected, rtol=1e-12, atol=1e-14)


def test_rank_deficient_matrix() -> None:
    matrix = [[1.0, 2.0], [2.0, 4.0]]
    factor = svd(matrix)
    assert factor.S[-1] < 1e-15
    np.testing.assert_allclose(_reconstruct(factor), np.array(matrix), atol=1e-14)
    assert loss_of_orthogonality(factor.U) < 1e-12
    assert loss_of_orthogonality(factor.V) < 1e-12


def test_zero_matrix() -> None:
    zero = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    factor = svd(zero)
    assert factor.S == [0.0, 0.0]
    np.testing.assert_allclose(_reconstruct(factor), np.zeros((3, 2)))
    assert loss_of_orthogonality(factor.U) < 1e-12


def test_diagonal_matrix() -> None:
    factor = svd([[3.0, 0.0], [0.0, -4.0]])
    np.testing.assert_allclose(factor.S, [4.0, 3.0])
    np.testing.assert_allclose(_reconstruct(factor), [[3.0, 0.0], [0.0, -4.0]], atol=1e-14)
