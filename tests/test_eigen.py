import math

import numpy as np
import pytest

from numlib import eigen_qr, eigenvalues, hessenberg, wilkinson


def _match(computed: list[complex], expected: np.ndarray) -> float:
    remaining = list(computed)
    worst = 0.0
    for value in expected:
        index = min(range(len(remaining)), key=lambda i: abs(remaining[i] - value))
        difference = abs(remaining[index] - value) / max(1.0, abs(value))
        worst = max(worst, difference)
        remaining.pop(index)
    return worst


def test_two_by_two_symmetric() -> None:
    values = eigenvalues([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(sorted(value.real for value in values), [1.0, 3.0])


def test_rotation_has_imaginary_pair() -> None:
    values = sorted(eigenvalues([[0.0, -1.0], [1.0, 0.0]]), key=lambda z: z.imag)
    np.testing.assert_allclose(values[0], -1j, atol=1e-15)
    np.testing.assert_allclose(values[1], 1j, atol=1e-15)


def test_random_symmetric_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((8, 8))
    matrix = 0.5 * (matrix + matrix.T)
    result = eigen_qr(matrix)
    assert _match(result.values, np.linalg.eigvalsh(matrix)) < 1e-12
    q = np.array(result.Q)
    t = np.array(result.T)
    np.testing.assert_allclose(q @ t @ q.T, matrix, atol=1e-12)


def test_random_nonsymmetric_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    for n in (3, 5, 7):
        matrix = rng.standard_normal((n, n))
        result = eigen_qr(matrix)
        assert _match(result.values, np.linalg.eigvals(matrix)) < 1e-10


def test_wilkinson_matrix() -> None:
    matrix = wilkinson(8)
    result = eigenvalues(matrix)
    assert _match(result, np.linalg.eigvalsh(matrix)) < 1e-12


def test_hessenberg_reduction() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((7, 7))
    h, q = hessenberg(matrix)
    np.testing.assert_allclose(np.array(q) @ np.array(h) @ np.array(q).T, matrix, atol=1e-13)
    for i in range(2, len(h)):
        for j in range(i - 1):
            assert h[i][j] == 0.0


def test_complex_pair_with_coupling() -> None:
    matrix = [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, -4.0],
        [0.0, 1.0, 0.0],
    ]
    values = eigenvalues(matrix)
    assert _match(values, np.linalg.eigvals(matrix)) < 1e-12


def test_eigenvalues_are_sorted() -> None:
    values = eigenvalues([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    keys = [(value.real, value.imag) for value in values]
    assert keys == sorted(keys)


def test_non_square_raises() -> None:
    with pytest.raises(ValueError):
        eigenvalues([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_identity_matrix() -> None:
    result = eigenvalues(np.eye(4))
    np.testing.assert_allclose([value.real for value in result], [1.0] * 4)
    assert all(abs(value.imag) < 1e-15 for value in result)


def test_known_cubic() -> None:
    matrix = [[0.0, 0.0, 6.0], [1.0, 0.0, -11.0], [0.0, 1.0, 6.0]]
    expected = np.array([1.0, 2.0, 3.0])
    assert _match(eigenvalues(matrix), expected) < 1e-10
    assert math.isfinite(max(abs(value) for value in eigenvalues(matrix)))
