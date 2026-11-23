import numpy as np
import pytest

from numlib import eigen_qr, eigenvalues


def _match_eigenvalues(computed: list[complex], expected: np.ndarray) -> list[int]:
    remaining = list(expected)
    indices: list[int] = []
    for value in computed:
        index = min(range(len(remaining)), key=lambda i: abs(remaining[i] - value))
        indices.append(index)
        remaining.pop(index)
    return indices


def _relative_eigenvalue_error(computed: list[complex], expected: np.ndarray) -> float:
    worst = 0.0
    for value in computed:
        distance = min(abs(value - candidate) for candidate in expected)
        worst = max(worst, distance / max(1.0, abs(value)))
    return worst


def _phase_error(ours: np.ndarray, reference: np.ndarray) -> float:
    ours = ours / np.linalg.norm(ours)
    reference = reference / np.linalg.norm(reference)
    phase = np.vdot(reference, ours)
    if abs(phase) == 0.0:
        return float(np.linalg.norm(ours - reference))
    phase = phase / abs(phase)
    return float(np.linalg.norm(ours - phase * reference))


def _residual(matrix: np.ndarray, values: list[complex], vectors: np.ndarray) -> float:
    return float(
        np.linalg.norm(matrix @ vectors - vectors @ np.diag(values))
        / max(np.linalg.norm(matrix), 1.0)
    )


def test_diagonal_matrix_has_coordinate_eigenvectors() -> None:
    matrix = np.diag([1.0, 2.0, 3.0])
    result = eigen_qr(matrix)
    vectors = np.array(result.vectors)
    for index, value in enumerate(result.values):
        column = vectors[:, index]
        pivot = int(np.argmax(np.abs(column)))
        assert value.real == pytest.approx(float(pivot + 1))
        np.testing.assert_allclose(column[pivot], 1.0, atol=1e-14)
        assert np.count_nonzero(np.abs(column) > 1e-12) == 1


def test_rotation_has_complex_conjugate_eigenvectors() -> None:
    matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = eigen_qr(matrix)
    np.testing.assert_allclose(
        sorted(value.imag for value in result.values), [-1.0, 1.0], atol=1e-14
    )
    vectors = np.array(result.vectors)
    assert np.linalg.norm(vectors.conj().T @ vectors - np.eye(2)) < 1e-12
    assert _residual(matrix, result.values, vectors) < 1e-12


def test_random_general_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((6, 6))
    result = eigen_qr(matrix)
    reference_values, reference_vectors = np.linalg.eig(matrix)
    assert _relative_eigenvalue_error(result.values, reference_values) < 1e-10
    vectors = np.array(result.vectors)
    assert _residual(matrix, result.values, vectors) < 1e-11

    used: set[int] = set()
    for index, value in enumerate(result.values):
        candidate = int(np.argmin(np.abs(reference_values - value)))
        assert candidate not in used
        used.add(candidate)
        error = _phase_error(vectors[:, index], reference_vectors[:, candidate])
        assert error < 1e-9


def test_random_matrices_satisfy_eigenvector_equation() -> None:
    rng = np.random.default_rng(1)
    for n in (3, 5, 8):
        matrix = rng.standard_normal((n, n))
        result = eigen_qr(matrix)
        vectors = np.array(result.vectors)
        assert vectors.shape == (n, n)
        assert _residual(matrix, result.values, vectors) < 1e-10
        assert max(result.residuals) < 1e-10


def test_symmetric_matrix_has_orthonormal_eigenvectors() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((7, 7))
    matrix = 0.5 * (matrix + matrix.T)
    result = eigen_qr(matrix)
    vectors = np.array(result.vectors)
    reference_values, reference_vectors = np.linalg.eigh(matrix)
    assert _relative_eigenvalue_error(result.values, reference_values) < 1e-12
    assert np.linalg.norm(vectors.conj().T @ vectors - np.eye(7)) < 1e-10
    assert _residual(matrix, result.values, vectors) < 1e-10

    for index in range(7):
        overlap = abs(np.vdot(reference_vectors[:, index], vectors[:, index]))
        assert overlap == pytest.approx(1.0, abs=1e-9)


def test_repeated_eigenvalues_have_full_eigenbasis() -> None:
    for matrix in (
        np.eye(5),
        np.diag([2.0, 2.0, 2.0, 3.0, 3.0]),
    ):
        result = eigen_qr(matrix)
        vectors = np.array(result.vectors)
        assert np.linalg.matrix_rank(vectors) == len(matrix)
        assert np.linalg.norm(vectors.conj().T @ vectors - np.eye(len(matrix))) < 1e-12
        assert _residual(matrix, result.values, vectors) < 1e-12


def test_defective_jordan_block_is_reported() -> None:
    matrix = np.array([[1.0, 1.0], [0.0, 1.0]])
    result = eigen_qr(matrix)
    vectors = np.array(result.vectors)
    values = np.array(result.values)
    assert _residual(matrix, result.values, vectors) < 1e-12
    assert np.linalg.matrix_rank(vectors, tol=1e-8) == 1
    assert np.linalg.cond(vectors) > 1e10
    assert len(values) == 2
    assert max(result.residuals) < 1e-12


def test_defective_three_by_three_is_reported() -> None:
    matrix = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]])
    result = eigen_qr(matrix)
    vectors = np.array(result.vectors)
    assert _residual(matrix, result.values, vectors) < 1e-12
    assert np.linalg.matrix_rank(vectors, tol=1e-8) == 1


def test_schur_form_and_householder_accumulation() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((7, 7))
    result = eigen_qr(matrix)
    q = np.array(result.Q)
    t = np.array(result.T)
    np.testing.assert_allclose(q @ t @ q.T, matrix, atol=1e-12)
    vectors = np.array(result.vectors)
    assert _residual(matrix, result.values, vectors) < 1e-10


def test_eigenvalues_only_skips_vectors() -> None:
    matrix = [[2.0, 1.0], [1.0, 2.0]]
    result = eigen_qr(matrix, compute_vectors=False)
    assert result.vectors == []
    assert result.residuals == []
    np.testing.assert_allclose(sorted(value.real for value in result.values), [1.0, 3.0])
    assert len(eigenvalues(matrix)) == 2


def test_symmetric_asymmetric_input_raises() -> None:
    with pytest.raises(ValueError):
        eigen_qr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
