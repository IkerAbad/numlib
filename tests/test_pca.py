import numpy as np
import pytest

from numlib import pca

COVARIANCE = [
    [4.0, 2.4, 1.2],
    [2.4, 3.0, 0.9],
    [1.2, 0.9, 1.0],
]


def test_pca_matches_numpy() -> None:
    result = pca(COVARIANCE)
    reference_values, _ = np.linalg.eigh(np.array(COVARIANCE))
    reference_values = np.sort(reference_values)[::-1]
    np.testing.assert_allclose(result.eigenvalues, reference_values, rtol=1e-12, atol=1e-14)
    reference_ratios = reference_values / reference_values.sum()
    np.testing.assert_allclose(
        result.explained_variance_ratio,
        reference_ratios,
        rtol=1e-12,
        atol=1e-14,
    )
    assert sum(result.explained_variance_ratio) == pytest.approx(1.0)
    assert result.cumulative_variance_ratio[-1] == pytest.approx(1.0)
    assert result.explained_variance_ratio[0] > result.explained_variance_ratio[1]


def test_pca_components_are_orthonormal() -> None:
    result = pca(COVARIANCE)
    components = np.array(result.components)
    assert np.linalg.norm(components.T @ components - np.eye(3)) < 1e-10


def test_pca_reconstructs_the_covariance() -> None:
    result = pca(COVARIANCE)
    components = np.array(result.components)
    values = np.array(result.eigenvalues)
    reconstructed = components @ np.diag(values) @ components.T
    np.testing.assert_allclose(reconstructed, np.array(COVARIANCE), atol=1e-10)


def test_pca_rejects_non_symmetric_matrix() -> None:
    with pytest.raises(ValueError):
        pca([[1.0, 2.0], [0.0, 1.0]])


def test_pca_diagonal_covariance() -> None:
    result = pca(np.diag([5.0, 3.0, 1.0]))
    np.testing.assert_allclose(result.eigenvalues, [5.0, 3.0, 1.0])
    np.testing.assert_allclose(result.explained_variance_ratio, [5.0 / 9.0, 3.0 / 9.0, 1.0 / 9.0])
