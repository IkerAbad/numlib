"""Principal component analysis from a covariance matrix."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from numlib._convert import as_square
from numlib._types import Matrix, Vector
from numlib.eigen import eigen_qr


@dataclass(frozen=True)
class PCA:
    """Principal components of a covariance matrix.

    ``components`` has one column per component, ordered by decreasing
    eigenvalue. ``explained_variance_ratio[i]`` is the fraction of the total
    variance carried by component ``i``.
    """

    eigenvalues: Vector
    components: Matrix
    explained_variance_ratio: Vector
    cumulative_variance_ratio: Vector


def pca(covariance: ArrayLike) -> PCA:
    """Compute the principal components of a symmetric covariance matrix."""
    matrix = as_square(covariance)
    n = len(matrix)
    scale = max(1.0, max(abs(value) for row in matrix for value in row))
    for i in range(n):
        for j in range(i):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-12 * scale:
                raise ValueError("covariance matrix must be symmetric")

    result = eigen_qr(matrix)
    imaginary = max(
        (abs(result.vectors[i][j].imag) for i in range(n) for j in range(n)),
        default=0.0,
    )
    if imaginary > 1e-8 * scale:
        raise ValueError(
            "eigenvectors of the covariance matrix are not real; "
            "the matrix is too ill-conditioned for a real PCA"
        )

    order = sorted(range(n), key=lambda index: result.values[index].real, reverse=True)
    eigenvalues = [result.values[index].real for index in order]
    total = sum(eigenvalues)
    if total <= 0.0:
        raise ValueError("covariance matrix must have positive trace")

    components = [[result.vectors[row][index].real for index in order] for row in range(n)]
    ratios = [value / total for value in eigenvalues]
    cumulative: list[float] = []
    running = 0.0
    for ratio in ratios:
        running += ratio
        cumulative.append(running)

    return PCA(
        eigenvalues=eigenvalues,
        components=components,
        explained_variance_ratio=ratios,
        cumulative_variance_ratio=cumulative,
    )
