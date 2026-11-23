import numpy as np
import pytest

from numlib import (
    RankDeficientError,
    hilbert,
    loss_of_orthogonality,
    qr_gram_schmidt,
    qr_householder,
)
from numlib._ops import matmul


def _check_qr(factor: object) -> None:
    from numlib.qr import QR

    assert isinstance(factor, QR)
    reconstructed = np.array(matmul(factor.Q, factor.R))
    assert reconstructed.shape == (len(factor.Q), len(factor.R[0]))
    assert np.allclose(np.triu(np.array(factor.R)), np.array(factor.R), atol=1e-14)
    assert loss_of_orthogonality(factor.Q) < 1e-12


def test_householder_tall_matrix() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((8, 3))
    factor = qr_householder(matrix)
    np.testing.assert_allclose(matmul(factor.Q, factor.R), matrix.tolist(), atol=1e-14)
    _check_qr(factor)
    assert factor.method == "householder"


def test_householder_square_and_scalar() -> None:
    rng = np.random.default_rng(1)
    for n in (1, 4):
        matrix = rng.standard_normal((n, n))
        factor = qr_householder(matrix)
        np.testing.assert_allclose(matmul(factor.Q, factor.R), matrix.tolist(), atol=1e-14)
        _check_qr(factor)


@pytest.mark.parametrize("orthogonalize", [False, True])
def test_modified_gram_schmidt(orthogonalize: bool) -> None:
    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((7, 4))
    factor = qr_gram_schmidt(matrix, reorthogonalize=orthogonalize)
    np.testing.assert_allclose(matmul(factor.Q, factor.R), matrix.tolist(), atol=1e-14)
    _check_qr(factor)
    assert factor.method == "modified-gram-schmidt"


def test_classical_gram_schmidt() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((6, 3))
    factor = qr_gram_schmidt(matrix, modified=False)
    np.testing.assert_allclose(matmul(factor.Q, factor.R), matrix.tolist(), atol=1e-14)
    _check_qr(factor)
    assert factor.method == "classical-gram-schmidt"


def test_householder_beats_mgs_on_hilbert_matrix() -> None:
    matrix = hilbert(8)
    householder = qr_householder(matrix)
    mgs = qr_gram_schmidt(matrix)
    twice = qr_gram_schmidt(matrix, reorthogonalize=True)
    householder_loss = loss_of_orthogonality(householder.Q)
    mgs_loss = loss_of_orthogonality(mgs.Q)
    twice_loss = loss_of_orthogonality(twice.Q)
    assert householder_loss < 1e-12
    assert mgs_loss > 1e3 * householder_loss
    assert twice_loss < 1e-10
    assert twice_loss < mgs_loss


def test_rank_deficient_gram_schmidt_raises() -> None:
    with pytest.raises(RankDeficientError):
        qr_gram_schmidt([[1.0, 2.0], [0.0, 0.0]])


def test_wide_matrix_raises() -> None:
    with pytest.raises(ValueError):
        qr_householder([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        qr_gram_schmidt([[1.0, 2.0, 3.0]])


def test_reorthogonalization_requires_modified_gs() -> None:
    with pytest.raises(ValueError):
        qr_gram_schmidt([[1.0], [1.0]], modified=False, reorthogonalize=True)
