from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from numlib import (
    SingularMatrixError,
    loss_of_orthogonality,
    lu,
    qr_householder,
    svd,
)
from numlib._ops import matmul

FLOAT = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def tall_matrices(draw):
    rows = draw(st.integers(min_value=1, max_value=6))
    columns = draw(st.integers(min_value=1, max_value=rows))
    values = draw(st.lists(FLOAT, min_size=rows * columns, max_size=rows * columns))
    return [values[i * columns : (i + 1) * columns] for i in range(rows)]


@st.composite
def square_matrices(draw):
    size = draw(st.integers(min_value=1, max_value=6))
    values = draw(st.lists(FLOAT, min_size=size * size, max_size=size * size))
    return [values[i * size : (i + 1) * size] for i in range(size)]


@given(tall_matrices())
@settings(max_examples=50, deadline=None)
def test_qr_reconstructs_the_matrix(matrix: list[list[float]]) -> None:
    scale = float(np.linalg.norm(matrix))
    assume(scale > 1e-6)
    factor = qr_householder(matrix)
    reconstructed = np.array(matmul(factor.Q, factor.R))
    np.testing.assert_allclose(
        reconstructed, np.array(matrix), atol=1e-11 * max(1.0, scale), rtol=0
    )
    assert loss_of_orthogonality(factor.Q) < 1e-12


@given(tall_matrices())
@settings(max_examples=50, deadline=None)
def test_svd_reconstructs_the_matrix(matrix: list[list[float]]) -> None:
    scale = float(np.linalg.norm(matrix))
    assume(scale > 1e-6)
    factor = svd(matrix)
    u = np.array(factor.U)
    singular = np.array(factor.S)
    v = np.array(factor.V)
    reconstructed = u @ np.diag(singular) @ v.T
    np.testing.assert_allclose(reconstructed, np.array(matrix), atol=1e-9 * max(1.0, scale), rtol=0)
    assert np.all(singular >= 0.0)
    assert np.all(np.diff(singular) <= 0.0)
    assert loss_of_orthogonality(factor.U) < 1e-9
    assert loss_of_orthogonality(factor.V) < 1e-9


@given(square_matrices())
@settings(max_examples=40, deadline=None)
def test_lu_reconstructs_the_matrix(matrix: list[list[float]]) -> None:
    scale = float(np.linalg.norm(matrix))
    assume(scale > 1e-6)
    try:
        factors = (lu(matrix), lu(matrix, pivoting="full"))
    except SingularMatrixError:
        assume(False)
        return
    for factor in factors:
        left = np.array(matmul(factor.P, matrix))
        if factor.Q is not None:
            left = left @ np.array(factor.Q)
        right = np.array(matmul(factor.L, factor.U))
        np.testing.assert_allclose(left, right, atol=1e-10 * max(1.0, scale), rtol=0)
