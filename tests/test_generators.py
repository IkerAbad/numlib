import numpy as np
import pytest

from numlib import (
    condition_number,
    hilbert,
    random_gaussian,
    vandermonde,
    wilkinson,
    with_condition,
)


def test_hilbert_values_and_symmetry() -> None:
    matrix = hilbert(4)
    assert matrix[0][0] == 1.0
    assert matrix[1][2] == pytest.approx(0.25)
    np.testing.assert_allclose(matrix, np.array(matrix).T)


def test_hilbert_condition_grows() -> None:
    assert condition_number(hilbert(8)) > condition_number(hilbert(5))


def test_vandermonde() -> None:
    matrix = vandermonde([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        matrix,
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 9.0],
        ],
    )
    assert len(matrix[0]) == 3


def test_vandermonde_degree() -> None:
    matrix = vandermonde([2.0, 4.0], degree=4)
    np.testing.assert_allclose(
        matrix,
        [[1.0, 2.0, 4.0, 8.0], [1.0, 4.0, 16.0, 64.0]],
    )


def test_random_gaussian_is_reproducible() -> None:
    first = random_gaussian(4, 3, seed=5)
    second = random_gaussian(4, 3, seed=5)
    assert first == second
    assert (len(first), len(first[0])) == (4, 3)


def test_with_condition_matches_requested_value() -> None:
    for target in (1.0, 1e3, 1e6, 1e9):
        matrix = with_condition(12, 8, target, seed=3)
        measured = condition_number(matrix)
        assert measured == pytest.approx(target, rel=0.05)


def test_with_condition_square() -> None:
    matrix = with_condition(10, 10, 1e8, seed=4)
    assert condition_number(matrix) == pytest.approx(1e8, rel=0.05)


def test_wilkinson_matrix_shape() -> None:
    matrix = wilkinson(6)
    np.testing.assert_allclose(matrix, np.array(matrix).T)
    assert len(matrix) == 6
    assert matrix[0][1] == 1.0
    assert matrix[1][0] == 1.0
    assert matrix[0][2] == 0.0


def test_invalid_generator_arguments_raise() -> None:
    with pytest.raises(ValueError):
        hilbert(0)
    with pytest.raises(ValueError):
        vandermonde([])
    with pytest.raises(ValueError):
        with_condition(2, 3, 10.0)
    with pytest.raises(ValueError):
        with_condition(3, 3, 0.5)
