"""numlib: numerical linear algebra from scratch in pure Python.

The algorithms operate on lists of lists of floats. NumPy is used for input
and output, and by the benchmark module that compares against LAPACK.
"""

from __future__ import annotations

from numlib.analysis import (
    backward_error,
    condition_number,
    forward_error,
    loss_of_orthogonality,
)
from numlib.eigen import Eigen, eigen_qr, eigenvalues, eigenvectors, hessenberg
from numlib.exceptions import (
    ConvergenceError,
    NumlibError,
    RankDeficientError,
    SingularMatrixError,
)
from numlib.generators import (
    hilbert,
    random_gaussian,
    vandermonde,
    wilkinson,
    with_condition,
)
from numlib.lstsq import LeastSquares, lstsq_normal, lstsq_qr
from numlib.lu import LU, lu, solve
from numlib.pca import PCA, pca
from numlib.qr import QR, qr_gram_schmidt, qr_householder
from numlib.svd import SVD, svd

__version__ = "0.1.0"

__all__ = [
    "LU",
    "PCA",
    "QR",
    "SVD",
    "ConvergenceError",
    "Eigen",
    "LeastSquares",
    "NumlibError",
    "RankDeficientError",
    "SingularMatrixError",
    "__version__",
    "backward_error",
    "condition_number",
    "eigen_qr",
    "eigenvalues",
    "eigenvectors",
    "forward_error",
    "hessenberg",
    "hilbert",
    "loss_of_orthogonality",
    "lstsq_normal",
    "lstsq_qr",
    "lu",
    "pca",
    "qr_gram_schmidt",
    "qr_householder",
    "random_gaussian",
    "solve",
    "svd",
    "vandermonde",
    "wilkinson",
    "with_condition",
]
