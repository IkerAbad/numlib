# Changelog

## 0.1.0 - 2026-11-23

- LU decomposition with partial and complete pivoting, triangular solves and
  growth-factor reporting.
- QR factorization by Householder reflections, modified Gram-Schmidt with
  optional reorthogonalization, and classical Gram-Schmidt for comparison.
- Singular value decomposition by one-sided Jacobi rotations.
- Real eigenvalues and eigenvectors by Hessenberg reduction, Wilkinson shifts
  and the implicit Francis double shift. Eigenvectors use inverse iteration on
  the real Schur form and back transformation with the accumulated Householder
  matrix.
- Linear least squares by Householder QR and by the normal equations.
- Condition-number, forward-error, backward-error and loss-of-orthogonality
  helpers.
- Test-matrix generators: Hilbert, Vandermonde, Gaussian, prescribed condition
  number and Wilkinson.
- Benchmarks against NumPy/LAPACK, four regenerable figures, and a generated
  eigenvector and PCA verification report.
- Principal component analysis from a covariance matrix, with a hand-built
  three-variable example.
- PyPI distribution name: `numlib-linalg` (`numlib` was already registered by
  an unrelated project). The import package remains `numlib`.
