# numlib 0.1.0

Dense numerical linear algebra from scratch in Python.

## Highlights

- LU with partial and complete pivoting, triangular solves and growth factor.
- QR by Householder, modified Gram-Schmidt with optional reorthogonalization,
  and classical Gram-Schmidt for comparison.
- SVD by one-sided Jacobi rotations.
- Real eigenvalues and right eigenvectors by Hessenberg reduction, Wilkinson
  shifts, the implicit Francis double shift, inverse iteration on the real
  Schur form and back transformation with the accumulated Householder matrix.
- Least squares by QR and by the normal equations.
- Condition number, forward error, normwise backward error and loss of
  orthogonality.
- PCA from a covariance matrix, with explained variance and loadings.
- Test-matrix generators, LAPACK benchmarks, four regenerable figures and a
  generated eigenvector/PCA verification report.

## Install

The PyPI distribution name is `numlib-linalg`; the import package is `numlib`.

The 0.1.0 wheel and sdist are attached to this release:

```bash
pip install https://github.com/IkerAbad/numlib/releases/download/v0.1.0/numlib_linalg-0.1.0-py3-none-any.whl
```

The PyPI upload is pending the trusted-publisher configuration documented in
`docs/PUBLISHING.md`. Once configured, the same version will be installable
with `pip install numlib-linalg==0.1.0`.

## Documentation

- `README.md`: usage and figures.
- `docs/eigenvectors.md`: eigenvector method, complexity and limits.
- `docs/RESULTS.md`: generated comparison with `numpy.linalg.eig` and PCA.
- `CHANGELOG.md`: release notes.
