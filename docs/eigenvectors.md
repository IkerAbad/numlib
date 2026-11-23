# Eigenvectors

`numlib.eigen_qr` returns the eigenvalues, the real Schur form, the accumulated
orthogonal matrix and the right eigenvectors of a real square matrix. The
eigenvectors are computed in two stages: the QR iteration accumulates the
Householder transformations, and inverse iteration solves the eigenvector
equation on the resulting quasi-triangular form.

## Method

1. **Hessenberg reduction.** Householder reflectors reduce `A` to upper
   Hessenberg form `H`, accumulating `Q0` so that `A = Q0 H Q0^T`.
2. **Shifted QR iteration.** The active trailing block is deflated when its
   subdiagonal entry is negligible. If the trailing 2x2 block has real
   eigenvalues, the step uses the Wilkinson shift closest to the bottom-right
   entry. If it has a complex conjugate pair, the step uses the implicit
   Francis double shift with the two eigenvalues of that block. Every
   Householder reflector is applied on both sides of `H` and accumulated into
   `Q`, so the final factorization is `A = Q T Q^T` with `T` quasi upper
   triangular.
3. **Eigenvalues of the Schur form.** A 1x1 block contributes one real
   eigenvalue. A 2x2 block is solved analytically with the quadratic formula,
   which yields either two real eigenvalues or a complex conjugate pair.
4. **Inverse iteration on `T`.** For each eigenvalue `lambda`, the shifted
   system `(T - lambda I) y = v` is solved by block back substitution on the
   quasi-triangular form. The starting vector is a coordinate vector located
   at the corresponding 1x1 or 2x2 block, so repeated eigenvalues in different
   blocks start from different directions. When a block is exactly singular,
   the pivot or the 2x2 determinant is replaced by
   `delta = 4 eps max(1, ||T||_F)`. This is the standard regularized inverse
   iteration: the solution is dominated by the eigenvector direction, and the
   iteration is normalized after every solve. The solve is repeated at most
   `vector_iterations` times (20 by default) and stops when the relative Schur
   residual is below `vector_tol` (1e-12 by default).
5. **Back transformation.** The eigenvector of `A` is `x = Q y`. It is
   normalized in the 2-norm and its largest component is rotated to be real
   and positive, which removes the arbitrary complex phase.
6. **Residual.** For every eigenvector the implementation reports
   `||T y - lambda y|| / (||T||_F + |lambda|)`. Since `Q` is orthogonal, this
   is the same relative residual as `||A x - lambda x||` up to the rounding
   error of the accumulated transformations.

## Complexity

With `n` the matrix order and `k` the number of inverse iterations per
eigenvalue:

| Stage | Time | Storage |
| --- | --- | --- |
| Hessenberg reduction | `O(n^3)` | `O(n^2)` |
| QR iteration and `Q` accumulation | `O(n^3)` for `O(n)` steps | `O(n^2)` |
| Inverse iteration on `T` | `O(k n^3)` for all eigenvectors | `O(n^2)` |
| Back transformation `Q y` | `O(n^3)` for all eigenvectors | `O(n^2)` |

Each block back substitution costs `O(n^2)`, so one eigenvector costs `O(k n^2)`
and all `n` eigenvectors cost `O(k n^3)`. A dense complex LU factorization
per eigenvalue would cost `O(n^4)` in total; the implementation deliberately
uses the quasi-triangular structure instead. `k` is usually one to three for
well-separated eigenvalues and can grow when eigenvalues are clustered.

## Why the eigenvectors are needed

The eigenvalues alone cannot distinguish the identity matrix from a Jordan
block with the same eigenvalue. Both `I2` and `[[1, 1], [0, 1]]` have the
spectrum `{1, 1}`, but the identity has two independent eigenvectors while the
Jordan block has only one. The rank of the returned eigenvector matrix exposes
the difference; the generated results table includes both cases side by side.
In PCA the eigenvalues give the explained variance, but only the eigenvectors
give the directions and loadings of the components.

## Accuracy and limits

- The eigenvalue error is governed by the QR iteration and is backward stable.
  The eigenvector error also depends on the separation between eigenvalues and
  on the condition number of the eigenvector matrix.
- For a **defective** eigenvalue the algebraic multiplicity is larger than the
  geometric multiplicity. There is no complete eigenbasis: the returned matrix
  `V` has rank smaller than `n`, its condition number is large, and some
  columns point in the same direction. Every column still satisfies
  `A v = lambda v` to working precision, and `Eigen.residuals` reports how well.
- For **repeated but diagonalizable** eigenvalues the block-local starting
  vectors produce an independent set, and `V` can be a full basis. The
  individual vectors are not unique, so an elementwise comparison with another
  library is not meaningful; compare invariant subspaces, residuals or the
  explained variance instead.
- For **non-normal** matrices the eigenvectors are not orthogonal. A large
  `||V^H V - I||` is not an error in that case; use the residual and
  `cond(V)` instead.
- The eigensolver returns right eigenvectors only. Left eigenvectors are not
  implemented.

## API

```python
from numlib import eigen_qr, eigenvectors

result = eigen_qr(A)
print(result.values)
print(result.vectors)  # columns are eigenvectors
print(result.residuals)  # one relative residual per column

values, vectors = eigenvectors(A)
```

`eigen_qr(..., compute_vectors=False)` skips the inverse iteration when only
the eigenvalues and the Schur form are needed.

## Verification

`python scripts/generate_report.py` runs the cases with known eigenvalues,
random general matrices, symmetric matrices, Wilkinson's matrix, repeated
eigenvalues and defective Jordan blocks. It compares the results with
`numpy.linalg.eig` and writes the measured table to
[`RESULTS.md`](RESULTS.md). The same command also writes the PCA explained
variance table.
