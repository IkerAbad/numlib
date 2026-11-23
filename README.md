# numlib

[![CI](https://github.com/IkerAbad/numlib/actions/workflows/ci.yml/badge.svg)](https://github.com/IkerAbad/numlib/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Numerical linear algebra implemented from scratch in Python. The algorithms
operate on plain lists of floats; NumPy is used only at the input/output
boundary and by the benchmark code that compares against LAPACK.

## What is implemented

| Module | Method | Notes |
| --- | --- | --- |
| `numlib.lu` | Gaussian elimination, partial and complete pivoting | `P A Q = L U`, growth factor, triangular solves |
| `numlib.qr` | Householder reflections | Backward stable, orthogonality at machine precision |
| `numlib.qr` | Modified Gram-Schmidt, optional reorthogonalization | Cheaper; loses orthogonality like `cond(A) * eps` |
| `numlib.qr` | Classical Gram-Schmidt | Included only to show the `cond(A)^2 * eps` loss |
| `numlib.svd` | One-sided Jacobi rotations | High relative accuracy, no bidiagonalization |
| `numlib.eigen` | Hessenberg reduction + implicit Francis double shift | Wilkinson shifts from the trailing 2x2 block; real Schur form |
| `numlib.lstsq` | Least squares by Householder QR and by normal equations | Compares `cond(A)` with `cond(A)^2` |
| `numlib.analysis` | Condition number, forward error, normwise backward error | `cond_2` estimated with the package SVD |
| `numlib.generators` | Hilbert, Vandermonde, Gaussian, prescribed condition, Wilkinson | Reproducible with explicit seeds |
| `numlib.benchmark` | Systematic comparisons against `numpy.linalg` | Relative error, backward error, wall-clock time |

## Install

```bash
git clone https://github.com/IkerAbad/numlib.git
cd numlib
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Python 3.12 or newer is required. NumPy is a runtime dependency; matplotlib,
pytest, hypothesis, ruff and mypy are installed by the `dev` extra.

## Quick start

```python
import numpy as np

from numlib import (
    eigenvalues,
    loss_of_orthogonality,
    lstsq_qr,
    lu,
    qr_householder,
    solve,
    svd,
)

A = np.array(
    [
        [4.0, -2.0, 1.0],
        [-2.0, 4.0, -2.0],
        [1.0, -2.0, 4.0],
    ]
)
b = np.array([11.0, -16.0, 17.0])

x = solve(A, b)
print("solution:", x)

factor = lu(A)
print("growth factor:", factor.growth_factor)

qr = qr_householder(A)
print("R:", qr.R)
print("orthogonality loss:", loss_of_orthogonality(qr.Q))

decomposition = svd(A)
print("singular values:", decomposition.S)

print("eigenvalues:", eigenvalues(A))

least_squares = lstsq_qr(A, b)
print("least-squares solution:", least_squares.x)
print("residual:", least_squares.residual_norm)
print("backward error:", least_squares.backward_error)
```

## Stability notes

- **LU with partial pivoting** is backward stable in practice. The returned
  growth factor is `max |U| / max |A|`; complete pivoting also permutes columns
  and keeps that factor smaller, at the cost of an `O(n^2)` search per step.
- **Householder QR** is backward stable, and `Q` stays orthogonal to about
  machine precision regardless of `cond(A)`. **Modified Gram-Schmidt** is not
  backward stable in the same sense: its loss of orthogonality grows like
  `cond(A) * eps`. A second MGS pass restores orthogonality for the condition
  numbers tested here. **Classical Gram-Schmidt** grows like `cond(A)^2 * eps`.
- **One-sided Jacobi SVD** rotates the columns until they are orthogonal. It
  avoids bidiagonalization and keeps high relative accuracy for small singular
  values. Values below `eps * max(m, n) * sigma_max` are treated as zero so
  that the corresponding columns of `U` can be completed to an orthonormal
  basis; the discarded contribution is at working precision.
- **The eigensolver** reduces the matrix to upper Hessenberg form and applies
  the implicit Francis double shift, i.e. the two Wilkinson shifts taken from
  the trailing 2x2 block. A 1x1 block gives a real eigenvalue; a 2x2 block is
  solved analytically, which produces complex conjugate pairs without complex
  arithmetic in the iteration.
- **Least squares.** QR solves `R x = Q^T b` and has forward error proportional
  to `cond(A) * eps`. The normal equations solve `A^T A x = A^T b`, whose
  condition number is `cond(A)^2`; this is visible in the measurements below.
- **Backward error.** For a square system the reported quantity is the
  normwise relative backward error
  `||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf)`.
  For least squares it is applied to the square system that each method
  actually solves (`R x = Q^T b` or `A^T A x = A^T b`).

## Figures

All figures are regenerated from measured runs with:

```bash
python scripts/make_figures.py
```

The command writes four PNG files under `figures/`. The CI workflow also runs
it and uploads the images as build artifacts.

### Error versus condition number

![Forward error versus condition number](figures/error_vs_condition.png)

LU tracks the `cond(A) * eps` line and stays within a small factor of LAPACK up
to condition number `1e12`. In least squares, QR follows `cond(A) * eps`, while
the normal equations follow `cond(A)^2 * eps`. In this run, at `cond(A) = 1e12`
the normal-equation solution had relative error about `9e2`, against `3e-5` for
QR and `1e-5` for LAPACK.

### Loss of orthogonality

![Loss of orthogonality](figures/orthogonality.png)

Householder QR stays at roughly `2e-15` for every condition number tested. MGS
grows like `eps * cond(A)` and reaches `3e-5` at `1e12`. One reorthogonalization
brings it back to about `1e-15`. Classical Gram-Schmidt grows like
`eps * cond(A)^2` and is already of order one at `cond(A) = 1e8`.

On the Hilbert matrix of order 8 the measured losses were: Householder
`1.3e-15`, MGS `5.6e-7`, MGS with one reorthogonalization `4.4e-16`, classical
Gram-Schmidt `1.4`.

### Error versus size

![Error versus size](figures/error_vs_size.png)

For matrices with condition number 100, the relative forward error stays
between `1e-15` and `1e-13` as the size grows. The SVD and the symmetric QR
eigensolver stay within a few units of `1e-15`. There is no visible growth with
`n` in this range, which is what backward stability predicts.

### Time versus size

![Time versus size](figures/time_vs_size.png)

The pure-Python implementation is about two orders of magnitude slower than
LAPACK, as expected from scalar loops against compiled, blocked kernels. On
this machine, at `n = 40`, LU took about `1.8 ms` against `14 us` for LAPACK,
and QR least squares took about `20 ms` against `0.24 ms`. At `n = 16`, the SVD
took about `5 ms` against `38 us`, and the eigensolver about `7 ms` against
`47 us`. Absolute times depend on the machine and on the BLAS build.

## Acceptance checks

The test suite includes an explicit comparison with LAPACK for well-conditioned
matrices. In a representative run with condition number 10:

| Problem | numlib error | LAPACK error | Backward error |
| --- | --- | --- | --- |
| LU solve, `n = 20` | `1.8e-15` | `1.3e-15` | `1.3e-16` |
| QR least squares, `30 x 15` | `5.9e-16` | `1.5e-15` | `2.1e-17` |

Both are within a factor of two of LAPACK, well inside the required two orders
of magnitude. The property tests verify `A = Q R`, `A = U diag(S) V^T` and
`P A Q = L U` on random matrices, and the acceptance tests check that the
backward errors stay at machine precision.

## Development

```bash
ruff check .
ruff format --check .
mypy
pytest
```

The test suite uses pytest and hypothesis. Property tests cover random
matrices up to `6 x 6`; deterministic tests cover known factorizations, rank
deficiency, singular systems and the LAPACK comparison.

## Project layout

```text
numlib/
├── .github/workflows/ci.yml
├── figures/
├── scripts/make_figures.py
├── src/numlib/
│   ├── analysis.py
│   ├── benchmark.py
│   ├── eigen.py
│   ├── generators.py
│   ├── lstsq.py
│   ├── lu.py
│   ├── qr.py
│   └── svd.py
└── tests/
```

## Scope and limitations

- Dense matrices only. Sparse matrices, parallelism and GPU execution are out
  of scope.
- The algorithms are written for readability, not speed. The benchmarks stop at
  sizes where pure-Python `O(n^3)` loops remain practical (`n = 40` for LU and
  QR, `n = 16` for the SVD and the eigensolver).
- The eigensolver returns eigenvalues and a real Schur form, not eigenvectors.
- No iterative refinement, scaling or equilibration is applied.
- The timing comparison uses the LAPACK build and threading configuration
  installed with NumPy; it is indicative, not a controlled benchmark of BLAS.

## License

MIT. Copyright (c) 2026 Iker Abad.
