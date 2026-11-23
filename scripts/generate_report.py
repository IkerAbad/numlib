"""Generate docs/RESULTS.md and the eigenvector verification JSON.

Every number published in the results document comes from this script. The
numerical work uses numlib; NumPy is used only as the reference implementation
and for formatting the measured data.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import numlib
from numlib import eigen_qr, wilkinson

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    matrix: NDArray[np.float64]
    compare_vectors: bool
    note: str = ""


@dataclass(frozen=True)
class VerificationRow:
    case: str
    n: int
    eigenvalue_error: float
    eigenvector_error: float | None
    residual: float
    max_reported_residual: float
    condition: float
    rank: int
    orthogonality: float
    note: str


def _random_general(seed: int, n: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n))


def _random_symmetric(seed: int, n: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n))
    return 0.5 * (matrix + matrix.T)


def _cases() -> list[Case]:
    return [
        Case("diagonal n=4", np.diag([1.0, 2.0, 3.0, 4.0]), True),
        Case("rotation n=2", np.array([[0.0, -1.0], [1.0, 0.0]]), True),
        Case("random general n=5", _random_general(0, 5), True),
        Case("random general n=8", _random_general(1, 8), True),
        Case("symmetric n=6", _random_symmetric(2, 6), True),
        Case("Wilkinson n=6", np.array(wilkinson(6)), True),
        Case("repeated identity n=4", np.eye(4), False, "repeated eigenvalue"),
        Case(
            "repeated diagonal n=4",
            np.diag([2.0, 2.0, 3.0, 3.0]),
            False,
            "repeated eigenvalues",
        ),
        Case(
            "defective Jordan n=2",
            np.array([[1.0, 1.0], [0.0, 1.0]]),
            False,
            "defective",
        ),
        Case(
            "defective Jordan n=3",
            np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]]),
            False,
            "defective",
        ),
    ]


def _nearest_eigenvalue_error(values: list[complex], reference: NDArray[np.complex128]) -> float:
    worst = 0.0
    for value in values:
        nearest = min(abs(value - candidate) for candidate in reference)
        worst = max(worst, nearest / max(1.0, abs(value)))
    return worst


def _phase_error(ours: NDArray[np.complex128], reference: NDArray[np.complex128]) -> float:
    ours = ours / np.linalg.norm(ours)
    reference = reference / np.linalg.norm(reference)
    phase = np.vdot(reference, ours)
    if abs(phase) == 0.0:
        return float(np.linalg.norm(ours - reference))
    return float(np.linalg.norm(ours - (phase / abs(phase)) * reference))


def _vector_error(
    values: list[complex],
    vectors: NDArray[np.complex128],
    reference_values: NDArray[np.complex128],
    reference_vectors: NDArray[np.complex128],
) -> float | None:
    for index, value in enumerate(values):
        separation = min(
            (abs(value - other) for j, other in enumerate(values) if j != index),
            default=1.0,
        )
        if separation < 1e-6:
            return None
    worst = 0.0
    used: set[int] = set()
    for index, value in enumerate(values):
        candidate = int(np.argmin(np.abs(reference_values - value)))
        if candidate in used:
            continue
        used.add(candidate)
        worst = max(
            worst,
            _phase_error(
                vectors[:, index],
                reference_vectors[:, candidate],
            ),
        )
    return worst


def _verify(case: Case) -> VerificationRow:
    matrix = case.matrix
    result = eigen_qr(matrix)
    vectors = np.array(result.vectors, dtype=np.complex128)
    values = list(result.values)
    reference_values, reference_vectors = np.linalg.eig(matrix)
    residual = float(
        np.linalg.norm(matrix @ vectors - vectors @ np.diag(values))
        / max(float(np.linalg.norm(matrix)), 1.0)
    )
    orthogonality = float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(len(matrix))))
    vector_error = (
        _vector_error(values, vectors, reference_values, reference_vectors)
        if case.compare_vectors
        else None
    )
    return VerificationRow(
        case=case.name,
        n=len(matrix),
        eigenvalue_error=_nearest_eigenvalue_error(values, reference_values),
        eigenvector_error=vector_error,
        residual=residual,
        max_reported_residual=max(result.residuals, default=0.0),
        condition=float(np.linalg.cond(vectors)),
        rank=int(np.linalg.matrix_rank(vectors, tol=1e-8)),
        orthogonality=orthogonality,
        note=case.note,
    )


def _format(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not np.isfinite(value):
        return "inf"
    return f"{value:.2e}"


def _max_eigenvector_error(rows: list[VerificationRow]) -> float:
    values: list[float] = []
    for row in rows:
        if row.eigenvector_error is not None:
            values.append(row.eigenvector_error)
    return max(values)


def _write_report(output: Path, json_output: Path) -> None:
    rows = [_verify(case) for case in _cases()]
    pca_process = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "pca.py"), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    pca_data: dict[str, Any] = json.loads(pca_process.stdout)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Measured results",
        "",
        f"Generated by `python scripts/generate_report.py` on {timestamp}.",
        f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, "
        f"numlib {numlib.__version__}, {platform.platform()}.",
        "",
        "## Eigenvectors against numpy.linalg.eig",
        "",
        "The eigenvalue error is the largest relative difference between a "
        "computed eigenvalue and the nearest eigenvalue from NumPy. The "
        "eigenvector error is the 2-norm distance after removing the arbitrary "
        "complex phase. For repeated or defective eigenvalues the individual "
        "vectors are not unique, so the elementwise comparison is not defined; "
        "the residual, the rank of the eigenvector matrix and its condition "
        "number describe those cases instead. The orthogonality column is a "
        "diagnostic: for a non-normal matrix the eigenvectors are not expected "
        "to be orthogonal.",
        "",
        "| Case | n | eigenvalue error | eigenvector error | residual | "
        "max reported residual | cond(V) | rank(V) | ||V^H V - I|| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.case} | {row.n} | {_format(row.eigenvalue_error)} | "
            f"{_format(row.eigenvector_error)} | {_format(row.residual)} | "
            f"{_format(row.max_reported_residual)} | {_format(row.condition)} | "
            f"{row.rank}/{row.n} | {_format(row.orthogonality)} |"
        )

    comparable = [row for row in rows if row.eigenvector_error is not None]
    if comparable:
        lines.extend(
            [
                "",
                f"Across the cases with distinct eigenvalues, the largest "
                f"relative eigenvalue error is "
                f"`{max(row.eigenvalue_error for row in comparable):.2e}` and "
                f"the largest eigenvector error is "
                f"`{_max_eigenvector_error(comparable):.2e}`. "
                f"These are the headline numbers against NumPy.",
            ]
        )

    defects = [row for row in rows if row.note]
    if defects:
        lines.extend(
            [
                "",
                "Cases marked as repeated or defective: "
                + ", ".join(f"{row.case} ({row.note}, rank {row.rank}/{row.n})" for row in defects)
                + ". A defective eigenvalue has fewer independent eigenvectors "
                "than its algebraic multiplicity, so `V` cannot be a basis even "
                "though every returned column satisfies `A v = lambda v` to "
                "working precision.",
                "",
            ]
        )

    eigenvalues = pca_data["eigenvalues"]
    ratios = pca_data["explained_variance_ratio"]
    cumulative = pca_data["cumulative_variance_ratio"]
    components = pca_data["components"]
    variables = pca_data["variables"]

    lines.extend(
        [
            "## PCA on a hand-built covariance matrix",
            "",
            "The data set is the eight-observation, three-variable table in "
            "`examples/pca.py`. The covariance matrix is computed from the "
            "observations with denominator `n - 1`; the decomposition uses "
            "numlib's eigensolver. The reference is `numpy.linalg.eigh`.",
            "",
            "| PC | eigenvalue | explained variance | cumulative |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for index, (value, ratio, total) in enumerate(
        zip(eigenvalues, ratios, cumulative, strict=True), start=1
    ):
        lines.append(f"| {index} | {value:.6f} | {ratio:.6%} | {total:.6%} |")
    lines.extend(
        [
            "",
            f"Maximum eigenvalue difference against NumPy: "
            f"`{pca_data['max_eigenvalue_difference']:.2e}`.",
            f"Maximum explained-variance difference against NumPy: "
            f"`{pca_data['max_ratio_difference']:.2e}`.",
            "",
            "| Variable | "
            + " | ".join(f"PC{index}" for index in range(1, len(variables) + 1))
            + " |",
            "| --- | " + " | ".join("---:" for _ in variables) + " |",
        ]
    )
    for variable_index, variable in enumerate(variables):
        lines.append(
            f"| {variable} | "
            + " | ".join(
                f"{components[variable_index][column]:+.6f}" for column in range(len(variables))
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Reproducing",
            "",
            "```bash",
            "python scripts/generate_report.py",
            "python scripts/make_figures.py",
            "```",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            {
                "eigenvectors": [
                    {
                        "case": row.case,
                        "n": row.n,
                        "eigenvalue_error": row.eigenvalue_error,
                        "eigenvector_error": row.eigenvector_error,
                        "residual": row.residual,
                        "max_reported_residual": row.max_reported_residual,
                        "condition": row.condition,
                        "rank": row.rank,
                        "orthogonality": row.orthogonality,
                        "note": row.note,
                    }
                    for row in rows
                ],
                "pca": pca_data,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {json_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "RESULTS.md",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "docs" / "eigenvector_verification.json",
    )
    arguments = parser.parse_args()
    _write_report(arguments.output, arguments.json_output)


if __name__ == "__main__":
    main()
