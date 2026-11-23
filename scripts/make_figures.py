"""Regenerate the figures in ``figures/`` from measured benchmarks."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from numlib.benchmark import (
    ErrorRecord,
    OrthogonalityRecord,
    ScalingRecord,
    decomposition_error_vs_size,
    decomposition_time_vs_size,
    lstsq_error_vs_condition,
    lstsq_error_vs_size,
    orthogonality_vs_condition,
    solve_error_vs_condition,
    solve_error_vs_size,
    solve_time_vs_size,
)

EPS = float(np.finfo(np.float64).eps)


def _floor(value: float) -> float:
    """Keep strictly positive values for log plots."""
    if not math.isfinite(value):
        return math.nan
    return max(value, 1e-17)


def _group_errors(records: Sequence[ErrorRecord], attribute: str) -> dict[str, list[ErrorRecord]]:
    grouped: dict[str, list[ErrorRecord]] = {}
    for record in records:
        grouped.setdefault(record.method, []).append(record)
    for items in grouped.values():
        items.sort(key=lambda record: float(getattr(record, attribute)))
    return grouped


def _group_orthogonality(
    records: Sequence[OrthogonalityRecord],
) -> dict[str, list[OrthogonalityRecord]]:
    grouped: dict[str, list[OrthogonalityRecord]] = {}
    for record in records:
        grouped.setdefault(record.method, []).append(record)
    for items in grouped.values():
        items.sort(key=lambda record: record.condition)
    return grouped


def _group_scaling(
    records: Sequence[ScalingRecord],
) -> dict[str, list[ScalingRecord]]:
    grouped: dict[str, list[ScalingRecord]] = {}
    for record in records:
        grouped.setdefault(record.method, []).append(record)
    for items in grouped.values():
        items.sort(key=lambda record: record.n)
    return grouped


def _style(axis: Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)


def plot_error_vs_condition(path: Path) -> None:
    solve_records = solve_error_vs_condition()
    lstsq_records = lstsq_error_vs_condition()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for method, records in _group_errors(solve_records, "condition").items():
        left.loglog(
            [record.condition for record in records],
            [_floor(record.forward_error) for record in records],
            marker="o",
            label=method,
        )
    grid = np.logspace(0.0, 12.0, 120)
    left.loglog(grid, EPS * grid, "k--", linewidth=1.0, label="eps * cond")
    left.set_ylim(1e-17, 1e2)
    _style(
        left,
        "LU solve: forward error vs conditioning",
        "2-norm condition number",
        "relative forward error",
    )

    for method, records in _group_errors(lstsq_records, "condition").items():
        right.loglog(
            [record.condition for record in records],
            [_floor(record.forward_error) for record in records],
            marker="o",
            label=method,
        )
    right.loglog(grid, EPS * grid, "k--", linewidth=1.0, label="eps * cond")
    right.loglog(
        grid,
        EPS * grid * grid,
        "k:",
        linewidth=1.0,
        label="eps * cond^2",
    )
    right.set_ylim(1e-17, 1e2)
    _style(
        right,
        "Least squares: QR vs normal equations",
        "2-norm condition number of A",
        "relative forward error",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"wrote {path}")


def plot_error_vs_size(path: Path) -> None:
    solve_records = solve_error_vs_size() + lstsq_error_vs_size()
    decomposition_records = decomposition_error_vs_size()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for method, records in _group_errors(solve_records, "n").items():
        left.semilogy(
            [record.n for record in records],
            [_floor(record.forward_error) for record in records],
            marker="o",
            label=method,
        )
    left.set_ylim(1e-17, 1e-8)
    _style(
        left,
        "Solve accuracy vs size (condition number 100)",
        "matrix size n",
        "relative forward error",
    )

    for method, records in _group_errors(decomposition_records, "n").items():
        right.semilogy(
            [record.n for record in records],
            [_floor(record.forward_error) for record in records],
            marker="o",
            label=method,
        )
    right.axhline(EPS, color="k", linestyle="--", linewidth=1.0, label="eps")
    right.set_ylim(1e-17, 1e-8)
    _style(
        right,
        "Decomposition accuracy vs size",
        "matrix size n",
        "relative error",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"wrote {path}")


def plot_time_vs_size(path: Path) -> None:
    solve_records = solve_time_vs_size()
    decomposition_records = decomposition_time_vs_size()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for method, records in _group_scaling(solve_records).items():
        left.loglog(
            [record.n for record in records],
            [record.seconds for record in records],
            marker="o",
            label=method,
        )
    _style(
        left,
        "LU and QR least squares: time vs size",
        "matrix size n",
        "wall-clock time (s)",
    )

    for method, records in _group_scaling(decomposition_records).items():
        right.loglog(
            [record.n for record in records],
            [record.seconds for record in records],
            marker="o",
            label=method,
        )
    _style(
        right,
        "SVD and eigenvalues: time vs size",
        "matrix size n",
        "wall-clock time (s)",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"wrote {path}")


def plot_orthogonality(path: Path) -> None:
    records = orthogonality_vs_condition()
    figure, axis = plt.subplots(figsize=(7.0, 4.8))

    for method, items in _group_orthogonality(records).items():
        axis.loglog(
            [item.condition for item in items],
            [_floor(item.loss) for item in items],
            marker="o",
            label=method,
        )
    grid = np.logspace(0.0, 12.0, 120)
    axis.loglog(grid, EPS * grid, "k--", linewidth=1.0, label="eps * cond")
    axis.loglog(grid, EPS * grid * grid, "k:", linewidth=1.0, label="eps * cond^2")
    axis.set_ylim(1e-17, 1e2)
    _style(
        axis,
        "Loss of orthogonality of the QR factorizations",
        "2-norm condition number",
        "||Q^T Q - I||_F",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="directory for the generated PNG files",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    plot_error_vs_condition(arguments.output / "error_vs_condition.png")
    plot_error_vs_size(arguments.output / "error_vs_size.png")
    plot_time_vs_size(arguments.output / "time_vs_size.png")
    plot_orthogonality(arguments.output / "orthogonality.png")


if __name__ == "__main__":
    main()
