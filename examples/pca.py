"""Principal component analysis of a hand-built covariance matrix.

The data set contains eight observations of three variables: temperature in
degrees Celsius, relative humidity in percent, and energy use in kWh. The
covariance matrix is computed from the data in pure Python, and the
eigen-decomposition uses numlib. NumPy is used only to check the result.

Run ``python examples/pca.py`` for the human-readable report, or
``python examples/pca.py --json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from numlib import pca
from numlib._types import Matrix

DATA: list[list[float]] = [
    [20.0, 45.0, 120.0],
    [22.0, 40.0, 130.0],
    [24.0, 38.0, 150.0],
    [26.0, 35.0, 165.0],
    [28.0, 40.0, 180.0],
    [30.0, 42.0, 195.0],
    [32.0, 45.0, 210.0],
    [34.0, 50.0, 225.0],
]

VARIABLES = ("temperature", "humidity", "energy")


def covariance_matrix(data: list[list[float]]) -> Matrix:
    """Return the sample covariance matrix (denominator ``n - 1``)."""
    observations = len(data)
    variables = len(data[0])
    means = [sum(row[column] for row in data) / observations for column in range(variables)]
    covariance = [[0.0] * variables for _ in range(variables)]
    for row in data:
        centered = [row[column] - means[column] for column in range(variables)]
        for i in range(variables):
            for j in range(variables):
                covariance[i][j] += centered[i] * centered[j]
    return [[value / (observations - 1) for value in row] for row in covariance]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    covariance = covariance_matrix(DATA)
    result = pca(covariance)
    reference_values, _ = np.linalg.eigh(np.array(covariance))
    reference_values = np.sort(reference_values)[::-1]
    reference_ratios = reference_values / reference_values.sum()

    max_eigenvalue_difference = float(
        np.max(np.abs(np.array(result.eigenvalues) - reference_values))
    )
    max_ratio_difference = float(
        np.max(np.abs(np.array(result.explained_variance_ratio) - reference_ratios))
    )

    if arguments.json:
        print(
            json.dumps(
                {
                    "variables": list(VARIABLES),
                    "covariance": covariance,
                    "eigenvalues": result.eigenvalues,
                    "components": result.components,
                    "explained_variance_ratio": result.explained_variance_ratio,
                    "cumulative_variance_ratio": result.cumulative_variance_ratio,
                    "numpy_eigenvalues": reference_values.tolist(),
                    "numpy_explained_variance_ratio": reference_ratios.tolist(),
                    "max_eigenvalue_difference": max_eigenvalue_difference,
                    "max_ratio_difference": max_ratio_difference,
                }
            )
        )
        return

    print("Covariance matrix")
    for row in covariance:
        print("  " + "  ".join(f"{value:10.4f}" for value in row))
    print()
    print("Principal components (descending eigenvalue)")
    print("  PC  eigenvalue  explained  cumulative")
    for index, (value, ratio, cumulative) in enumerate(
        zip(
            result.eigenvalues,
            result.explained_variance_ratio,
            result.cumulative_variance_ratio,
            strict=True,
        ),
        start=1,
    ):
        print(f"  {index:2d}  {value:10.4f}  {ratio:9.4%}  {cumulative:10.4%}")
    print()
    print("Largest component loadings")
    for index in range(len(VARIABLES)):
        loadings = "  ".join(
            f"{VARIABLES[row]}={result.components[row][index]:+.4f}"
            for row in range(len(VARIABLES))
        )
        print(f"  PC{index + 1}: {loadings}")
    print()
    print(f"max eigenvalue difference vs numpy: {max_eigenvalue_difference:.2e}")
    print(f"max explained-variance difference vs numpy: {max_ratio_difference:.2e}")


if __name__ == "__main__":
    main()
