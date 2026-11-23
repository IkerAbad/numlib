"""Exceptions raised by numlib."""

from __future__ import annotations


class NumlibError(Exception):
    """Base class for all numlib errors."""


class SingularMatrixError(NumlibError, ValueError):
    """Raised when a factorization meets an exactly zero pivot."""


class RankDeficientError(NumlibError, ValueError):
    """Raised when a Gram-Schmidt factorization meets a dependent column."""


class ConvergenceError(NumlibError, RuntimeError):
    """Raised when an iterative algorithm fails to converge."""
