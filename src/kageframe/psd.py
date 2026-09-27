"""Nearest correlation matrix with fixed entries (Higham 2002, Dykstra-corrected)."""

from __future__ import annotations

from typing import Any

import numpy as np

EIG_FLOOR = 1e-6


def _project_psd(a: np.ndarray, eps: float) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    x = (v * np.maximum(w, eps)) @ v.T
    return (x + x.T) / 2


def nearest_correlation(
    a: np.ndarray,
    fixed: np.ndarray | None = None,
    *,
    eps: float = EIG_FLOOR,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a positive definite matrix close to ``a`` (Frobenius norm).

    ``fixed`` marks off-diagonal entries that must keep their value in ``a`` (the
    diagonal is always fixed to 1). The result has minimum eigenvalue >= ``eps / 2``
    and satisfies the fixed entries exactly.
    """
    a = (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T) / 2
    d = a.shape[0]
    mask = np.eye(d, dtype=bool) if fixed is None else (np.asarray(fixed, dtype=bool)
                                                         | np.eye(d, dtype=bool))
    target = a.copy()
    np.fill_diagonal(target, 1.0)
    start = a.copy()
    start[mask] = target[mask]
    info: dict[str, Any] = {"applied": False, "min_eigenvalue_before": None,
                            "frobenius_change": 0.0, "max_abs_change": 0.0, "iterations": 0}
    if d == 0:
        return start, info
    min_eig = float(np.linalg.eigvalsh(start).min())
    info["min_eigenvalue_before"] = min_eig
    if min_eig >= eps:
        return start, info

    y = start.copy()
    correction = np.zeros_like(y)
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007
        r = y - correction
        x = _project_psd(r, eps)
        correction = x - r
        y_new = x.copy()
        y_new[mask] = target[mask]
        step = np.linalg.norm(y_new - y) / max(1.0, np.linalg.norm(y))
        y = y_new
        if step < tol:
            break

    # Shrink toward the identity if rounding left it short of the floor; this keeps the
    # unit diagonal and fixed zeros unchanged.
    final_min = float(np.linalg.eigvalsh(y).min())
    if final_min < eps / 2:
        delta = eps - final_min
        y = (y + delta * np.eye(d)) / (1 + delta)
        np.fill_diagonal(y, 1.0)
    y = (y + y.T) / 2
    info.update(applied=True, frobenius_change=float(np.linalg.norm(y - start)),
                max_abs_change=float(np.abs(y - start).max()), iterations=iterations)
    return y, info
