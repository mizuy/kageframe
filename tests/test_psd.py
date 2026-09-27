from __future__ import annotations

import numpy as np
import pytest

from kageframe.psd import EIG_FLOOR, nearest_correlation


def _indefinite() -> np.ndarray:
    return np.array([
        [1.0, 0.9, 0.7, 0.0],
        [0.9, 1.0, -0.4, 0.0],
        [0.7, -0.4, 1.0, 0.5],
        [0.0, 0.0, 0.5, 1.0],
    ])


def test_positive_definite_input_unchanged() -> None:
    a = np.array([[1.0, 0.3], [0.3, 1.0]])
    out, info = nearest_correlation(a)
    assert np.array_equal(out, a) and not info["applied"]
    assert info["min_eigenvalue_before"] == pytest.approx(0.7)


def test_indefinite_input_is_repaired() -> None:
    a = _indefinite()
    assert np.linalg.eigvalsh(a).min() < 0
    out, info = nearest_correlation(a)
    assert info["applied"] and info["min_eigenvalue_before"] < 0
    assert np.linalg.eigvalsh(out).min() >= EIG_FLOOR / 2
    assert np.allclose(np.diag(out), 1.0) and np.allclose(out, out.T)
    assert info["max_abs_change"] == pytest.approx(np.abs(out - a).max())
    assert info["frobenius_change"] < 0.5
    np.linalg.cholesky(out)


def test_fixed_entries_are_kept() -> None:
    a = _indefinite()
    fixed = np.zeros((4, 4), dtype=bool)
    fixed[0, 3] = fixed[3, 0] = fixed[1, 3] = fixed[3, 1] = True
    out, _ = nearest_correlation(a, fixed)
    assert out[0, 3] == 0.0 and out[1, 3] == 0.0
    assert np.linalg.eigvalsh(out).min() >= EIG_FLOOR / 2


def test_empty_matrix() -> None:
    out, info = nearest_correlation(np.zeros((0, 0)))
    assert out.shape == (0, 0) and not info["applied"]
