from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kageframe import profile_dataframe
from kageframe.datasets import latent_correlation
from kageframe.dependence import observed_to_latent, pairwise_latent_correlation
from kageframe.exceptions import KageFrameWarning
from kageframe.latent import (
    hermite_coefficients,
    interval_scores,
    invert_mehler,
    nominal_b_matrix,
    nominal_choice,
)
from kageframe.schema import LatentLabel


def _latent(profile) -> tuple[list[str], np.ndarray]:
    labels = [lb.key for lb in profile.latent_labels()]
    return labels, np.array(profile.dependence.latent.matrix, dtype=float)


def test_interval_scores() -> None:
    scores, sd = interval_scores(np.array([0.5, 0.5]))
    assert scores == pytest.approx([-np.sqrt(2 / np.pi), np.sqrt(2 / np.pi)])
    assert sd == pytest.approx(np.sqrt(2 / np.pi))
    p = np.array([0.1, 0.2, 0.3, 0.4])
    scores, _ = interval_scores(p)
    assert np.dot(p, scores) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(scores) > 0)


@pytest.mark.parametrize("probs", [[0.5, 0.5], [0.1, 0.2, 0.3, 0.4], [0.02, 0.9, 0.08]])
def test_hermite_first_coefficient_is_score_variance(probs) -> None:
    p = np.array(probs)
    scores, _ = interval_scores(p)
    beta = hermite_coefficients(p)
    assert beta[0] == pytest.approx(np.dot(p, scores**2))
    # Parseval: the truncated series (40 terms) keeps most of the variance of a step function.
    assert 0.85 * beta[0] < np.sum(beta**2) <= beta[0] * (1 + 1e-9)


def test_invert_mehler_recovers_rho_exactly() -> None:
    bi = hermite_coefficients(np.array([0.7, 0.3]))
    bj = hermite_coefficients(np.array([0.2, 0.3, 0.4, 0.1]))
    rho = np.array([-0.8, -0.3, 0.0, 0.4, 0.9])
    powers = np.arange(1, len(bi) + 1)
    cov = np.sum(bi * bj * rho[:, None] ** powers, axis=1)
    got = invert_mehler(cov, np.tile(bi, (5, 1)), np.tile(bj, (5, 1)))
    assert got == pytest.approx(rho, abs=1e-6)


MIXED_TYPES = {"x": "numeric", "b1": "binary", "b2": "binary", "o": "ordinal"}


def _equicorrelation(rho: float, signs=(1, 1, 1, 1)) -> np.ndarray:
    d = np.diag(signs).astype(float)
    return d @ (np.full((4, 4), rho) + (1 - rho) * np.eye(4)) @ d


def _simulated(n: int, cov: np.ndarray, seed: int) -> pd.DataFrame:
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(4), cov, size=n)
    return pd.DataFrame({
        "x": np.exp(z[:, 0]),
        "b1": (z[:, 1] > stats.norm.ppf(0.85)).astype(int),
        "b2": (z[:, 2] > stats.norm.ppf(0.3)).astype(int),
        "o": np.searchsorted(stats.norm.ppf([0.1, 0.5, 0.8]), z[:, 3]),
    })


@pytest.mark.parametrize(("rho", "signs"), [(0.3, (1, 1, 1, 1)), (0.7, (1, 1, 1, 1)),
                                            (0.5, (1, -1, 1, -1))])
def test_latent_correlation_recovered_for_mixed_types(rho, signs) -> None:
    truth = _equicorrelation(rho, signs)
    p = profile_dataframe(_simulated(40000, truth, seed=1), types=MIXED_TYPES)
    _, m = _latent(p)
    assert np.max(np.abs(m - truth)) < 0.03


def test_pairwise_missing_uses_joint_rows() -> None:
    df = _simulated(20000, _equicorrelation(0.5), seed=4)
    rng = np.random.default_rng(5)
    df["x"] = df["x"].mask(rng.random(len(df)) < 0.3)
    df["o"] = df["o"].astype("Int64").mask(rng.random(len(df)) < 0.4)
    p = profile_dataframe(df, types=MIXED_TYPES)
    labels, m = _latent(p)
    assert np.max(np.abs(m[np.triu_indices(4, 1)] - 0.5)) < 0.04
    i, j = labels.index("x"), labels.index("o")
    expected = int((df["x"].notna() & df["o"].notna()).sum())
    assert p.dependence.observed.n_pairs[i][j] == expected


def test_sparse_pairs_set_to_zero_and_warn() -> None:
    n = 400
    rng = np.random.default_rng(0)
    z = rng.normal(size=n)
    a = np.where(np.arange(n) < 200, z, np.nan)
    b = np.where(np.arange(n) >= 190, z + 0.01 * rng.normal(size=n), np.nan)
    df = pd.DataFrame({"a": a, "b": b})
    with pytest.warns(KageFrameWarning, match="jointly observed"):
        p = profile_dataframe(df)
    assert p.dependence.latent.matrix[0][1] == 0.0
    assert any(w.code == "sparse_pairs" for w in p.warnings)


def test_pairwise_latent_correlation_marks_nominal_blocks() -> None:
    feats = np.random.default_rng(0).normal(size=(100, 3))
    cells = [np.full(100, 0.01)] * 3
    corr, n_pairs, n_sparse = pairwise_latent_correlation(feats, cells, [0, 1, 1], 30)
    assert np.isnan(corr[1, 2]) and not np.isnan(corr[0, 1])
    assert n_pairs[0, 1] == 100 and n_sparse == 0


@pytest.mark.parametrize("probs", [[0.5, 0.3, 0.2], [0.42, 0.23, 0.2, 0.15]])
def test_b_matrix_matches_monte_carlo(probs) -> None:
    p = np.array(probs)
    k = len(p)
    rng = np.random.default_rng(11)
    n = 400000
    r = np.linspace(0.5, -0.3, k)
    z = rng.standard_normal((n, k))  # scores within a nominal block are independent
    x = z @ r + np.sqrt(1 - r @ r) * rng.standard_normal(n)
    y = nominal_choice(z, p)
    assert np.bincount(y, minlength=k) / n == pytest.approx(p, abs=0.004)
    cov = np.array([np.mean(x * (y == j)) for j in range(k)])
    assert cov == pytest.approx(nominal_b_matrix(p) @ r, abs=0.005)
    assert nominal_b_matrix(p).sum(axis=0) == pytest.approx(np.zeros(k), abs=1e-6)


def test_observed_to_latent_nominal_mapping_reproduces_indicator_covariance() -> None:
    p = np.array([0.5, 0.3, 0.2])
    labels = [LatentLabel("x", "value")] + [LatentLabel("n", "level", lv)
                                            for lv in ("a", "b", "c")]
    b = nominal_b_matrix(p)
    c = b @ np.array([0.4, -0.1, -0.3])
    phi = stats.norm.pdf(stats.norm.ppf(1 - p))
    observed = np.eye(4)
    observed[0, 1:] = observed[1:, 0] = c / phi
    latent = observed_to_latent(observed, labels, {"n": p})
    assert b @ latent[1:, 0] == pytest.approx(c, abs=1e-8)
    assert latent[1:, 1:] == pytest.approx(np.eye(3))


def test_fixture_latent_correlation_recovered(clinical_profile) -> None:
    true_labels, truth = latent_correlation()
    labels, m = _latent(clinical_profile)
    idx = [true_labels.index(lb) for lb in labels]
    t = truth[np.ix_(idx, idx)]
    nominal = np.array(["[" in lb for lb in labels])
    plain = ~nominal
    assert np.max(np.abs(m[np.ix_(plain, plain)] - t[np.ix_(plain, plain)])) < 0.06
    # Nominal score correlations are identified only through B r (B has a null space).
    probs = np.array(clinical_profile.column("indication").probabilities)
    b = nominal_b_matrix(probs)
    diff = b @ m[np.ix_(nominal, plain)] - b @ t[np.ix_(nominal, plain)]
    assert np.max(np.abs(diff)) < 0.035
    assert m[np.ix_(nominal, nominal)] == pytest.approx(np.eye(nominal.sum()))
    assert not clinical_profile.dependence.psd_correction["applied"] or \
        clinical_profile.dependence.psd_correction["max_abs_change"] < 0.1


def test_latent_matrix_is_positive_definite(clinical_profile) -> None:
    _, m = _latent(clinical_profile)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1.0)
    assert np.linalg.eigvalsh(m).min() > 0


def test_no_latent_columns() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = profile_dataframe(pd.DataFrame({"id": [f"P{i:05d}" for i in range(200)]}))
    assert p.latent_labels() == []
