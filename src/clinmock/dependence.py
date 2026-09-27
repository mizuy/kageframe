"""Pairwise latent correlation estimation and the observed -> latent (generation) mapping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import special

from .latent import hermite_coefficients, invert_mehler, nominal_b_matrix, normal_pdf
from .schema import LatentLabel, nominal_blocks

MAX_ABS_CORRELATION = 0.999


def pairwise_latent_correlation(
    features: np.ndarray, cells: Sequence[np.ndarray], blocks: Sequence[int],
    min_pair_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Latent correlations from pairwise-complete correlations of interval normal scores.

    ``features`` is ``n x d`` with NaN for missing values and ``cells[i]`` the latent cell
    probabilities of feature ``i``. The Pearson correlation ``r`` over rows where both are
    observed is converted to ``cov = r * sd_i * sd_j`` and the latent rho is found by
    inverting the Mehler series ``cov = sum_n rho^n beta_i[n] beta_j[n]`` (exact under the
    latent Gaussian model; first order it is ``r / (sd_i sd_j)``). Pairs with fewer than
    ``min_pair_count`` rows (or no variance) are 0; entries within the same nominal block
    are NaN. Returns (matrix, n_pairs, #sparse off-diagonal pairs).
    """
    n, d = features.shape
    if d == 0:
        return np.zeros((0, 0)), np.zeros((0, 0), dtype=np.int64), 0
    observed = ~np.isnan(features)
    m = observed.astype(float)
    s = np.where(observed, features, 0.0)
    n_pairs = m.T @ m
    sums = s.T @ m                      # sums[i, j] = sum of feature i where j observed
    sq = (s * s).T @ m
    cross = s.T @ s
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_ij = sums / n_pairs
        cov = cross / n_pairs - mean_ij * mean_ij.T
        var_ij = sq / n_pairs - mean_ij**2
        sample_corr = cov / np.sqrt(var_ij * var_ij.T)
    beta = np.array([hermite_coefficients(c) for c in cells])
    score_var = beta[:, 0]
    sparse = n_pairs < min_pair_count
    invalid = (~np.isfinite(sample_corr) | (var_ij <= 1e-12) | (var_ij.T <= 1e-12)
               | (score_var[:, None] <= 1e-12) | (score_var[None, :] <= 1e-12))
    corr = np.zeros((d, d))
    iu, ju = np.triu_indices(d, 1)
    ok = ~(sparse | invalid)[iu, ju]
    iu, ju = iu[ok], ju[ok]
    target = sample_corr[iu, ju] * np.sqrt(score_var[iu] * score_var[ju])
    corr[iu, ju] = invert_mehler(target, beta[iu], beta[ju])
    corr = np.clip(corr + corr.T, -MAX_ABS_CORRELATION, MAX_ABS_CORRELATION)
    block_ids = np.asarray(blocks)
    corr[block_ids[:, None] == block_ids[None, :]] = np.nan
    np.fill_diagonal(corr, 1.0)
    off_diag_sparse = int((np.triu(sparse, 1)).sum())
    return corr, n_pairs.astype(np.int64), off_diag_sparse


def _groups(labels: Sequence[LatentLabel]) -> list[list[int]]:
    blocks = nominal_blocks(list(labels))
    groups: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        groups.setdefault(b, []).append(i)
    return list(groups.values())


def observed_to_latent(
    observed: np.ndarray, labels: Sequence[LatentLabel],
    nominal_probabilities: dict[str, np.ndarray],
) -> np.ndarray:
    """Map the observed-feature matrix to the generation-space latent correlation matrix.

    Non-nominal pairs are copied. For a nominal column with level probabilities ``p`` the
    indicator covariances ``c_k = rho_obs * phi(Phi^-1(1 - p_k))`` satisfy ``c = B r`` for
    the score correlations ``r`` (exact for Gaussian partners), solved with the
    pseudo-inverse. Nominal x nominal blocks use the first-order analogue
    ``C = B_A^+ D (B_B^+)^T``. Blocks within one nominal column are the identity.
    """
    d = observed.shape[0]
    latent = np.eye(d)
    groups = _groups(labels)
    transforms: list[np.ndarray] = []
    for idx in groups:
        label = labels[idx[0]]
        if label.component == "level":
            p = nominal_probabilities[label.column]
            phi = np.zeros_like(p)
            ok = (p > 0) & (p < 1)
            phi[ok] = normal_pdf(special.ndtri(1 - p[ok]))
            transforms.append(np.linalg.pinv(nominal_b_matrix(p)) @ np.diag(phi))
        else:
            transforms.append(np.eye(1))
    for gi, idx_i in enumerate(groups):
        for gj in range(gi + 1, len(groups)):
            idx_j = groups[gj]
            block = observed[np.ix_(idx_i, idx_j)]
            mapped = transforms[gi] @ block @ transforms[gj].T
            latent[np.ix_(idx_i, idx_j)] = mapped
            latent[np.ix_(idx_j, idx_i)] = mapped.T
    return latent


def nominal_fixed_mask(labels: Sequence[LatentLabel]) -> np.ndarray:
    blocks = np.asarray(nominal_blocks(list(labels)))
    return blocks[:, None] == blocks[None, :]
