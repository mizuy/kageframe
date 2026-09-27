"""Latent Gaussian representations: interval normal scores, thresholds, Gumbel scores."""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

THRESHOLD_CLIP = 8.0


def interval_scores(probabilities: np.ndarray) -> tuple[np.ndarray, float]:
    """Normal scores for cells with the given probabilities (in latent order).

    Cell ``c`` covers ``(Phi^-1(a_c), Phi^-1(b_c)]`` with ``b = cumsum(p)``; its score is
    ``E[Z | cell] = (phi(Phi^-1(a)) - phi(Phi^-1(b))) / (b - a)``. Returns the scores and
    their standard deviation ``sqrt(sum p s^2)``, the attenuation factor for correlations.
    Cells with zero probability get score 0 (they never occur).
    """
    p = np.asarray(probabilities, dtype=float)
    b = np.clip(np.cumsum(p), 0.0, 1.0)
    b[-1] = 1.0
    a = np.concatenate([[0.0], b[:-1]])
    density = special.ndtri(np.concatenate([[0.0], b]))
    phi = np.exp(-0.5 * density**2) / np.sqrt(2 * np.pi)
    phi[~np.isfinite(density)] = 0.0
    width = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(width > 0, (phi[:-1] - phi[1:]) / width, 0.0)
    sd = float(np.sqrt(np.sum(width * scores**2)))
    return scores, sd


HERMITE_TERMS = 40


def hermite_coefficients(probabilities: np.ndarray, n_terms: int = HERMITE_TERMS
                         ) -> np.ndarray:
    """Mehler coefficients ``beta_n = E[s(Y) He_n(Z)] / sqrt(n!)`` of the interval scores.

    For two such variables with latent correlation rho,
    ``cov(s_1, s_2) = sum_n rho^n beta1_n beta2_n``; ``beta_1`` is the score variance.
    Uses ``int_a^b He_n phi = phi(a) He_{n-1}(a) - phi(b) He_{n-1}(b)``.
    """
    p = np.asarray(probabilities, dtype=float)
    scores, _ = interval_scores(p)
    b = np.clip(np.cumsum(p), 0.0, 1.0)
    b[-1] = 1.0
    bounds = np.clip(special.ndtri(np.concatenate([[0.0], b])), -40.0, 40.0)
    phi = normal_pdf(bounds)
    phi[0] = phi[-1] = 0.0
    beta = np.zeros(n_terms)
    h_prev, h = np.zeros_like(bounds), np.ones_like(bounds)  # normalized He_{n-1}/sqrt((n-1)!)
    for n in range(1, n_terms + 1):
        g = phi * h
        beta[n - 1] = np.sum(scores * (g[:-1] - g[1:])) / np.sqrt(n)
        h_prev, h = h, (bounds * h - np.sqrt(n - 1) * h_prev) / np.sqrt(n)
    return beta


def invert_mehler(target_cov: np.ndarray, beta_i: np.ndarray, beta_j: np.ndarray,
                  iterations: int = 50) -> np.ndarray:
    """Solve ``sum_n rho^n beta_i[n] beta_j[n] = target`` for rho in [-1, 1] (bisection).

    Shapes: ``target_cov`` (m,), ``beta_i``/``beta_j`` (m, N). The left side is increasing
    in rho because interval scores are monotone in the latent variable.
    """
    coef = beta_i * beta_j
    powers = np.arange(1, coef.shape[1] + 1)

    def f(rho: np.ndarray) -> np.ndarray:
        return np.sum(coef * rho[:, None] ** powers, axis=1)

    lo = np.full(len(target_cov), -1.0)
    hi = np.full(len(target_cov), 1.0)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        above = f(mid) > target_cov
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return (lo + hi) / 2


def thresholds(probabilities: np.ndarray) -> np.ndarray:
    """Latent thresholds ``Phi^-1(cumsum(p))`` between consecutive cells (K-1 values)."""
    cum = np.cumsum(np.asarray(probabilities, dtype=float))[:-1]
    return np.clip(special.ndtri(np.clip(cum, 0.0, 1.0)), -THRESHOLD_CLIP, THRESHOLD_CLIP)


def indicator_sd(p: np.ndarray) -> np.ndarray:
    """Score SD of a binary indicator with success probability ``p``: phi(tau)/sqrt(p(1-p))."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    ok = (p > 0) & (p < 1)
    out[ok] = normal_pdf(special.ndtri(1 - p[ok])) / np.sqrt(p[ok] * (1 - p[ok]))
    return out


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / np.sqrt(2 * np.pi)


def gumbel_scores(z: np.ndarray) -> np.ndarray:
    """Map standard normal scores to standard Gumbel scores: -log(-log Phi(z))."""
    neg_log_cdf = np.maximum(-special.log_ndtr(z), 1e-300)
    return -np.log(neg_log_cdf)


def nominal_choice(z: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """argmax_k(log p_k + Gumbel(z_k)); with independent scores P(choice = k) = p_k exactly."""
    p = np.asarray(probabilities, dtype=float)
    with np.errstate(divide="ignore"):
        log_p = np.where(p > 0, np.log(p), -np.inf)
    return np.argmax(log_p + gumbel_scores(z), axis=1)


_GRID = np.linspace(-12.0, 12.0, 24001)


def nominal_b_matrix(probabilities: np.ndarray) -> np.ndarray:
    """``B[k, j] = E[1{Y=k} Z_j]`` for the Gumbel-max nominal model with independent scores.

    ``P(Y=j | Z_j=z) = Phi(z)^((1-p_j)/p_j)`` gives ``B[j, j]`` as a 1-D integral, and
    ``B[k, j] = -p_k / (1 - p_j) * B[j, j]`` for ``k != j`` (columns sum to zero).
    """
    p = np.asarray(probabilities, dtype=float)
    k = len(p)
    z = _GRID
    log_cdf = special.log_ndtr(z)
    weight = z * normal_pdf(z)
    diag = np.zeros(k)
    for j in range(k):
        if 0 < p[j] < 1:
            integrand = weight * np.exp((1 - p[j]) / p[j] * log_cdf)
            diag[j] = integrate.trapezoid(integrand, z)
    b = np.zeros((k, k))
    for j in range(k):
        if diag[j] == 0:
            continue
        b[:, j] = -p / (1 - p[j]) * diag[j]
        b[j, j] = diag[j]
    return b
