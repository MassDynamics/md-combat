"""
Private helper functions shared by ComBat and ComBatSeq.

Translated from sva/R/helper.R (GPL-3).
"""

import numpy as np


def _aprior(gamma_hat: np.ndarray) -> float:
    """Empirical hyper-prior shape parameter for inverse-gamma prior on delta."""
    m = np.mean(gamma_hat)
    s2 = np.var(gamma_hat, ddof=1)
    if s2 == 0:
        raise ValueError(
            "_aprior received delta_hat with zero variance (all values identical). "
            "This should have been prevented by input validation. "
            "Please report this as a bug with your input data."
        )
    return (2 * s2 + m**2) / s2


def _bprior(gamma_hat: np.ndarray) -> float:
    """Empirical hyper-prior scale parameter for inverse-gamma prior on delta."""
    m = np.mean(gamma_hat)
    s2 = np.var(gamma_hat, ddof=1)
    if s2 == 0:
        raise ValueError(
            "_bprior received delta_hat with zero variance (all values identical). "
            "This should have been prevented by input validation. "
            "Please report this as a bug with your input data."
        )
    return (m * s2 + m**3) / s2


def _postmean(
    g_hat: np.ndarray,
    g_bar: float,
    n: np.ndarray,
    d_star: np.ndarray,
    t2: float,
) -> np.ndarray:
    """Posterior mean of gamma (additive batch effect)."""
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2: np.ndarray, n: np.ndarray, a: float, b: float) -> np.ndarray:
    """Posterior variance of delta (multiplicative batch effect)."""
    return (0.5 * sum2 + b) / (n / 2 + a - 1)


def _it_sol(
    sdat: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 1e-4,
) -> np.ndarray:
    """
    EM algorithm to find parametric EB adjustments for one batch.

    Returns a (2, n_genes) array: row 0 = gamma.star, row 1 = delta.star.
    Translated from sva/R/helper.R::it.sol
    """
    n = np.sum(~np.isnan(sdat), axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1.0
    while change > conv:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        resid = sdat - g_new[:, np.newaxis]
        sum2 = np.nansum(resid**2, axis=1)
        d_new = _postvar(sum2, n, a, b)

        change = np.max(np.abs(g_new - g_old) / np.abs(g_old + 1e-10))
        change = max(change, np.max(np.abs(d_new - d_old) / np.abs(d_old + 1e-10)))

        g_old = g_new
        d_old = d_new

    return np.vstack([g_new, d_new])
