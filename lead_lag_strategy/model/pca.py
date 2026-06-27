"""
Subspace regularized PCA (SR-PCA).

Reference: Nakagawa et al. (2025) SIG-FIN-035, and the present paper.

Key formulas (paper Section 3):

Prior subspace V0 ∈ R^{N×K0}:
  v1 ∝ 1               (global factor)
  v2 ∝ (1_Nu, -1_Nj)  orthogonalised to v1  (country spread)
  v3 ∝ sign_cyc_def    orthogonalised to v1,v2  (cyclical-defensive)

Prior target matrix C0:
  D0     = diag(V0^T C_full V0)           (eq. 10)
  C0_raw = V0 D0 V0^T                     (eq. 11)
  C0     = Δ^{-1/2} C0_raw Δ^{-1/2},  Δ=diag(C0_raw), then set diag=1  (eq. 12)

Regularised correlation:
  C_reg_t = (1-λ) C_t + λ C0             (eq. 13)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_lag_strategy.config import (
    JP_CYCLICAL,
    JP_DEFENSIVE,
    JP_TICKERS,
    K,
    K0,
    LAMBDA,
    N_JP,
    N_US,
    US_CYCLICAL,
    US_DEFENSIVE,
    US_TICKERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prior subspace construction
# ─────────────────────────────────────────────────────────────────────────────

def _gram_schmidt_step(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Orthogonalise v against each vector in basis, then normalise."""
    for b in basis:
        v = v - np.dot(v, b) * b
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Degenerate prior vector (norm≈0 after orthogonalisation).")
    return v / norm


def build_prior_subspace(
    all_tickers: list[str] | None = None,
) -> np.ndarray:
    """
    Construct the K0=3 orthonormal prior eigenvector matrix V0 ∈ R^{N×K0}.

    Ordering: [US tickers ..., JP tickers ...]
    """
    if all_tickers is None:
        all_tickers = US_TICKERS + JP_TICKERS
    N = len(all_tickers)
    ticker_idx = {t: i for i, t in enumerate(all_tickers)}

    # v1: global factor – uniform weights
    v1 = np.ones(N) / np.sqrt(N)

    # v2: country spread – US positive, JP negative
    v2_raw = np.zeros(N)
    for t in US_TICKERS:
        v2_raw[ticker_idx[t]] = 1.0
    for t in JP_TICKERS:
        v2_raw[ticker_idx[t]] = -1.0
    v2 = _gram_schmidt_step(v2_raw, [v1])

    # v3: cyclical-defensive – cyclical positive, defensive negative
    v3_raw = np.zeros(N)
    for t in US_CYCLICAL + JP_CYCLICAL:
        if t in ticker_idx:
            v3_raw[ticker_idx[t]] = 1.0
    for t in US_DEFENSIVE + JP_DEFENSIVE:
        if t in ticker_idx:
            v3_raw[ticker_idx[t]] = -1.0
    v3 = _gram_schmidt_step(v3_raw, [v1, v2])

    V0 = np.column_stack([v1, v2, v3])   # (N, K0)
    return V0


def build_prior_target(
    c_full: np.ndarray,
    V0: np.ndarray,
) -> np.ndarray:
    """
    Compute the prior target correlation matrix C0.

    D0  = diag(V0^T C_full V0)      (eq. 10)
    C0_raw = V0 D0 V0^T             (eq. 11)
    C0  = normalised to correlation  (eq. 12)
    """
    D0 = np.diag(np.diag(V0.T @ c_full @ V0))          # (K0, K0)
    C0_raw = V0 @ D0 @ V0.T                             # (N, N)

    # Normalise to correlation matrix: C0_ij = C0_raw_ij / sqrt(C0_raw_ii * C0_raw_jj)
    diag_sqrt = np.sqrt(np.maximum(np.diag(C0_raw), 1e-12))
    C0 = C0_raw / np.outer(diag_sqrt, diag_sqrt)

    # Enforce diagonal = 1
    np.fill_diagonal(C0, 1.0)
    return C0


# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix utilities
# ─────────────────────────────────────────────────────────────────────────────

def correlation_matrix(z: np.ndarray) -> np.ndarray:
    """
    Compute the sample correlation matrix from a (T, N) standardised return
    matrix.  Any NaN rows are dropped before computation.
    """
    mask = ~np.any(np.isnan(z), axis=1)
    z_clean = z[mask]
    if z_clean.shape[0] < 2:
        raise ValueError("Too few clean rows to estimate correlation matrix.")
    return np.corrcoef(z_clean, rowvar=False)


def regularised_correlation(
    c_t: np.ndarray,
    c0: np.ndarray,
    lam: float = LAMBDA,
) -> np.ndarray:
    """C_reg = (1-λ) C_t + λ C0  (eq. 13)."""
    return (1.0 - lam) * c_t + lam * c0


# ─────────────────────────────────────────────────────────────────────────────
# Rolling standardised return
# ─────────────────────────────────────────────────────────────────────────────

def standardise(
    returns: np.ndarray,
    window_returns: np.ndarray,
) -> np.ndarray:
    """
    Standardise `returns` (1-D, length N) using mean/std estimated from
    `window_returns` (T×N) over the estimation window.

    z_i = (r_i - mu_i) / sigma_i   (eq. 9)
    """
    mu = np.nanmean(window_returns, axis=0)
    sigma = np.nanstd(window_returns, axis=0, ddof=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (returns - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Main SR-PCA class
# ─────────────────────────────────────────────────────────────────────────────

class SubspaceRegularisedPCA:
    """
    Implements the subspace-regularised PCA estimator for the lead-lag signal.

    Parameters
    ----------
    K      : number of principal components to retain
    lam    : shrinkage weight toward prior (λ in eq. 13)
    window : rolling estimation window length L
    """

    def __init__(
        self,
        K: int = K,
        lam: float = LAMBDA,
        window: int = 60,
    ):
        self.K = K
        self.lam = lam
        self.window = window

        self._V0: np.ndarray | None = None      # prior subspace (N, K0)
        self._C0: np.ndarray | None = None      # prior target correlation

    # ── Fit prior ──────────────────────────────────────────────────────────

    def fit_prior(
        self,
        cc_returns: pd.DataFrame,
        prior_start: str,
        prior_end: str,
    ) -> "SubspaceRegularisedPCA":
        """
        Estimate C_full and build C0 from the prior estimation window.

        cc_returns : DataFrame of close-to-close returns,
                     columns = US_TICKERS + JP_TICKERS
        """
        all_tickers = list(cc_returns.columns)
        self._V0 = build_prior_subspace(all_tickers)

        prior = cc_returns.loc[prior_start:prior_end].dropna(how="all")

        # Standardise within the full prior window
        mu    = prior.mean()
        sigma = prior.std(ddof=0).replace(0, np.nan).fillna(1.0)
        z_full = ((prior - mu) / sigma).values

        C_full = correlation_matrix(z_full)
        self._C0 = build_prior_target(C_full, self._V0)

        return self

    # ── Rolling signal ─────────────────────────────────────────────────────

    def compute_signal(
        self,
        cc_returns: pd.DataFrame,
        t: pd.Timestamp,
    ) -> np.ndarray | None:
        """
        Compute the lead-lag signal vector ẑ_{J,t+1} ∈ R^{N_J} at time t.

        Uses rolling window W_t = {t-L, ..., t-1} for estimation,
        and the current day's US return z_{U,t}.

        Returns None if there are insufficient data or prior not fitted.
        """
        if self._C0 is None or self._V0 is None:
            raise RuntimeError("Call fit_prior() before compute_signal().")

        loc = cc_returns.index.get_loc(t)
        if loc < self.window:
            return None

        # Window data: rows [loc-window, loc-1]  (eq. 8 – Wt = {t-L,...,t-1})
        window_data = cc_returns.iloc[loc - self.window: loc]

        # Current day data (US close-to-close at time t)
        current = cc_returns.iloc[loc]

        # Compute rolling mean/std over window
        mu    = window_data.mean()
        sigma = window_data.std(ddof=0).replace(0, np.nan).fillna(1.0)

        # Standardise window data
        z_window = ((window_data - mu) / sigma).values   # (L, N)

        # Correlation matrix C_t
        try:
            C_t = correlation_matrix(z_window)
        except ValueError:
            return None

        # Regularised correlation matrix (eq. 13)
        C_reg = regularised_correlation(C_t, self._C0, self.lam)

        # Eigendecomposition (eq. 14-15)
        eigvals, eigvecs = np.linalg.eigh(C_reg)
        # eigh returns ascending order; reverse to get descending
        idx = np.argsort(eigvals)[::-1]
        V_K = eigvecs[:, idx[:self.K]]   # (N, K), top-K eigenvectors

        # Split into US and JP blocks (eq. 16)
        V_U = V_K[:N_US, :]   # (N_US, K)
        V_J = V_K[N_US:, :]   # (N_JP, K)

        # Standardise current US returns (eq. 17)
        us_cols = list(cc_returns.columns[:N_US])
        jp_cols = list(cc_returns.columns[N_US:])

        mu_u    = mu[us_cols].values
        sigma_u = sigma[us_cols].values
        r_u     = current[us_cols].values

        if np.any(np.isnan(r_u)):
            return None

        z_U = (r_u - mu_u) / sigma_u    # (N_US,)

        # Factor score (eq. 18)
        f_t = V_U.T @ z_U               # (K,)

        # Lead-lag signal (eq. 19)
        z_hat_J = V_J @ f_t             # (N_JP,)

        return z_hat_J

    def compute_propagation_matrix(
        self,
        cc_returns: pd.DataFrame,
        t: pd.Timestamp,
    ) -> np.ndarray | None:
        """
        Return B_t^{(K)} = V_J V_U^T ∈ R^{N_JP × N_US}  (eq. 21).
        """
        if self._C0 is None or self._V0 is None:
            raise RuntimeError("Call fit_prior() before compute_propagation_matrix().")

        loc = cc_returns.index.get_loc(t)
        if loc < self.window:
            return None

        window_data = cc_returns.iloc[loc - self.window: loc]
        mu    = window_data.mean()
        sigma = window_data.std(ddof=0).replace(0, np.nan).fillna(1.0)
        z_window = ((window_data - mu) / sigma).values

        try:
            C_t = correlation_matrix(z_window)
        except ValueError:
            return None

        C_reg = regularised_correlation(C_t, self._C0, self.lam)
        eigvals, eigvecs = np.linalg.eigh(C_reg)
        idx = np.argsort(eigvals)[::-1]
        V_K = eigvecs[:, idx[:self.K]]
        V_U = V_K[:N_US, :]
        V_J = V_K[N_US:, :]

        return V_J @ V_U.T    # (N_JP, N_US)
