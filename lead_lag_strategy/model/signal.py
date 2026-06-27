"""
Signal construction for all strategy variants (paper Section 4.3).

Strategies:
  MOM       – simple momentum (eq. 31)
  PCA_PLAIN – PCA without regularisation (λ=0)
  PCA_SUB   – subspace-regularised PCA  (proposed)
  DOUBLE    – double sort of MOM × PCA_SUB
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from lead_lag_strategy.config import (
    BACKTEST_END,
    BACKTEST_START,
    JP_TICKERS,
    K,
    L,
    N_JP,
    N_US,
    PRIOR_END,
    PRIOR_START,
    US_TICKERS,
)
from lead_lag_strategy.model.pca import (
    SubspaceRegularisedPCA,
    correlation_matrix,
    regularised_correlation,
    standardise,
)

StrategyName = Literal["MOM", "PCA_PLAIN", "PCA_SUB", "DOUBLE"]


# ─────────────────────────────────────────────────────────────────────────────
# Individual signal generators
# ─────────────────────────────────────────────────────────────────────────────

def _momentum_signal(cc_jp: pd.DataFrame, loc: int, window: int) -> np.ndarray | None:
    """
    Simple momentum signal (eq. 31):
      m_{j,t} = mean_{τ∈Wt} r^cc_{j,τ}
    """
    if loc < window:
        return None
    window_data = cc_jp.iloc[loc - window: loc]
    signal = window_data.mean().values
    if np.all(np.isnan(signal)):
        return None
    return signal


def _pca_plain_signal(
    cc_all: pd.DataFrame,
    loc: int,
    window: int,
    K: int,
) -> np.ndarray | None:
    """PCA without regularisation (λ=0): directly eigendecompose C_t."""
    if loc < window:
        return None

    window_data = cc_all.iloc[loc - window: loc]
    mu    = window_data.mean()
    sigma = window_data.std(ddof=0).replace(0, np.nan).fillna(1.0)
    z_window = ((window_data - mu) / sigma).values

    try:
        C_t = correlation_matrix(z_window)
    except ValueError:
        return None

    current = cc_all.iloc[loc]
    r_u = current[list(cc_all.columns[:N_US])].values
    if np.any(np.isnan(r_u)):
        return None

    eigvals, eigvecs = np.linalg.eigh(C_t)
    idx = np.argsort(eigvals)[::-1]
    V_K = eigvecs[:, idx[:K]]
    V_U = V_K[:N_US, :]
    V_J = V_K[N_US:, :]

    mu_u    = mu.values[:N_US]
    sigma_u = sigma.values[:N_US]
    z_U = (r_u - mu_u) / sigma_u
    f_t = V_U.T @ z_U
    return V_J @ f_t


# ─────────────────────────────────────────────────────────────────────────────
# Rolling backtest signal computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_signals(
    cc_all: pd.DataFrame,          # combined US+JP close-to-close, all dates
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
    prior_start: str = PRIOR_START,
    prior_end: str = PRIOR_END,
    window: int = L,
    K_components: int = K,
) -> dict[str, pd.DataFrame]:
    """
    Compute daily signals for all strategies over the backtest period.

    Returns dict mapping strategy name → DataFrame of JP signals,
    indexed by the date on which the signal is generated (t),
    applicable to JP OC returns on day t+1.

    Columns = JP_TICKERS
    """
    # Fit prior for PCA_SUB
    model_sub = SubspaceRegularisedPCA(K=K_components, lam=0.9, window=window)
    model_sub.fit_prior(cc_all, prior_start=prior_start, prior_end=prior_end)

    jp_cols = list(cc_all.columns[N_US:])   # JP ticker columns in cc_all
    us_cols = list(cc_all.columns[:N_US])

    dates = cc_all.loc[backtest_start:backtest_end].index

    records: dict[str, list] = {
        "MOM": [], "PCA_PLAIN": [], "PCA_SUB": [], "PCA_SUB_raw": [],
    }

    for t in dates:
        loc = cc_all.index.get_loc(t)

        # --- MOM ---
        cc_jp = cc_all[jp_cols]
        mom_sig = _momentum_signal(cc_jp, loc, window)
        records["MOM"].append(
            pd.Series(mom_sig, index=jp_cols, name=t) if mom_sig is not None
            else pd.Series(np.nan, index=jp_cols, name=t)
        )

        # --- PCA PLAIN ---
        plain_sig = _pca_plain_signal(cc_all, loc, window, K_components)
        records["PCA_PLAIN"].append(
            pd.Series(plain_sig, index=jp_cols, name=t) if plain_sig is not None
            else pd.Series(np.nan, index=jp_cols, name=t)
        )

        # --- PCA SUB ---
        sub_sig = model_sub.compute_signal(cc_all, t)
        records["PCA_SUB"].append(
            pd.Series(sub_sig, index=jp_cols, name=t) if sub_sig is not None
            else pd.Series(np.nan, index=jp_cols, name=t)
        )

    signals = {
        name: pd.DataFrame(records[name]) for name in ["MOM", "PCA_PLAIN", "PCA_SUB"]
    }

    # --- DOUBLE: median split on MOM and PCA_SUB ---
    signals["DOUBLE"] = _build_double_signal(signals["MOM"], signals["PCA_SUB"])

    return signals


def _build_double_signal(
    mom: pd.DataFrame,
    pca_sub: pd.DataFrame,
) -> pd.DataFrame:
    """
    DOUBLE strategy: 2×2 sort on MOM and PCA_SUB.
    High×High → signal=+1, Low×Low → signal=-1, else 0.
    """
    mom_high = mom.ge(mom.median(axis=1), axis=0)
    pca_high = pca_sub.ge(pca_sub.median(axis=1), axis=0)

    double = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
    double[mom_high & pca_high] = 1.0
    double[~mom_high & ~pca_high] = -1.0
    return double
