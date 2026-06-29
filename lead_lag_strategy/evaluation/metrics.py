"""
Performance metrics (paper Section 4.2) and Fama-French factor regression.

Metrics:
  AR   – annualised return     (eq. 27)
  RISK – annualised volatility (eq. 28)
  R/R  – Sharpe-like ratio     (eq. 29)
  MDD  – maximum drawdown      (eq. 30)

Factor models:
  Fama-French 3-factor (Table 3)
  Carhart 4-factor     (Table 4)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from lead_lag_strategy.config import TRADING_DAYS_PER_YEAR

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

try:
    import pandas_datareader.data as pdr
    _HAS_PDR = True
except ImportError:
    _HAS_PDR = False


# ─────────────────────────────────────────────────────────────────────────────
# Core performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def annualised_return(returns: pd.Series) -> float:
    """AR = (252/T) * Σ R_t  (eq. 27, daily returns assumed)."""
    T = len(returns)
    if T == 0:
        return np.nan
    return TRADING_DAYS_PER_YEAR / T * returns.sum()


def annualised_risk(returns: pd.Series) -> float:
    """RISK = sqrt(252 / (T-1) * Σ(R_t - μ)^2)  (eq. 28)."""
    T = len(returns)
    if T < 2:
        return np.nan
    mu = returns.mean()
    return np.sqrt(TRADING_DAYS_PER_YEAR / (T - 1) * ((returns - mu) ** 2).sum())


def return_to_risk(ar: float, risk: float) -> float:
    """R/R = AR / RISK  (eq. 29)."""
    if risk == 0 or np.isnan(risk):
        return np.nan
    return ar / risk


def maximum_drawdown(returns: pd.Series) -> float:
    """
    MDD = min_t { W_t / max_{τ≤t} W_τ - 1 }   (eq. 30).
    Returns a negative number (or 0 if no drawdown).
    """
    wealth = (1 + returns).cumprod()
    rolling_max = wealth.cummax()
    drawdown = wealth / rolling_max - 1
    return float(drawdown.min())


def performance_summary(returns: pd.Series) -> dict[str, float]:
    """Compute AR, RISK, R/R, MDD for a return series."""
    ar   = annualised_return(returns)
    risk = annualised_risk(returns)
    rr   = return_to_risk(ar, risk)
    mdd  = maximum_drawdown(returns)
    return {"AR": ar * 100, "RISK": risk * 100, "R/R": rr, "MDD": abs(mdd) * 100}


def summary_table(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """Build a summary statistics table (Table 2 in the paper)."""
    rows = {}
    for name in strategy_returns.columns:
        ret = strategy_returns[name].dropna()
        rows[name] = performance_summary(ret)
    df = pd.DataFrame(rows).T[["AR", "RISK", "R/R", "MDD"]]
    return df.round(2)


# ─────────────────────────────────────────────────────────────────────────────
# Fama-French factor data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ff_factors(
    start: str,
    end: str,
    model: str = "FF3",
) -> Optional[pd.DataFrame]:
    """
    Download Fama-French factors from Ken French's data library.

    model : 'FF3' (Fama-French 3-factor) or 'C4' (Carhart 4-factor)
    Returns daily factor DataFrame or None if unavailable.
    """
    if not _HAS_PDR:
        warnings.warn(
            "pandas-datareader not installed. Factor regression skipped. "
            "Install with: pip install pandas-datareader"
        )
        return None

    try:
        if model in ("FF3", "C4"):
            ff3 = pdr.DataReader(
                "F-F_Research_Data_Factors_daily",
                "famafrench",
                start=start,
                end=end,
            )[0] / 100
            ff3.index = pd.to_datetime(ff3.index, format="%Y%m%d")
        if model == "C4":
            mom = pdr.DataReader(
                "F-F_Momentum_Factor_daily",
                "famafrench",
                start=start,
                end=end,
            )[0] / 100
            mom.index = pd.to_datetime(mom.index, format="%Y%m%d")
            mom.columns = ["WML"]
            return ff3.join(mom, how="inner").loc[start:end]
        return ff3.loc[start:end]
    except Exception as exc:
        warnings.warn(f"Failed to fetch FF factors: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Factor regression
# ─────────────────────────────────────────────────────────────────────────────

def factor_regression(
    strategy_return: pd.Series,
    factors: pd.DataFrame,
) -> dict:
    """
    OLS regression of daily strategy return on risk factors.

    strategy_return : daily returns (net of risk-free rate optionally)
    factors         : DataFrame with columns like ['Mkt-RF', 'SMB', 'HML', 'WML']
                      and optional 'RF' column

    Returns dict with alpha (annualised %), betas, t-stats, adj_R2.
    Newey-West HAC standard errors (8 lags, following convention).
    """
    if not _HAS_SM:
        warnings.warn(
            "statsmodels not installed. Factor regression skipped. "
            "Install with: pip install statsmodels"
        )
        return {}

    # Align dates
    common = strategy_return.index.intersection(factors.index)
    y = strategy_return.loc[common]
    X = factors.loc[common]

    # Excess return (subtract RF if available)
    if "RF" in X.columns:
        y = y - X["RF"]
        X = X.drop(columns=["RF"])

    # Remove 'Mkt-RF' -> 'MKT' for cleaner display
    X = X.rename(columns={"Mkt-RF": "MKT"})
    factor_names = list(X.columns)

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 8},
    )

    alpha_daily = model.params["const"]
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR * 100

    result = {
        "alpha_pct_yr": round(alpha_annual, 2),
        "alpha_tstat": round(model.tvalues["const"], 2),
        "adj_R2": round(model.rsquared_adj, 3),
    }
    for fn in factor_names:
        result[fn] = round(model.params[fn], 4)
        result[f"{fn}_t"] = round(model.tvalues[fn], 2)

    return result


def factor_regression_table(
    strategy_returns: pd.DataFrame,
    model: str = "FF3",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run factor regression for all strategies; return summary table.

    model : 'FF3' or 'C4'
    """
    s = start or str(strategy_returns.index[0].date())
    e = end   or str(strategy_returns.index[-1].date())

    factors = fetch_ff_factors(s, e, model=model)
    if factors is None:
        return pd.DataFrame()

    rows = {}
    for name in strategy_returns.columns:
        ret = strategy_returns[name].dropna()
        rows[name] = factor_regression(ret, factors)

    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative wealth
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_wealth(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound returns into a cumulative wealth series starting at `initial`."""
    return initial * (1 + returns).cumprod()
