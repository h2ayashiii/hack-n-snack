"""
Long-short portfolio construction (paper Section 2.2).

Equations:
  L_{t+1} = Top-q set of {s_{j,t}}          (eq. 3)
  S_{t+1} = Bottom-q set of {s_{j,t}}       (eq. 4)
  w_{j,t+1} = +1/|L| if j∈L, -1/|S| if j∈S, 0 otherwise  (eq. 5)
  R_{t+1} = Σ_j w_{j,t+1} r^oc_{j,t+1}     (eq. 7)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_lag_strategy.config import JP_TICKERS, Q


def build_weights(
    signal: pd.Series,
    q: float = Q,
) -> pd.Series:
    """
    Compute equal-weight long-short weights for one date given signal values.

    Parameters
    ----------
    signal : JP-ticker signals (can contain NaN which are excluded)
    q      : quantile threshold (0 < q < 0.5)

    Returns
    -------
    weights : pd.Series indexed by JP_TICKERS, summing to 0, |w|=2
    """
    valid = signal.dropna()
    if len(valid) == 0:
        return pd.Series(0.0, index=signal.index)

    n = len(valid)
    n_long  = max(1, int(np.floor(n * q)))
    n_short = max(1, int(np.floor(n * q)))

    sorted_vals = valid.sort_values(ascending=False)
    long_set    = set(sorted_vals.index[:n_long])
    short_set   = set(sorted_vals.index[-n_short:])

    weights = pd.Series(0.0, index=signal.index)
    for j in long_set:
        weights[j] = +1.0 / len(long_set)
    for j in short_set:
        weights[j] = -1.0 / len(short_set)

    return weights


def build_all_weights(
    signals: pd.DataFrame,
    q: float = Q,
) -> pd.DataFrame:
    """Apply build_weights to each row of a signal DataFrame."""
    return signals.apply(lambda row: build_weights(row, q), axis=1)


def compute_strategy_returns(
    signals: pd.DataFrame,
    jp_oc: pd.DataFrame,
    q: float = Q,
) -> pd.Series:
    """
    Compute daily strategy returns R_{t+1} using signal from day t
    and JP OC return on day t+1.

    Parameters
    ----------
    signals : (T, N_JP) signal DataFrame indexed by date t
    jp_oc   : (T, N_JP) JP open-to-close returns
    q       : quantile

    Returns
    -------
    strategy_returns : pd.Series indexed by dates t+1
    """
    returns_list = []
    dates = []

    all_dates = signals.index
    jp_dates  = jp_oc.index

    for i, t in enumerate(all_dates):
        signal_t = signals.iloc[i]
        w_t = build_weights(signal_t, q)

        # Find next trading date in JP
        future_jp = jp_dates[jp_dates > t]
        if len(future_jp) == 0:
            continue
        t1 = future_jp[0]

        if t1 not in jp_oc.index:
            continue

        r_jp = jp_oc.loc[t1]
        R = (w_t * r_jp).sum()
        returns_list.append(R)
        dates.append(t1)

    return pd.Series(returns_list, index=pd.DatetimeIndex(dates), name="strategy")


def compute_all_strategy_returns(
    all_signals: dict[str, pd.DataFrame],
    jp_oc: pd.DataFrame,
    q: float = Q,
) -> pd.DataFrame:
    """Compute strategy returns for all signal variants."""
    result = {}
    for name, sig in all_signals.items():
        result[name] = compute_strategy_returns(sig, jp_oc, q)

    return pd.DataFrame(result)
