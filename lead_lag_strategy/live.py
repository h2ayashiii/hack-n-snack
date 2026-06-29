"""
Live signal generation for the Japan-US sector lead-lag strategy.

Computes today's trading signal using the latest available US sector ETF data
and outputs the JP sector long/short positions for the next Japan trading day.

Usage:
    python -m lead_lag_strategy.live
    python -m lead_lag_strategy.live --date 2025-06-26
    python -m lead_lag_strategy.live --strategy PCA_SUB --json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from lead_lag_strategy.config import (
    JP_SECTOR_NAMES,
    JP_TICKERS,
    K,
    L,
    PRIOR_END,
    PRIOR_START,
    Q,
    US_SECTOR_NAMES,
    US_TICKERS,
)
from lead_lag_strategy.data.fetcher import build_returns, download_ohlcv
from lead_lag_strategy.model.pca import SubspaceRegularisedPCA
from lead_lag_strategy.model.portfolio import build_weights


# ─────────────────────────────────────────────────────────────────────────────
# Live signal computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_live_signal(
    signal_date: str | None = None,
    strategy: str = "PCA_SUB",
    prior_start: str = PRIOR_START,
    prior_end: str = PRIOR_END,
    window: int = L,
    K_components: int = K,
    q: float = Q,
    force_refresh: bool = False,
) -> dict:
    """
    Generate today's lead-lag signal.

    Parameters
    ----------
    signal_date : ISO date string (default: today / most recent US trading day)
    strategy    : 'PCA_SUB' (recommended) or 'PCA_PLAIN' or 'MOM'

    Returns
    -------
    dict with keys:
      signal_date, strategy, signal_vector, weights, long_positions, short_positions
    """
    # ── Download latest data ───────────────────────────────────────────────
    end_download = signal_date or str(date.today())
    print(f"[live] Downloading data up to {end_download} …")
    ohlcv = download_ohlcv(
        start="2009-12-01",
        end=end_download,
        force_refresh=force_refresh,
    )
    returns = build_returns(ohlcv, start="2010-01-01", end=end_download)

    us_cc = returns["us_cc"]
    jp_cc = returns["jp_cc"]
    jp_oc = returns["jp_oc"]

    cc_all = pd.concat([us_cc, jp_cc], axis=1)
    cc_all.columns = US_TICKERS + JP_TICKERS

    # ── Select signal date ─────────────────────────────────────────────────
    if signal_date is None:
        # Use the most recent date with complete US data
        valid_dates = us_cc.dropna(how="all").index
        if len(valid_dates) == 0:
            raise RuntimeError("No valid US data found.")
        t = valid_dates[-1]
    else:
        t = pd.Timestamp(signal_date)
        if t not in cc_all.index:
            # Try the most recent prior date
            prior = cc_all.index[cc_all.index <= t]
            if len(prior) == 0:
                raise RuntimeError(f"No data available on or before {signal_date}")
            t = prior[-1]
            warnings.warn(f"No data on {signal_date}, using {t.date()} instead.")

    print(f"[live] Signal date: {t.date()}")

    # ── Compute signal ────────────────────────────────────────────────────
    signal_vec: np.ndarray | None = None

    if strategy == "MOM":
        loc = cc_all.index.get_loc(t)
        if loc < window:
            raise RuntimeError("Insufficient data for momentum signal.")
        cc_jp = cc_all[JP_TICKERS]
        window_data = cc_jp.iloc[loc - window: loc]
        signal_vec = window_data.mean().values

    elif strategy == "PCA_PLAIN":
        from lead_lag_strategy.model.signal import _pca_plain_signal
        loc = cc_all.index.get_loc(t)
        signal_vec = _pca_plain_signal(cc_all, loc, window, K_components)
        if signal_vec is None:
            raise RuntimeError("Insufficient data for PCA_PLAIN signal.")

    elif strategy == "PCA_SUB":
        model = SubspaceRegularisedPCA(K=K_components, lam=0.9, window=window)
        model.fit_prior(cc_all, prior_start=prior_start, prior_end=prior_end)
        signal_vec = model.compute_signal(cc_all, t)
        if signal_vec is None:
            raise RuntimeError("Insufficient data for PCA_SUB signal.")
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: MOM, PCA_PLAIN, PCA_SUB")

    # ── Build portfolio weights ───────────────────────────────────────────
    signal_series = pd.Series(signal_vec, index=JP_TICKERS)
    weights = build_weights(signal_series, q=q)

    long_set  = weights[weights > 0].index.tolist()
    short_set = weights[weights < 0].index.tolist()

    result = {
        "signal_date": str(t.date()),
        "strategy": strategy,
        "signal_vector": {
            ticker: round(float(v), 6)
            for ticker, v in zip(JP_TICKERS, signal_vec)
        },
        "weights": {
            ticker: round(float(w), 6)
            for ticker, w in weights.items()
            if abs(w) > 1e-9
        },
        "long_positions": [
            {"ticker": t, "name": JP_SECTOR_NAMES.get(t, t), "weight": round(float(weights[t]), 4)}
            for t in sorted(long_set)
        ],
        "short_positions": [
            {"ticker": t, "name": JP_SECTOR_NAMES.get(t, t), "weight": round(float(weights[t]), 4)}
            for t in sorted(short_set)
        ],
    }

    return result


def _print_signal(result: dict, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n{'='*60}")
    print(f"  Lead-Lag Signal  |  Strategy: {result['strategy']}")
    print(f"  Signal Date: {result['signal_date']}")
    print(f"  (Trade JP open-to-close on the NEXT trading day)")
    print(f"{'='*60}")

    print(f"\n  LONG  (+{100/max(1,len(result['long_positions'])):.1f}% each):")
    for pos in result["long_positions"]:
        print(f"    {pos['ticker']:10s}  {pos['name']}")

    print(f"\n  SHORT (-{100/max(1,len(result['short_positions'])):.1f}% each):")
    for pos in result["short_positions"]:
        print(f"    {pos['ticker']:10s}  {pos['name']}")

    print(f"\n  Full signal vector (z_hat_J):")
    sig_items = sorted(result["signal_vector"].items(), key=lambda x: -x[1])
    for ticker, val in sig_items:
        name = JP_SECTOR_NAMES.get(ticker, ticker)
        bar = "█" * int(abs(val) * 20) if abs(val) < 5 else "█" * 20
        sign = "+" if val >= 0 else "-"
        print(f"    {ticker:10s}  {sign}{abs(val):.4f}  {bar}  {name}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Propagation matrix visualisation
# ─────────────────────────────────────────────────────────────────────────────

def show_propagation_matrix(
    signal_date: str | None = None,
    prior_start: str = PRIOR_START,
    prior_end: str = PRIOR_END,
    window: int = L,
    K_components: int = K,
    save_path: str | None = None,
) -> None:
    """Print/plot the propagation matrix B_t = V_J V_U^T at a given date."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn required. Install with: pip install matplotlib seaborn")
        return

    end_download = signal_date or str(date.today())
    ohlcv = download_ohlcv(start="2009-12-01", end=end_download)
    returns = build_returns(ohlcv, start="2010-01-01", end=end_download)
    us_cc = returns["us_cc"]
    jp_cc = returns["jp_cc"]
    cc_all = pd.concat([us_cc, jp_cc], axis=1)
    cc_all.columns = US_TICKERS + JP_TICKERS

    t = pd.Timestamp(signal_date) if signal_date else cc_all.index[-1]

    model = SubspaceRegularisedPCA(K=K_components, lam=0.9, window=window)
    model.fit_prior(cc_all, prior_start=prior_start, prior_end=prior_end)
    B = model.compute_propagation_matrix(cc_all, t)

    if B is None:
        print("Insufficient data for propagation matrix.")
        return

    B_df = pd.DataFrame(
        B,
        index=[JP_SECTOR_NAMES.get(t, t) for t in JP_TICKERS],
        columns=[US_SECTOR_NAMES.get(t, t) for t in US_TICKERS],
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    vmax = np.abs(B).max()
    im = ax.imshow(B, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Propagation weight")
    ax.set_xticks(range(len(US_TICKERS)))
    ax.set_xticklabels([US_SECTOR_NAMES.get(t, t) for t in US_TICKERS], rotation=45, ha="right")
    ax.set_yticks(range(len(JP_TICKERS)))
    ax.set_yticklabels([JP_SECTOR_NAMES.get(t, t) for t in JP_TICKERS])
    ax.set_title(f"Propagation Matrix B_t  ({t.date()})\nUS → JP sector transmission")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate live lead-lag trading signal."
    )
    p.add_argument("--date",         default=None,       help="Signal date (YYYY-MM-DD); default=today")
    p.add_argument("--strategy",     default="PCA_SUB",  choices=["PCA_SUB", "PCA_PLAIN", "MOM"],
                   help="Signal generation strategy")
    p.add_argument("--K",            type=int, default=K, help="Number of PCA components")
    p.add_argument("--window",       type=int, default=L, help="Rolling window (days)")
    p.add_argument("--q",            type=float, default=Q, help="Quantile for long-short")
    p.add_argument("--json",         action="store_true", help="Output as JSON")
    p.add_argument("--propagation",  action="store_true", help="Also plot propagation matrix")
    p.add_argument("--force-refresh",action="store_true", help="Re-download data")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    result = compute_live_signal(
        signal_date=args.date,
        strategy=args.strategy,
        window=args.window,
        K_components=args.K,
        q=args.q,
        force_refresh=args.force_refresh,
    )
    _print_signal(result, as_json=args.json)

    if args.propagation:
        show_propagation_matrix(
            signal_date=args.date,
            window=args.window,
            K_components=args.K,
            save_path="results/propagation_matrix.png",
        )
