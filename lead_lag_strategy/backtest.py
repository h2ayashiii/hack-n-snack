"""
Backtest / verification script for the Japan-US sector lead-lag strategy.

Reproduces Table 2, Table 3, Table 4, and Figure 2 from the paper.

Usage:
    python -m lead_lag_strategy.backtest
    python -m lead_lag_strategy.backtest --start 2015-01-01 --end 2025-12-31
    python -m lead_lag_strategy.backtest --no-plot
"""

from __future__ import annotations

import argparse
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # headless default; override by calling show() after

from lead_lag_strategy.config import (
    BACKTEST_END,
    BACKTEST_START,
    JP_TICKERS,
    PRIOR_END,
    PRIOR_START,
    US_TICKERS,
    K,
    L,
    Q,
)
from lead_lag_strategy.data.fetcher import build_returns, download_ohlcv
from lead_lag_strategy.evaluation.metrics import (
    cumulative_wealth,
    factor_regression_table,
    summary_table,
)
from lead_lag_strategy.model.portfolio import compute_all_strategy_returns
from lead_lag_strategy.model.signal import compute_all_signals


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest function
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
    prior_start: str = PRIOR_START,
    prior_end: str = PRIOR_END,
    window: int = L,
    K_components: int = K,
    q: float = Q,
    plot: bool = True,
    save_dir: str = "results",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run full backtest.

    Returns
    -------
    strategy_returns : DataFrame of daily returns per strategy
    all_signals      : dict of signal DataFrames per strategy
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Download data ───────────────────────────────────────────────────
    print("[1/5] Downloading price data …")
    ohlcv = download_ohlcv(
        start="2009-12-01",
        end=backtest_end,
        force_refresh=force_refresh,
    )
    returns = build_returns(ohlcv, start="2010-01-01", end=backtest_end)

    us_cc  = returns["us_cc"]
    jp_cc  = returns["jp_cc"]
    jp_oc  = returns["jp_oc"]

    # Combined close-to-close (US + JP), ordered US first then JP
    cc_all = pd.concat([us_cc, jp_cc], axis=1)
    cc_all.columns = US_TICKERS + JP_TICKERS

    print(f"   US  : {us_cc.shape[0]} dates, {us_cc.shape[1]} sectors")
    print(f"   JP  : {jp_cc.shape[0]} dates, {jp_cc.shape[1]} sectors")

    # ── 2. Compute signals ─────────────────────────────────────────────────
    print("[2/5] Computing signals …")
    all_signals = compute_all_signals(
        cc_all=cc_all,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        prior_start=prior_start,
        prior_end=prior_end,
        window=window,
        K_components=K_components,
    )
    print(f"   Signal dates: {next(iter(all_signals.values())).index[0].date()} "
          f"→ {next(iter(all_signals.values())).index[-1].date()}")

    # ── 3. Portfolio returns ───────────────────────────────────────────────
    print("[3/5] Building portfolios …")
    strategy_returns = compute_all_strategy_returns(all_signals, jp_oc, q=q)
    strategy_returns = strategy_returns.dropna(how="all")
    print(f"   Return dates: {strategy_returns.index[0].date()} "
          f"→ {strategy_returns.index[-1].date()}")

    # ── 4. Performance summary ─────────────────────────────────────────────
    print("\n[4/5] Performance summary (Table 2)")
    perf = summary_table(strategy_returns)
    print(perf.to_string())
    perf.to_csv(f"{save_dir}/performance_summary.csv")

    print("\n   Fama-French 3-factor regression (Table 3)")
    ff3_table = factor_regression_table(strategy_returns, model="FF3")
    if not ff3_table.empty:
        print(ff3_table[["alpha_pct_yr", "alpha_tstat", "MKT", "SMB", "HML", "adj_R2"]].to_string())
        ff3_table.to_csv(f"{save_dir}/ff3_regression.csv")
    else:
        print("   (skipped – pandas-datareader / statsmodels not available)")

    print("\n   Carhart 4-factor regression (Table 4)")
    c4_table = factor_regression_table(strategy_returns, model="C4")
    if not c4_table.empty:
        print(c4_table[["alpha_pct_yr", "alpha_tstat", "MKT", "SMB", "HML", "WML", "adj_R2"]].to_string())
        c4_table.to_csv(f"{save_dir}/carhart4_regression.csv")
    else:
        print("   (skipped – pandas-datareader / statsmodels not available)")

    # ── 5. Plot cumulative returns ─────────────────────────────────────────
    if plot:
        print("\n[5/5] Plotting cumulative returns (Figure 2) …")
        _plot_cumulative(strategy_returns, save_path=f"{save_dir}/cumulative_returns.png")
        print(f"   Saved → {save_dir}/cumulative_returns.png")

    # Save returns
    strategy_returns.to_csv(f"{save_dir}/strategy_returns.csv")

    return strategy_returns, all_signals


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cumulative(
    strategy_returns: pd.DataFrame,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        "PCA_SUB":   ("navy",   "-",  2.0),
        "DOUBLE":    ("steelblue", "--", 1.5),
        "PCA_PLAIN": ("grey",   "-.", 1.2),
        "MOM":       ("black",  ":",  1.2),
    }

    for name in ["PCA_SUB", "DOUBLE", "PCA_PLAIN", "MOM"]:
        if name not in strategy_returns.columns:
            continue
        ret = strategy_returns[name].dropna()
        wealth = cumulative_wealth(ret)
        color, ls, lw = styles.get(name, ("C0", "-", 1.0))
        ax.plot(wealth.index, wealth.values, label=name.replace("_", " "),
                color=color, linestyle=ls, linewidth=lw)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Wealth")
    ax.set_title("Cumulative Returns by Strategy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the Japan-US sector lead-lag strategy."
    )
    p.add_argument("--start",         default=BACKTEST_START, help="Backtest start date")
    p.add_argument("--end",           default=BACKTEST_END,   help="Backtest end date")
    p.add_argument("--prior-start",   default=PRIOR_START,    help="Prior window start")
    p.add_argument("--prior-end",     default=PRIOR_END,      help="Prior window end")
    p.add_argument("--window",        type=int, default=L,    help="Rolling window (days)")
    p.add_argument("--K",             type=int, default=K,    help="PCA components")
    p.add_argument("--q",             type=float, default=Q,  help="Long-short quantile")
    p.add_argument("--no-plot",       action="store_true",    help="Skip plot generation")
    p.add_argument("--save-dir",      default="results",      help="Output directory")
    p.add_argument("--force-refresh", action="store_true",    help="Re-download data")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        backtest_start=args.start,
        backtest_end=args.end,
        prior_start=args.prior_start,
        prior_end=args.prior_end,
        window=args.window,
        K_components=args.K,
        q=args.q,
        plot=not args.no_plot,
        save_dir=args.save_dir,
        force_refresh=args.force_refresh,
    )
