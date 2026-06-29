"""
realtime_run.py
===============
Run the subspace-regularized PCA lead-lag model and print the signal it
produces for the next Japanese session.

Supports single-day (point-in-time) and multi-day range prediction.

The paper's model is unsupervised (PCA, not a trained predictor).
The meaningful "real-time output" is the current state of the model —
the factor structure and the long/short book it implies for tomorrow:

    * today's U.S. close-to-close shock z_{U,t}                  (eq. 17)
    * the K common-factor scores f_t it projects onto            (eq. 18)
    * the eigen-spectrum of the regularized correlation matrix   (eq. 14)
    * the predicted standardised Japanese returns zhat_{J,t+1}   (eq. 19)
    * the resulting long / short TOPIX-17 ETF book for t+1        (eqs. 3-7)

Data
----
Tries to download the most recent prices with yfinance.  If the network
or yfinance is unavailable, it transparently falls back to a synthetic
window so the script always produces a snapshot (a banner makes the data
source explicit).

Run
---
    # Single snapshot (latest available date)
    python realtime_run.py

    # Specific single date
    python realtime_run.py --date 2024-11-01

    # Date range (produces heatmap chart)
    python realtime_run.py --start-date 2024-10-01 --end-date 2024-11-30

    # Text only, no chart
    python realtime_run.py --no-chart

    # Watch mode (auto-refresh)
    python realtime_run.py --watch 300

    # Custom params
    python realtime_run.py --lam 0.9 --L 60 --K 3 --q 0.3
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import numpy as np
import pandas as pd

import common as C


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
def fetch_prices(prior_start="2010-01-01"):
    """Download open/close prices via yfinance from prior_start to today."""
    import yfinance as yf

    tickers = C.US_TICKERS + C.JP_TICKERS
    end = dt.date.today() + dt.timedelta(days=1)
    raw = yf.download(tickers, start=prior_start, end=end.isoformat(),
                      auto_adjust=True, progress=False, group_by="column")
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data")

    open_ = raw["Open"][tickers]
    close = raw["Close"][tickers]
    close = close.ffill(limit=2)
    open_ = open_.ffill(limit=2)
    good = close.dropna(how="any")
    open_ = open_.loc[good.index]
    close = good
    if len(close) < 120:
        raise RuntimeError(f"insufficient history ({len(close)} rows)")
    return open_, close


def synthetic_window(seed=None):
    """Fallback: generate a full history from the idealized model."""
    from verify_logic import make_synthetic
    if seed is None:
        seed = int(time.time()) % 100000
    days = 4100
    syn = make_synthetic(seed=seed, days=days, start_date="2010-01-01")
    rcc = syn["rcc"]
    close = (1.0 + rcc).cumprod() * 100.0
    roc_J = syn["roc_J"].reindex(columns=C.JP_TICKERS)
    open_ = close.shift(1).copy()
    for j in C.JP_TICKERS:
        open_[j] = close[j] / (1.0 + roc_J[j].fillna(0.0))
    open_ = open_.dropna()
    close = close.loc[open_.index]
    return open_, close


def get_data(prior_start="2010-01-01", allow_network=True):
    """Return (open_df, close_df, source_str)."""
    if allow_network:
        try:
            o, c = fetch_prices(prior_start)
            return o, c, "yfinance (live market data)"
        except Exception as e:
            print(f"[warn] live download failed ({type(e).__name__}: {e});"
                  " falling back to synthetic window.", file=sys.stderr)
    o, c = synthetic_window()
    return o, c, "SYNTHETIC fallback (idealized model)"


# ---------------------------------------------------------------------------
# Prior estimation (shared across all snapshots)
# ---------------------------------------------------------------------------
def build_prior(close_df, tickers, prior_end="2014-12-31", prior_days=400):
    """Build the fixed C0 prior from the training period."""
    rcc = C.close_to_close_returns(close_df[tickers])
    V0 = C.build_prior_vectors(tickers)
    prior_mask = rcc.index <= pd.Timestamp(prior_end)
    rcc_v = rcc.values
    if prior_mask.sum() >= prior_days:
        prior_rcc = rcc_v[prior_mask]
    else:
        prior_rcc = rcc_v[:prior_days]
    Zprior, _, _ = C.standardize_window(prior_rcc)
    C_full = C.correlation_from_Z(Zprior)
    C0 = C.build_C0(C_full, V0)
    return rcc, C0


# ---------------------------------------------------------------------------
# Single-date snapshot
# ---------------------------------------------------------------------------
def snapshot_at(rcc: pd.DataFrame, C0: np.ndarray, signal_date,
                source: str, L=60, lam=0.9, K=3, q=0.3):
    """Compute the lead-lag signal for a specific signal date.

    Parameters
    ----------
    signal_date : the date of the U.S. close we use as the signal (day t).
                  The prediction is for the JP open-to-close on day t+1.
    """
    tickers = list(rcc.columns)
    idx = {t: i for i, t in enumerate(tickers)}
    us_idx = np.array([idx[t] for t in C.US_TICKERS])
    jp_idx = np.array([idx[t] for t in C.JP_TICKERS])

    signal_ts = pd.Timestamp(signal_date)
    # find the positional index of the signal row
    rcc_v = rcc.values
    dates = rcc.index

    pos = dates.searchsorted(signal_ts, side="right") - 1
    if pos < 0 or dates[pos] > signal_ts:
        raise ValueError(f"No trading data on or before {signal_date}")
    actual_date = dates[pos]

    if pos < L:
        raise ValueError(
            f"Not enough history before {actual_date.date()} "
            f"(need {L} rows, have {pos})"
        )

    window = rcc_v[pos - L:pos]
    today_us = rcc_v[pos, us_idx]

    res = C.compute_signal_for_day(window, today_us, us_idx, jp_idx,
                                   C0, lam=lam, K=K)
    z_hat = res["z_hat_J"]
    w = C.long_short_weights(z_hat, q=q)

    return dict(source=source, signal_date=actual_date, z_hat=z_hat,
                f=res["f"], evals=res["evals"], z_U=res["z_U"], w=w,
                K=K, lam=lam, L=L, q=q, us_idx=us_idx, jp_idx=jp_idx)


def snapshot_range(rcc: pd.DataFrame, C0: np.ndarray, start_date, end_date,
                   source: str, L=60, lam=0.9, K=3, q=0.3):
    """Compute snapshots for all trading dates in [start_date, end_date].

    Returns a list of snapshot dicts (only dates with sufficient history).
    """
    dates = rcc.index
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    target_dates = dates[(dates >= start_ts) & (dates <= end_ts)]

    snapshots = []
    for d in target_dates:
        try:
            s = snapshot_at(rcc, C0, d, source, L=L, lam=lam, K=K, q=q)
            snapshots.append(s)
        except ValueError:
            pass

    if not snapshots:
        raise RuntimeError(
            f"No valid trading dates in [{start_date}, {end_date}] "
            f"with sufficient history (L={L})."
        )
    return snapshots


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------
def print_snapshot(s):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "=" * 74
    print(line)
    print(" SUBSPACE-REGULARIZED PCA LEAD-LAG  --  REAL-TIME SNAPSHOT")
    print(line)
    print(f" generated at      : {now}")
    print(f" data source       : {s['source']}")
    print(f" signal day (t)    : {pd.Timestamp(s['signal_date']).date()}  "
          f"(latest U.S. close-to-close)")
    print(f" predicting        : next Japanese open-to-close (t+1)")
    print(f" params            : L={s['L']}  lambda={s['lam']}  K={s['K']}  q={s['q']}")
    print(line)

    ev = s["evals"]
    tot = ev[ev > 0].sum()
    print("\n Regularized correlation spectrum (top 6 eigenvalues):")
    for k in range(min(6, len(ev))):
        bar = "#" * int(40 * ev[k] / ev[0])
        tag = "  <- retained" if k < s["K"] else ""
        print(f"   lambda_{k+1:<2d} = {ev[k]:7.3f}  ({100*ev[k]/tot:5.1f}%) {bar}{tag}")

    print("\n Common-factor scores f_t extracted from today's U.S. shock:")
    names = ["global", "US-Japan spread", "cyclical-defensive"]
    for k in range(s["K"]):
        nm = names[k] if k < len(names) else f"factor {k+1}"
        print(f"   f_{k+1} ({nm:<18s}) = {s['f'][k]:+.3f}")

    order = np.argsort(s["z_hat"])[::-1]
    print("\n Predicted standardised Japanese returns  zhat_{J,t+1}  (ranked):")
    print("   rank  ETF      sector                              zhat    book")
    book = {i: "" for i in range(len(s["z_hat"]))}
    longs = np.where(s["w"] > 0)[0]
    shorts = np.where(s["w"] < 0)[0]
    for i in longs:
        book[i] = "LONG"
    for i in shorts:
        book[i] = "SHORT"
    for rk, i in enumerate(order, 1):
        tk = C.JP_TICKERS[i]
        lbl = C.JP_LABELS.get(tk, "")
        print(f"   {rk:>3d}   {tk:<7s} {lbl:<35s} {s['z_hat'][i]:+.3f}  {book[i]}")

    print("\n Implied long/short book for next session "
          "(equal weight, dollar neutral):")
    print("   LONG :")
    for i in sorted(longs, key=lambda j: -s["z_hat"][j]):
        tk = C.JP_TICKERS[i]
        print(f"      {tk:<7s} {C.JP_LABELS.get(tk,''):<35s} w=+{s['w'][i]:.3f}")
    print("   SHORT:")
    for i in sorted(shorts, key=lambda j: s["z_hat"][j]):
        tk = C.JP_TICKERS[i]
        print(f"      {tk:<7s} {C.JP_LABELS.get(tk,''):<35s} w={s['w'][i]:.3f}")
    print(f"\n   sum(w) = {s['w'].sum():+.3f}   sum|w| = {np.abs(s['w']).sum():.3f}"
          "   (target: 0 and 2)")
    print(line)


def print_range_summary(snapshots):
    """Print a compact table of predicted rankings across multiple dates."""
    line = "=" * 74
    print(line)
    print(f" SIGNAL RANGE SUMMARY  ({len(snapshots)} trading days)")
    s0 = snapshots[0]
    print(f" params: L={s0['L']}  lambda={s0['lam']}  K={s0['K']}  q={s0['q']}")
    print(f" dates:  {pd.Timestamp(snapshots[0]['signal_date']).date()}"
          f"  →  {pd.Timestamp(snapshots[-1]['signal_date']).date()}")
    print(line)

    for s in snapshots:
        d = pd.Timestamp(s["signal_date"]).date()
        longs = [C.JP_TICKERS[i] for i in np.where(s["w"] > 0)[0]]
        shorts = [C.JP_TICKERS[i] for i in np.where(s["w"] < 0)[0]]
        longs_str = " ".join(t.replace(".T", "") for t in longs)
        shorts_str = " ".join(t.replace(".T", "") for t in shorts)
        print(f" {d}  LONG: {longs_str:<30s}  SHORT: {shorts_str}")
    print(line)


# ---------------------------------------------------------------------------
# Chart output
# ---------------------------------------------------------------------------
def save_chart_single(s, path):
    """Horizontal bar chart for a single signal date."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(s["z_hat"])
    z = s["z_hat"][order]
    labels = [C.JP_LABELS.get(C.JP_TICKERS[i], C.JP_TICKERS[i]) for i in order]
    tickers_ordered = [C.JP_TICKERS[i] for i in order]
    colors = ["#d62728" if s["w"][i] < 0 else
              ("#2ca02c" if s["w"][i] > 0 else "#bbbbbb") for i in order]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(z)), z, color=colors)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([f"{tk}  {lb}" for tk, lb in zip(tickers_ordered, labels)],
                       fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Predicted standardised return  $\\hat{z}_{J,t+1}$")
    ax.set_title(
        f"Lead-lag signal for next JP session\n"
        f"Signal day: {pd.Timestamp(s['signal_date']).date()}  "
        f"(green=LONG, red=SHORT, grey=neutral)  "
        f"[{s['source'].split()[0]}]",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="LONG"),
        Patch(facecolor="#d62728", label="SHORT"),
        Patch(facecolor="#bbbbbb", label="neutral"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[chart] written to {path}")
    plt.close(fig)


def save_chart_range(snapshots, path):
    """Heatmap of predicted returns across a date range.

    Rows: JP sectors (sorted by mean signal strength, strongest at top)
    Columns: signal dates
    Color: diverging colormap (red=negative, white=neutral, green=positive)
    Annotations: ▲ = LONG position, ▼ = SHORT position
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_jp = len(C.JP_TICKERS)
    n_dates = len(snapshots)

    # Build matrices: zhat (n_jp x n_dates), weights (n_jp x n_dates)
    zhat_mat = np.zeros((n_jp, n_dates))
    weight_mat = np.zeros((n_jp, n_dates))
    date_labels = []

    for col, s in enumerate(snapshots):
        zhat_mat[:, col] = s["z_hat"]
        weight_mat[:, col] = s["w"]
        date_labels.append(str(pd.Timestamp(s["signal_date"]).date()))

    # Sort rows by mean absolute signal (strongest sectors at top)
    row_order = np.argsort(np.abs(zhat_mat).mean(axis=1))[::-1]
    zhat_sorted = zhat_mat[row_order, :]
    weight_sorted = weight_mat[row_order, :]
    row_labels = [
        f"{C.JP_TICKERS[i].replace('.T', '')}  {C.JP_LABELS.get(C.JP_TICKERS[i], '')}"
        for i in row_order
    ]

    # Dynamic figure size based on number of dates
    fig_w = max(10, min(2.0 + n_dates * 0.55, 32))
    fig_h = max(6, n_jp * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Symmetric color scale
    vmax = np.nanpercentile(np.abs(zhat_sorted), 95)
    vmax = max(vmax, 0.1)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(zhat_sorted, aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Annotate long/short positions
    for col in range(n_dates):
        for row in range(n_jp):
            w = weight_sorted[row, col]
            if w > 0:
                ax.text(col, row, "▲", ha="center", va="center",
                        fontsize=7, color="darkgreen", fontweight="bold")
            elif w < 0:
                ax.text(col, row, "▼", ha="center", va="center",
                        fontsize=7, color="darkred", fontweight="bold")

    # Axes
    ax.set_xticks(range(n_dates))
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_jp))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Signal date (U.S. close day t)", fontsize=9)
    ax.set_ylabel("JP sector ETF", fontsize=9)

    s0 = snapshots[0]
    ax.set_title(
        f"Lead-lag signal heatmap  ({date_labels[0]} → {date_labels[-1]},  "
        f"{n_dates} days)\n"
        f"▲=LONG  ▼=SHORT  |  L={s0['L']}  λ={s0['lam']}  K={s0['K']}  "
        f"q={s0['q']}  [{s0['source'].split()[0]}]",
        fontsize=10,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("$\\hat{z}_{J,t+1}$ (predicted std. return)", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[chart] written to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Subspace-regularized PCA lead-lag real-time signal"
    )
    # Date selection
    date_grp = ap.add_argument_group("Date selection")
    date_grp.add_argument("--date", default=None,
                          help="Single signal date (YYYY-MM-DD). "
                               "Default: latest available trading day.")
    date_grp.add_argument("--start-date", default=None,
                          help="Start of date range (YYYY-MM-DD). "
                               "Use with --end-date for multi-day output.")
    date_grp.add_argument("--end-date", default=None,
                          help="End of date range (YYYY-MM-DD).")
    # Model params
    ap.add_argument("--L", type=int, default=60, help="estimation window")
    ap.add_argument("--lam", type=float, default=0.9, help="shrinkage lambda")
    ap.add_argument("--K", type=int, default=3, help="number of factors")
    ap.add_argument("--q", type=float, default=0.3, help="long/short quantile")
    ap.add_argument("--prior-start", default="2010-01-01",
                    help="start date for yfinance download")
    ap.add_argument("--prior-end", default="2014-12-31",
                    help="end of prior training window")
    # Output
    ap.add_argument("--output-dir", default="output",
                    help="directory for charts and results (default: output/)")
    ap.add_argument("--no-chart", action="store_true",
                    help="skip chart generation")
    ap.add_argument("--offline", action="store_true",
                    help="skip the network and use the synthetic fallback")
    ap.add_argument("--watch", type=int, default=0,
                    help="refresh every N seconds for single-date mode (0=once)")
    args = ap.parse_args()

    # Validate date args
    if args.start_date and not args.end_date:
        ap.error("--start-date requires --end-date")
    if args.end_date and not args.start_date:
        ap.error("--end-date requires --start-date")
    if args.date and args.start_date:
        ap.error("--date and --start-date/--end-date are mutually exclusive")

    out_dir = ensure_output_dir(args.output_dir)

    def run_once():
        o, c, src = get_data(args.prior_start, allow_network=not args.offline)
        tickers = C.US_TICKERS + C.JP_TICKERS
        rcc, C0 = build_prior(c, tickers, prior_end=args.prior_end)

        # --- Range mode ---
        if args.start_date:
            snapshots = snapshot_range(
                rcc, C0, args.start_date, args.end_date, src,
                L=args.L, lam=args.lam, K=args.K, q=args.q,
            )
            print_range_summary(snapshots)
            if not args.no_chart:
                fname = (f"realtime_signal"
                         f"_{args.start_date}_{args.end_date}.png")
                save_chart_range(snapshots, os.path.join(out_dir, fname))
            return

        # --- Single-date mode ---
        if args.date:
            signal_date = args.date
        else:
            signal_date = rcc.index[-1]

        s = snapshot_at(rcc, C0, signal_date, src,
                        L=args.L, lam=args.lam, K=args.K, q=args.q)
        print_snapshot(s)
        if not args.no_chart:
            d_str = str(pd.Timestamp(s["signal_date"]).date())
            fname = f"realtime_signal_{d_str}.png"
            save_chart_single(s, os.path.join(out_dir, fname))

    if args.watch > 0 and not args.start_date:
        print(f"[watch] refreshing every {args.watch}s -- Ctrl-C to stop")
        try:
            while True:
                run_once()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[watch] stopped.")
    else:
        run_once()


if __name__ == "__main__":
    main()
