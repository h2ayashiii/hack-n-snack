"""
realtime_run.py
===============
Run the subspace-regularized PCA lead-lag model *at the current point in
time* and print the signal it produces for the next Japanese session.

The paper's model is unsupervised (it is PCA, not a trained predictor):
there is no label to score against in real time.  The meaningful
"real-time output" is therefore the current state of the model -- the
factor structure extracted today and the long/short book it implies for
tomorrow:

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
    python3 realtime_run.py                 # one snapshot, save a chart
    python3 realtime_run.py --no-chart      # text only
    python3 realtime_run.py --watch 300     # refresh every 300s
    python3 realtime_run.py --lam 0.9 --L 60 --K 3 --q 0.3
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import numpy as np
import pandas as pd

import common as C


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
def fetch_prices(prior_start="2010-01-01"):
    """Download open/close prices via yfinance from prior_start to today.

    Uses auto_adjust=True so dividends and splits are reflected in the
    price series consistently (same as lead_lag_strategy v1).
    Returns (open_df, close_df) or raises on any failure.
    """
    import yfinance as yf  # local import so the script loads without it

    tickers = C.US_TICKERS + C.JP_TICKERS
    end = dt.date.today() + dt.timedelta(days=1)
    raw = yf.download(tickers, start=prior_start, end=end.isoformat(),
                      auto_adjust=True, progress=False, group_by="column")
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data")

    open_ = raw["Open"][tickers]
    close = raw["Close"][tickers]
    # keep rows where at least the U.S. side is present, forward-fill small gaps
    close = close.ffill(limit=2)
    open_ = open_.ffill(limit=2)
    good = close.dropna(how="any")
    open_ = open_.loc[good.index]
    close = good
    if len(close) < 120:
        raise RuntimeError(f"insufficient history ({len(close)} rows)")
    return open_, close


def synthetic_window(seed=None):
    """Fallback: generate a full history from the idealized model so the
    pipeline still has something to chew on.

    Starts from 2010-01-01 (matching the real prior_start default) so that
    date-based prior slicing works correctly in snapshot().
    """
    from verify_logic import make_synthetic
    if seed is None:
        seed = int(time.time()) % 100000
    # 2010-01-01 to ~2026: ~4000 trading days; make_synthetic uses bdate_range
    # so pass enough days to cover the full desired span.
    days = 4100
    syn = make_synthetic(seed=seed, days=days, start_date="2010-01-01")
    # turn returns into pseudo prices so the real pipeline path is exercised
    rcc = syn["rcc"]
    tickers = syn["tickers"]
    close = (1.0 + rcc).cumprod() * 100.0
    # synthetic intraday open: close_{t-1} nudged so roc carries the spill
    roc_J = syn["roc_J"].reindex(columns=C.JP_TICKERS)
    open_ = close.shift(1).copy()
    # for JP, set open so that close/open-1 == roc_J  (open = close/(1+roc))
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
        except Exception as e:  # noqa: BLE001
            print(f"[warn] live download failed ({type(e).__name__}: {e});"
                  " falling back to synthetic window.", file=sys.stderr)
    o, c = synthetic_window()
    return o, c, "SYNTHETIC fallback (idealized model)"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def snapshot(open_df, close_df, source, L=60, lam=0.9, K=3, q=0.3,
             prior_end="2014-12-31", prior_days=400):
    """Compute the real-time signal snapshot.

    Parameters
    ----------
    prior_end  : rows on or before this date are used to estimate C_full
                 (mirrors the 2010-2014 training window in v1 / the paper).
                 Prevents the prior from being contaminated by the backtest
                 period.  Falls back to the first ``prior_days`` rows when
                 the date range yields insufficient data (e.g. synthetic).
    prior_days : minimum number of rows required for the date-based prior;
                 also the fallback row count for synthetic / short histories.
    """
    tickers = C.US_TICKERS + C.JP_TICKERS
    close_df = close_df[tickers]
    rcc = C.close_to_close_returns(close_df)

    if len(rcc) < L + 1:
        raise RuntimeError(f"need >= {L+1} return rows, have {len(rcc)}")

    idx = {t: i for i, t in enumerate(tickers)}
    us_idx = np.array([idx[t] for t in C.US_TICKERS])
    jp_idx = np.array([idx[t] for t in C.JP_TICKERS])

    rcc_v = rcc[tickers].values
    signal_date = rcc.index[-1]                 # day t (latest U.S. close)
    window = rcc_v[-L - 1:-1]                    # W_t = previous L days
    today_us = rcc_v[-1, us_idx]                 # today's U.S. shock

    # Build prior C0 from a fixed training period (paper Section 4.2:
    # 2010-2014) to avoid lookahead bias.  When the date-based slice has
    # fewer than `prior_days` rows (e.g. synthetic data whose dates start
    # after prior_end), fall back to the first `prior_days` rows instead.
    V0 = C.build_prior_vectors(tickers)
    prior_mask = rcc.index <= pd.Timestamp(prior_end)
    if prior_mask.sum() >= prior_days:
        prior_rcc = rcc_v[prior_mask]
    else:
        prior_rcc = rcc_v[:prior_days]
    Zprior, _, _ = C.standardize_window(prior_rcc)
    C_full = C.correlation_from_Z(Zprior)
    C0 = C.build_C0(C_full, V0)

    res = C.compute_signal_for_day(window, today_us, us_idx, jp_idx,
                                   C0, lam=lam, K=K)
    z_hat = res["z_hat_J"]
    w = C.long_short_weights(z_hat, q=q)

    return dict(source=source, signal_date=signal_date, z_hat=z_hat,
                f=res["f"], evals=res["evals"], z_U=res["z_U"], w=w,
                K=K, lam=lam, L=L, q=q, us_idx=us_idx, jp_idx=jp_idx)


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

    # eigen-spectrum
    ev = s["evals"]
    tot = ev[ev > 0].sum()
    print("\n Regularized correlation spectrum (top 6 eigenvalues):")
    for k in range(min(6, len(ev))):
        bar = "#" * int(40 * ev[k] / ev[0])
        tag = "  <- retained" if k < s["K"] else ""
        print(f"   lambda_{k+1:<2d} = {ev[k]:7.3f}  ({100*ev[k]/tot:5.1f}%) {bar}{tag}")

    # factor scores extracted from today's U.S. move
    print("\n Common-factor scores f_t extracted from today's U.S. shock:")
    names = ["global", "US-Japan spread", "cyclical-defensive"]
    for k in range(s["K"]):
        nm = names[k] if k < len(names) else f"factor {k+1}"
        print(f"   f_{k+1} ({nm:<18s}) = {s['f'][k]:+.3f}")

    # predicted JP returns ranked
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

    # the book
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


def save_chart(s, path="realtime_signal.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(s["z_hat"])
    z = s["z_hat"][order]
    labels = [C.JP_TICKERS[i] for i in order]
    colors = ["#d62728" if s["w"][i] < 0 else
              ("#2ca02c" if s["w"][i] > 0 else "#bbbbbb") for i in order]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(z)), z, color=colors)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("predicted standardised return  zhat_{J,t+1}")
    ax.set_title(f"Lead-lag signal for next JP session\n"
                 f"signal day {pd.Timestamp(s['signal_date']).date()}  "
                 f"(green=LONG, red=SHORT)  [{s['source'].split()[0]}]",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[chart] written to {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=60, help="estimation window")
    ap.add_argument("--lam", type=float, default=0.9, help="shrinkage lambda")
    ap.add_argument("--K", type=int, default=3, help="number of factors")
    ap.add_argument("--q", type=float, default=0.3, help="long/short quantile")
    ap.add_argument("--prior-start", default="2010-01-01",
                    help="start date for yfinance download (covers the prior period)")
    ap.add_argument("--prior-end", default="2014-12-31",
                    help="end of prior training window for C_full estimation")
    ap.add_argument("--offline", action="store_true",
                    help="skip the network and use the synthetic fallback")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--watch", type=int, default=0,
                    help="refresh every N seconds (0 = single snapshot)")
    args = ap.parse_args()

    def one():
        o, c, src = get_data(args.prior_start, allow_network=not args.offline)
        s = snapshot(o, c, src, L=args.L, lam=args.lam, K=args.K, q=args.q,
                     prior_end=args.prior_end)
        print_snapshot(s)
        if not args.no_chart:
            save_chart(s)

    if args.watch > 0:
        print(f"[watch] refreshing every {args.watch}s -- Ctrl-C to stop")
        try:
            while True:
                one()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[watch] stopped.")
    else:
        one()


if __name__ == "__main__":
    main()
