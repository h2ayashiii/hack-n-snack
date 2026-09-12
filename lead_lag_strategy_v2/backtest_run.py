"""
backtest_run.py
===============
Reproduce the paper's empirical results (Section 4.4) on real ETF prices.

    "Lead-lag strategies for Japanese and U.S. sectors using
     subspace regularization PCA"  (SIG-FIN-036-13)

``verify_logic.py`` checks the algebra on synthetic data where the truth is
known.  This script is the other half: it runs the same walk-forward loop on
the actual Select Sector SPDR / TOPIX-17 history and prints Table 1 and
Table 2.

Sample period
-------------
The paper says the data run from 2010-01 to 2025-12, but Table 1 reports
2590 observations for the tickers listed throughout, 2409 for XLRE and 1758
for XLC.  Those gaps -- 181 and 832 common trading days -- line up with
2015-01-01 to each ETF's inception (XLRE 2015-10-08, XLC 2018-06-19), so the
*evaluation* window is 2015-01 onward and 2010-2014 is there to estimate
C_full.  Hence the defaults below: prices from 2010-01-01, prior over
2010-2014, backtest from 2015-01-01, with the two kept disjoint.

Run
---
    python backtest_run.py                        # paper's settings
    python backtest_run.py --eval-start 2018-07-01
    python backtest_run.py --predictor ridge      # Prop. 2 correction
    python backtest_run.py --offline              # synthetic, for plumbing

Writes ``output/backtest_cumulative.png`` and
``output/backtest_returns.csv``.

Not implemented: the Fama-French 3 / Carhart 4 regressions of Tables 3-4,
which need Japanese factor return series this repository does not source.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import common as C
from realtime_run import build_prior, ensure_output_dir, get_data


# ---------------------------------------------------------------------------
# Table 1: per-ticker descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_table(rcc: pd.DataFrame, periods_per_year: int = 252):
    """Annualised return / vol / ratio, skew, kurtosis and N per ticker.

    Mirrors Table 1.  N is the number of usable close-to-close observations,
    which is the column to sanity-check the sample period against: the paper
    reports 2590 for most tickers, 2409 for XLRE and 1758 for XLC.
    """
    rows = {}
    for tk in rcc.columns:
        r = rcc[tk].dropna()
        if len(r) < 2:
            continue
        ret = periods_per_year * r.mean() * 100
        vol = np.sqrt(periods_per_year) * r.std(ddof=1) * 100
        rows[tk] = dict(Ret=ret, Vol=vol, RetVol=ret / vol if vol else np.nan,
                        Skew=r.skew(), Kurtosis=r.kurt(), N=len(r))
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Table 2: strategy performance
# ---------------------------------------------------------------------------
def performance_table(rets: pd.DataFrame):
    """AR / RISK / R/R / MDD per strategy, MDD as a positive number.

    ``common.performance_metrics`` returns the drawdown signed (it is a
    loss), while Table 2 prints its magnitude; flipped here to line the two
    up side by side.
    """
    rows = {m: C.performance_metrics(rets[m].values) for m in rets.columns}
    out = pd.DataFrame(rows).T[["AR", "RISK", "RR", "MDD", "N"]]
    out["MDD"] = out["MDD"].abs()
    return out


def save_chart(rets: pd.DataFrame, path: str, title_suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cum = (1.0 + rets).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    styles = {"MOM": ("#888888", "--"), "PCA_PLAIN": ("#aaaaaa", "--"),
              "PCA_SUB": ("#1f77b4", "-"), "DOUBLE": ("#ff7f0e", "-")}
    for col in cum.columns:
        colour, ls = styles.get(col, ("#333333", "-"))
        ax.plot(cum.index, cum[col].values, label=col, color=colour, ls=ls,
                lw=1.8 if ls == "-" else 1.2)
    ax.set_ylabel("growth of 1")
    ax.set_title(f"Cumulative return of the lead-lag strategies{title_suffix}",
                 fontsize=11)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.legend()
    ax.grid(alpha=0.3)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[chart] written to {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Empirical backtest of the subspace-regularized PCA "
                    "lead-lag strategies (Tables 1-2)")
    ap.add_argument("--prior-start", default="2010-01-01",
                    help="download start and beginning of the C_full window")
    ap.add_argument("--prior-end", default="2014-12-31",
                    help="end of the C_full window")
    ap.add_argument("--eval-start", default="2015-01-01",
                    help="first trade date of the backtest; keep it after "
                         "--prior-end or C0 is estimated in-sample")
    ap.add_argument("--eval-end", default=None, help="last trade date")
    ap.add_argument("--L", type=int, default=60, help="estimation window")
    ap.add_argument("--lam", type=float, default=0.9, help="shrinkage lambda")
    ap.add_argument("--K", type=int, default=3, help="number of factors")
    ap.add_argument("--q", type=float, default=0.3, help="long/short quantile")
    ap.add_argument("--predictor", choices=C.PREDICTOR_MODES, default="paper",
                    help="'paper' = eq. (21) verbatim (default), 'ridge' = "
                         "the Prop. 2 correction")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="skip the network and use the synthetic fallback "
                         "(exercises the plumbing; the numbers mean nothing)")
    args = ap.parse_args()

    out_dir = ensure_output_dir(args.output_dir)

    open_, close, source = get_data(args.prior_start,
                                    allow_network=not args.offline)
    # get_data falls back to the synthetic generator when the download
    # fails, which is right for a daily signal (better a flagged book than
    # none) and wrong here: a Table 2 that silently came from simulated
    # returns is worse than no Table 2 at all. Refuse it unless the caller
    # asked for synthetic on purpose.
    synthetic = source.startswith("SYNTHETIC")
    if synthetic and not args.offline:
        raise SystemExit(
            f"refusing to report a backtest on {source}: the live download "
            f"failed, and these numbers would not be the paper's. Re-run "
            f"when the data source is reachable, or pass --offline to "
            f"exercise the plumbing on simulated returns.")
    tickers = C.US_TICKERS + C.JP_TICKERS
    rcc, C0, prior_window = build_prior(close, tickers,
                                        prior_start=args.prior_start,
                                        prior_end=args.prior_end)
    roc_J = C.open_to_close_returns(open_[C.JP_TICKERS], close[C.JP_TICKERS])

    eval_start = pd.Timestamp(args.eval_start)
    if prior_window and eval_start <= prior_window[1]:
        print(f"[warn] --eval-start {eval_start.date()} is not after the "
              f"prior window (ends {prior_window[1].date()}); C0 has seen "
              f"part of the evaluation sample.", file=sys.stderr)

    # The estimation window reaches L rows back, so the backtest has to start
    # its loop that far before the first trade date it is meant to produce.
    first = rcc.index.searchsorted(eval_start)
    lo = max(0, first - args.L - 1)
    hi = (len(rcc) if args.eval_end is None
          else rcc.index.searchsorted(pd.Timestamp(args.eval_end), "right"))
    rcc_slice = rcc.iloc[lo:hi]
    if len(rcc_slice) < args.L + 2:
        raise SystemExit(f"not enough rows ({len(rcc_slice)}) for L={args.L}")

    line = "=" * 74
    print(line)
    print(" SUBSPACE-REGULARIZED PCA LEAD-LAG  --  EMPIRICAL BACKTEST")
    print(line)
    print(f" data source   : {source}")
    if synthetic:
        print(" *** SIMULATED RETURNS -- these are NOT the paper's numbers "
              "and reproduce nothing. ***")
    print(f" prior C_full  : {prior_window[0].date()} .. "
          f"{prior_window[1].date()}" if prior_window else " prior: n/a")
    print(f" evaluation    : from {eval_start.date()}"
          + (f" to {args.eval_end}" if args.eval_end else ""))
    print(f" params        : L={args.L}  lambda={args.lam}  K={args.K}  "
          f"q={args.q}  predictor={args.predictor}")
    print(line)

    # --- Table 1 ----------------------------------------------------------
    desc = descriptive_table(rcc.loc[rcc.index >= eval_start])
    print("\n Table 1 -- sector ETF descriptive statistics "
          "(evaluation sample):")
    print(desc.to_string(formatters={
        "Ret": "{:.2f}".format, "Vol": "{:.2f}".format,
        "RetVol": "{:.2f}".format, "Skew": "{:.2f}".format,
        "Kurtosis": "{:.2f}".format, "N": "{:.0f}".format}))
    print("\n   (the paper reports N = 2590 for tickers listed throughout, "
          "2409 for XLRE, 1758 for XLC;\n    a large gap means the sample "
          "period does not match the paper's)")

    # --- Table 2 ----------------------------------------------------------
    rets = C.run_backtest(rcc_slice, roc_J.reindex(rcc_slice.index), tickers,
                          C.JP_TICKERS, C0, L=args.L, lam=args.lam, K=args.K,
                          q=args.q, mode=args.predictor)
    rets = rets.loc[rets.index >= eval_start]
    if rets.empty:
        raise SystemExit("no tradeable days in the evaluation window")

    perf = performance_table(rets)
    print("\n Table 2 -- strategy performance (annualised, 252d; "
          "MDD as a positive number):")
    print(perf.to_string(formatters={
        "AR": "{:.2f}".format, "RISK": "{:.2f}".format,
        "RR": "{:.2f}".format, "MDD": "{:.2f}".format, "N": "{:.0f}".format}))
    best = perf["RR"].idxmax()
    print(f"\n   best R/R: {best}   smallest MDD: {perf['MDD'].idxmin()}")
    print(f"   paper: PCA_SUB on both (AR 23.79, RISK 10.70, R/R 2.22, "
          f"MDD 9.58)")
    print("\n   Tables 3-4 (Fama-French 3 / Carhart 4) are not implemented: "
          "they need\n   Japanese factor returns this repository does not "
          "source.")

    csv_path = os.path.join(out_dir, "backtest_returns.csv")
    rets.to_csv(csv_path)
    print(f"\n[data] daily strategy returns written to {csv_path}")
    if not args.no_chart:
        save_chart(rets, os.path.join(out_dir, "backtest_cumulative.png"),
                   title_suffix=f"  ({rets.index[0].date()} to "
                                f"{rets.index[-1].date()}, "
                                f"predictor={args.predictor})")
    print(line)


if __name__ == "__main__":
    main()
