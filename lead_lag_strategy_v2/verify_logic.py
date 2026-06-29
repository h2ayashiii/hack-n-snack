"""
verify_logic.py
===============
Verify the *logic* of the paper

    "Lead-lag strategies for Japanese and U.S. sectors using
     subspace regularization PCA"

on synthetic data generated from the idealized factor model of
Propositions 1-2 (eqs. 23-26).  Because the data-generating process is
known, we can check every claim the paper makes, rather than relying on
the empirical ETF history.

What this script demonstrates
-----------------------------
1.  Under the model  z_{U,t}   = V*_U g_t + e_U
                     z_{J,t+1} = V*_J g_t + e_J     (U.S. factors spill
    over to Japan the next day), the best linear predictor of tomorrow's
    Japanese returns from today's U.S. returns is the rank-K matrix

        B* = 1/(1+sigma_U^2) * V*_J V*_U^T            (eq. 25, Prop. 2)

2.  The subspace-regularized PCA predictor B^(K)_t (eq. 21) recovers B*
    (up to scale) and tracks it far more accurately than plain PCA when
    the estimation window L is short -- exactly the regime where the
    paper argues regularization matters.

3.  A long/short strategy built on the regularized signal dominates
    momentum, plain PCA and the double sort, reproducing the qualitative
    ranking of Table 2 (PCA_SUB best by risk-adjusted return and MDD).

Run
---
    python3 verify_logic.py [--seed 0] [--days 1500] [--out fig.png]

It prints a metrics table and writes a multi-panel figure.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: write PNG, never open a window
import matplotlib.pyplot as plt

import common as C


# ---------------------------------------------------------------------------
# Synthetic idealized model (eqs. 23-24, extended so the joint correlation
# matrix is informative)
# ---------------------------------------------------------------------------
def make_synthetic(seed=0, days=1500, sigma_u=0.9, sigma_j=0.9,
                   factor_vol=0.011, K_true=3, start_date="2015-01-01"):
    """Generate prices/returns consistent with the lead-lag factor model.

    The *same* common factor g_t drives
        * U.S.   close-to-close on day t,
        * Japan  close-to-close on day t   (contemporaneous co-movement,
          so the joint correlation matrix reveals the subspace), and
        * Japan  open-to-close on day t+1  (the tradeable spillover).

    The true loadings V*_U, V*_J are the paper's three prior directions
    (global / U.S.-Japan spread / cyclical-defensive), so the subspace
    prior C0 is a genuinely good regularizer.
    """
    rng = np.random.default_rng(seed)

    tickers = C.US_TICKERS + C.JP_TICKERS
    N = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    us_idx = np.array([idx[t] for t in C.US_TICKERS])
    jp_idx = np.array([idx[t] for t in C.JP_TICKERS])

    # True loadings = orthonormal prior directions (+ mild perturbation).
    V0 = C.build_prior_vectors(tickers)            # (N, 3)
    Vstar = V0[:, :K_true].copy()
    Vstar += 0.10 * rng.standard_normal(Vstar.shape)
    Vstar_U = Vstar[us_idx]
    Vstar_J = Vstar[jp_idx]

    NU, NJ = len(us_idx), len(jp_idx)
    g = rng.standard_normal((days, K_true))         # common factors g_t

    # standardised structural shocks
    zU = g @ Vstar_U.T + sigma_u * rng.standard_normal((days, NU))
    zJ_cc = g @ Vstar_J.T + sigma_j * rng.standard_normal((days, NJ))    # same day
    zJ_oc_next = g @ Vstar_J.T + sigma_j * rng.standard_normal((days, NJ))  # t+1 spill

    # turn standardised shocks into return series with realistic vol
    base_vol = factor_vol
    us_vol = base_vol * (1.0 + 0.3 * rng.random(NU))
    jp_vol = base_vol * (1.0 + 0.3 * rng.random(NJ))

    rcc = np.zeros((days, N))
    rcc[:, us_idx] = zU * us_vol
    rcc[:, jp_idx] = zJ_cc * jp_vol

    # tradeable Japanese open-to-close, lagged one day (roc_J[t+1] uses g_t)
    roc_J = np.full((days, NJ), np.nan)
    roc_J[1:] = (zJ_oc_next[:-1]) * jp_vol          # row t+1 driven by g_t

    dates = pd.bdate_range(start_date, periods=days)
    rcc_df = pd.DataFrame(rcc, index=dates, columns=tickers)
    roc_df = pd.DataFrame(roc_J, index=dates, columns=C.JP_TICKERS)

    return dict(rcc=rcc_df, roc_J=roc_df, tickers=tickers,
                Vstar=Vstar, Vstar_U=Vstar_U, Vstar_J=Vstar_J,
                us_idx=us_idx, jp_idx=jp_idx, sigma_u=sigma_u)


# ---------------------------------------------------------------------------
# Claim 1 + 2: recover B* and compare estimation accuracy vs window length
# ---------------------------------------------------------------------------
def true_predictor(syn):
    """B* = 1/(1+sigma_U^2) V*_J V*_U^T   (eq. 25)."""
    return (1.0 / (1.0 + syn["sigma_u"] ** 2)) * (syn["Vstar_J"] @ syn["Vstar_U"].T)


def estimate_B(syn, t, L, lam, K=3):
    """Estimate B^(K)_t from a window ending at t (eq. 21)."""
    rcc_v = syn["rcc"].values
    window = rcc_v[t - L:t]
    Z, mu, sig = C.standardize_window(window)
    C_t = C.correlation_from_Z(Z)
    C0 = build_prior(syn, lam)
    VK, _ = C.regularized_pca(C_t, C0, lam=lam, K=K)
    return C.predictor_matrix(VK, syn["us_idx"], syn["jp_idx"])


def build_prior(syn, lam):
    """Subspace prior from a long 'training' slice (mirrors the paper's
    2010-2014 C_full)."""
    V0 = C.build_prior_vectors(syn["tickers"])
    train = syn["rcc"].values[:400]
    Zt, _, _ = C.standardize_window(train)
    C_full = C.correlation_from_Z(Zt)
    return C.build_C0(C_full, V0)


def subspace_alignment(B_est, B_true):
    """Cosine similarity between the two predictor matrices (scale-free)."""
    a, b = B_est.ravel(), B_true.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def accuracy_vs_window(syn, windows=(20, 30, 45, 60, 90, 120, 180), reps=40):
    """For each L, average alignment of plain vs regularized estimate to B*."""
    B_true = true_predictor(syn)
    T = len(syn["rcc"])
    rng = np.random.default_rng(123)
    res = {"L": [], "plain": [], "sub": []}
    for L in windows:
        al_p, al_s = [], []
        for _ in range(reps):
            t = rng.integers(L, T - 1)
            al_p.append(subspace_alignment(estimate_B(syn, t, L, 0.0), B_true))
            al_s.append(subspace_alignment(estimate_B(syn, t, L, 0.9), B_true))
        res["L"].append(L)
        res["plain"].append(np.mean(al_p))
        res["sub"].append(np.mean(al_s))
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# Claim 3: strategy backtest on synthetic data
# ---------------------------------------------------------------------------
def run_strategies(syn, L=60, lam=0.9):
    C0 = build_prior(syn, lam)
    rets = C.run_backtest(syn["rcc"], syn["roc_J"], syn["tickers"],
                          C.JP_TICKERS, C0, L=L, lam=lam)
    metrics = {m: C.performance_metrics(rets[m].values) for m in rets.columns}
    return rets, pd.DataFrame(metrics).T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(syn, acc_df, rets, metrics, out_path):
    B_true = true_predictor(syn)
    # representative estimates at a short window
    t = len(syn["rcc"]) // 2
    B_sub = estimate_B(syn, t, 60, 0.9)
    B_plain = estimate_B(syn, t, 60, 0.0)
    # fix sign of plain estimate to best match true (PCA sign is arbitrary)
    if subspace_alignment(B_plain, B_true) < 0:
        B_plain = -B_plain
    if subspace_alignment(B_sub, B_true) < 0:
        B_sub = -B_sub

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    vmax = np.abs(B_true).max()
    for ax, M, title in [
        (fig.add_subplot(gs[0, 0]), B_true, "True B*  (eq. 25)"),
        (fig.add_subplot(gs[0, 1]), B_sub, "Subspace-reg PCA  B^(K)\n(window L=60, lambda=0.9)"),
        (fig.add_subplot(gs[0, 2]), B_plain, "Plain PCA  B^(K)\n(window L=60, lambda=0)"),
    ]:
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("U.S. sector")
        ax.set_ylabel("Japan sector")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # accuracy vs window
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(acc_df["L"], acc_df["sub"], "o-", label="subspace-reg (lambda=0.9)")
    ax2.plot(acc_df["L"], acc_df["plain"], "s--", label="plain PCA (lambda=0)")
    ax2.set_xlabel("estimation window L (days)")
    ax2.set_ylabel("cosine similarity of B^(K) to B*")
    ax2.set_title("Recovery of optimal predictor\n(regularization wins at short L)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # cumulative returns
    ax3 = fig.add_subplot(gs[1, 1])
    cum = (1.0 + rets).cumprod()
    for col in cum.columns:
        ax3.plot(cum.index, cum[col].values, label=col)
    ax3.set_title("Cumulative return on synthetic data", fontsize=10)
    ax3.set_ylabel("growth of 1")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    for lab in ax3.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")

    # metrics bars
    ax4 = fig.add_subplot(gs[1, 2])
    order = ["MOM", "PCA_PLAIN", "PCA_SUB", "DOUBLE"]
    rr = [metrics.loc[m, "RR"] for m in order]
    colors = ["#888", "#aaa", "#1f77b4", "#ff7f0e"]
    ax4.bar(order, rr, color=colors)
    ax4.set_title("Risk-adjusted return (R/R)", fontsize=10)
    ax4.set_ylabel("AR / RISK")
    for i, v in enumerate(rr):
        ax4.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for lab in ax4.get_xticklabels():
        lab.set_rotation(20)

    fig.suptitle("Verification of subspace-regularized PCA lead-lag logic "
                 "(synthetic idealized model)", fontsize=13, y=0.98)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[figure] written to {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default=os.path.join("output", "verify_logic.png"))
    args = ap.parse_args()

    print("=" * 70)
    print(" Subspace-regularized PCA lead-lag : LOGIC VERIFICATION")
    print(" (synthetic data from the idealized model of Prop. 1-2)")
    print("=" * 70)

    syn = make_synthetic(seed=args.seed, days=args.days)
    print(f"\nUniverse: {len(C.US_TICKERS)} U.S. + {len(C.JP_TICKERS)} Japan "
          f"= {len(syn['tickers'])} assets, {args.days} trading days.")

    # --- Claim 1: optimal predictor & one estimate ------------------------
    B_true = true_predictor(syn)
    t = len(syn["rcc"]) // 2
    B_sub = estimate_B(syn, t, 60, 0.9)
    B_plain = estimate_B(syn, t, 60, 0.0)
    print("\n[Claim 1-2] Predictor recovery (cosine similarity to B*, L=60):")
    print(f"    subspace-reg PCA : {abs(subspace_alignment(B_sub, B_true)):.3f}")
    print(f"    plain PCA        : {abs(subspace_alignment(B_plain, B_true)):.3f}")
    print(f"    rank(B*)         : {np.linalg.matrix_rank(B_true)}  "
          f"(<= K=3 as Prop. 1 requires)")

    # --- accuracy vs window ----------------------------------------------
    acc_df = accuracy_vs_window(syn)
    print("\n[Claim 2] Mean cosine similarity of B^(K) to B* vs window length:")
    print(acc_df.to_string(index=False,
                           formatters={"plain": "{:.3f}".format,
                                       "sub": "{:.3f}".format}))

    # --- Claim 3: strategy backtest --------------------------------------
    rets, metrics = run_strategies(syn)
    print("\n[Claim 3] Long/short performance on synthetic data "
          "(annualised, 252d):")
    show = metrics[["AR", "RISK", "RR", "MDD"]].copy()
    print(show.to_string(formatters={c: "{:.2f}".format for c in show.columns}))
    best = metrics["RR"].idxmax()
    print(f"\n  -> best risk-adjusted method: {best} "
          f"(matches the paper's PCA_SUB conclusion: {best == 'PCA_SUB'})")

    make_figure(syn, acc_df, rets, metrics, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
