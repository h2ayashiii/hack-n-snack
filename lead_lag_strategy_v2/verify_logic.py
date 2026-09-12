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
    Japanese returns from today's U.S. returns is

        B* = Sigma_JU Sigma_UU^-1                    (Prop. 2)

    which collapses to the paper's closed form 1/(1+sigma_U^2) V*_J V*_U^T
    (eq. 25) *only* when V*_U has orthonormal columns.  The generator is
    built so that assumption genuinely holds, and the script checks that
    the two expressions agree -- turning the premise of eq. (25) into a
    test rather than an assumption.

2.  The subspace-regularized PCA predictor B^(K)_t (eq. 21) is compared to
    B* on two axes: the cosine of the whole matrix, and the rank correlation
    of the *book* the two imply -- which is what the strategy actually uses,
    and where shrinkage looks very different.  Under the defaults the
    regularized estimate has the worse matrix cosine and much the better
    book rank IC: eqs. (3)-(7) only need the ordering of the 17 Japanese
    sectors, so the entrywise bias shrinkage buys is largely free.

3.  A long/short strategy built on the regularized signal is compared to
    momentum, plain PCA and the double sort (Table 2).

None of this is rigged to succeed.  The generator has three knobs that each
break the paper's conclusion, and the defaults are not the flattering end of
any of them -- raise ``--prior-misspec`` or ``--loading-drift`` far enough
and PCA_PLAIN wins, which is the correct answer under those conditions.

Three assumptions the paper leaves implicit are made explicit here:

* **Same-day co-movement.**  Eqs. (23)-(24) only relate z_{U,t} to
  z_{J,t+1}, but the estimator reads V*_J off the *contemporaneous* joint
  correlation matrix C_t.  If Japan did not also co-move with the U.S. on
  the same day, V*_J would not be identified at all.  ``--contemp-rho``
  controls that channel; at 0 the estimator has nothing to latch onto.
* **Orthonormality of V*_U** (see 1 above), required by the Woodbury step
  in the proof of Prop. 2 but not satisfied by a *block* of a joint PCA
  eigenbasis.  ``--predictor ridge`` applies the correction.
* **A moving factor structure.**  Shrinkage is only worth its bias if the
  short window is genuinely unable to pin the structure down (Section 4.4).
  With frozen loadings plain PCA at L=60 is already accurate, so the paper's
  own argument needs the loadings to wobble: ``--loading-drift`` does that,
  mean-reverting so that the prior stays a valid long-run description.

Run
---
    python3 verify_logic.py [--seed 0] [--days 1500] [--out fig.png]
    python3 verify_logic.py --prior-misspec 1.0     # prior carries no truth
    python3 verify_logic.py --contemp-rho 0.0       # break identification
    python3 verify_logic.py --loading-drift 0       # freeze the structure
    python3 verify_logic.py --predictor ridge       # Prop. 2 correction

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

# Rows of the synthetic sample reserved for estimating the prior C0.  The
# backtest starts after them, mirroring the paper's split (C0 from
# 2010-2014, evaluation from 2015 on) instead of overlapping the two.
PRIOR_ROWS = 400


# ---------------------------------------------------------------------------
# Synthetic idealized model (eqs. 23-24, extended so the joint correlation
# matrix is informative)
# ---------------------------------------------------------------------------
def _equal_norm_frame(M: np.ndarray, iters: int = 200) -> np.ndarray:
    """Nearest equal-norm tight frame to ``M`` (n, K), by alternating projection.

    Alternates between the two properties the idealized model needs:

    * orthonormal columns  ``V^T V = I_K``  -- required by the Woodbury step
      in the proof of Prop. 2, and
    * equal row norms ``||v_i||^2 = K/n`` -- so that every asset has the same
      total variance ``||v_i||^2 + sigma^2``.  Without this the pipeline's
      per-asset standardisation (eq. 9) divides each row by a *different*
      factor, the effective loadings stop being orthonormal, and eq. (25)
      cannot hold exactly however the columns were built.

    Both are fixed points, and the iteration keeps the frame close to the
    starting directions, so a prior-derived ``M`` stays prior-derived.
    """
    n, K = M.shape
    target = np.sqrt(K / n)
    V = M.astype(float).copy()
    for _ in range(iters):
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.where(nrm < 1e-12, 1e-12, nrm) * target
        Q, _ = np.linalg.qr(V)
        V = Q[:, :K]
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(nrm < 1e-12, 1e-12, nrm) * target


def _loading_path(base: np.ndarray, days: int, drift: float, rng, phi=0.98):
    """(days, n, K) loadings wobbling around ``base`` as an OU process.

    The paper's case for shrinkage is that a short rolling window cannot pin
    down a factor structure that *moves* (Section 4.4).  A stationary
    generator cannot exhibit that: with fixed loadings, plain PCA at L=60 is
    already accurate and regularization only adds bias.

    The deviation is mean-reverting, not a random walk, because the paper's
    prior stands for a *long-run* structure: the short window is noisy about
    it, but it does not become permanently wrong.  A random walk would drift
    the truth away from the prior for good, which tests a different (and
    unflattering) claim than the one the paper makes.  ``phi=0.98`` gives a
    half-life of about 35 sessions -- short relative to the 60-day window,
    long relative to a day.

    Each asset's loading norm is held at its base value, so the unit total
    variance the model assumes survives the wobble.
    """
    x = np.zeros(base.shape)
    out = np.empty((days, *base.shape))
    for t in range(days):
        x = phi * x + drift * rng.standard_normal(base.shape)
        out[t] = base + x
    tgt = np.linalg.norm(base, axis=1, keepdims=True)[None, :, :]
    nrm = np.linalg.norm(out, axis=2, keepdims=True)
    return out * (tgt / np.maximum(nrm, 1e-12))


def make_synthetic(seed=0, days=1500, factor_vol=0.011, K_true=3,
                   start_date="2015-01-01", prior_misspec=0.15,
                   contemp_rho=1.0, loading_drift=0.01):
    """Generate prices/returns consistent with the lead-lag factor model.

    The common factor g_t drives

        * U.S.   close-to-close on day t,
        * Japan  close-to-close on day t   (scaled by ``contemp_rho``: this
          is what makes the joint correlation matrix informative about
          V*_J, and it is an assumption the paper never states), and
        * Japan  open-to-close on day t+1  (the tradeable spillover).

    Everything is generated directly in *standardised* space with unit
    variance per asset, so the rolling standardisation of eqs. (8)-(9) is a
    no-op in expectation and the estimator sees exactly the model of
    eqs. (23)-(24).  That in turn makes the noise levels implied rather than
    free: with orthonormal columns ``sum_i ||v_i||^2 = K``, so equal row
    norms give ``||v_i||^2 = K/n`` and ``sigma^2 = 1 - K/n``.

    ``prior_misspec`` in [0, 1] blends the paper's prior directions with a
    random subspace: 0 makes the prior exactly right (which is what the
    original generator did, so regularization could not lose), 1 makes it
    carry no information about the truth.
    """
    rng = np.random.default_rng(seed)

    tickers = C.US_TICKERS + C.JP_TICKERS
    N = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    us_idx = np.array([idx[t] for t in C.US_TICKERS])
    jp_idx = np.array([idx[t] for t in C.JP_TICKERS])
    NU, NJ = len(us_idx), len(jp_idx)

    # True loadings: the paper's prior directions blended toward a random
    # subspace by `prior_misspec`, then shaped into the frame the idealized
    # model needs.
    V0 = C.build_prior_vectors(tickers)[:, :K_true]
    R = rng.standard_normal((N, K_true))
    R /= np.linalg.norm(R, axis=0, keepdims=True)
    M = (1.0 - prior_misspec) * V0 + prior_misspec * R

    # U.S.: orthonormal columns *and* equal row norms (Prop. 2 needs both).
    Vstar_U = _equal_norm_frame(M[us_idx])
    # The frame construction rotates the columns of M[us] within their span,
    # so column k of V*_U no longer names the same factor as column k of
    # M[jp].  Recover that rotation and apply it to the Japanese block too --
    # both blocks have to be written in the *same* factor basis, or the two
    # halves of the model load on different g's and B* comes out anti-aligned.
    T = np.linalg.pinv(M[us_idx]) @ Vstar_U
    # Japan: Prop. 2 puts no orthogonality condition on V*_J, only the unit
    # total variance that equal row norms give.  Row scaling is diagonal on
    # the left, so it preserves the factor basis fixed above.
    Vstar_J = M[jp_idx] @ T
    Vstar_J *= np.sqrt(K_true / NJ) / np.linalg.norm(Vstar_J, axis=1,
                                                     keepdims=True)

    # Implied idiosyncratic variances: total variance per asset is 1.
    sigma2_u = 1.0 - K_true / NU
    sigma2_j = 1.0 - K_true / NJ
    sigma_u, sigma_j = np.sqrt(sigma2_u), np.sqrt(sigma2_j)

    # Loading path: constant when loading_drift == 0, otherwise a slow walk
    # around the base loadings so a 60-day window is genuinely too short.
    base = np.empty((N, K_true))
    base[us_idx], base[jp_idx] = Vstar_U, Vstar_J
    if loading_drift > 0:
        Vpath = _loading_path(base, days, loading_drift, rng)
    else:
        Vpath = np.broadcast_to(base, (days, N, K_true))
    VU_t = np.ascontiguousarray(Vpath[:, us_idx, :])   # (days, NU, K)
    VJ_t = np.ascontiguousarray(Vpath[:, jp_idx, :])   # (days, NJ, K)

    g = rng.standard_normal((days, K_true))         # common factors g_t

    zU = (np.einsum("tk,tnk->tn", g, VU_t)
          + sigma_u * rng.standard_normal((days, NU)))
    # Same-day Japanese co-movement, dialled by contemp_rho but always unit
    # variance so the correlation matrix stays a correlation matrix.
    cc_noise = np.sqrt(max(1.0 - contemp_rho ** 2 * K_true / NJ, 1e-12))
    zJ_cc = (contemp_rho * np.einsum("tk,tnk->tn", g, VJ_t)
             + cc_noise * rng.standard_normal((days, NJ)))
    # Japan on t+1 loads on g_t through the loadings *in force at t*.
    zJ_oc_next = (np.einsum("tk,tnk->tn", g, VJ_t)
                  + sigma_j * rng.standard_normal((days, NJ)))

    # turn standardised shocks into return series with realistic vol
    us_vol = factor_vol * (1.0 + 0.3 * rng.random(NU))
    jp_vol = factor_vol * (1.0 + 0.3 * rng.random(NJ))

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
                Vstar_U=Vstar_U, Vstar_J=Vstar_J,
                VU_t=VU_t, VJ_t=VJ_t,
                us_idx=us_idx, jp_idx=jp_idx,
                sigma2_u=sigma2_u, sigma2_j=sigma2_j,
                prior_misspec=prior_misspec, contemp_rho=contemp_rho,
                loading_drift=loading_drift)


# ---------------------------------------------------------------------------
# Claim 1 + 2: recover B* and compare estimation accuracy vs window length
# ---------------------------------------------------------------------------
def blp_predictor(syn, t=None):
    """B* = Sigma_JU Sigma_UU^-1, the best linear predictor (Prop. 2).

    Computed straight from the definition, so it is the correct target
    whatever shape V*_U happens to have.  With ``loading_drift > 0`` the
    loadings move, so pass the day ``t`` whose predictor you want; omitting
    it uses the base loadings.
    """
    if t is None:
        VU, VJ = syn["Vstar_U"], syn["Vstar_J"]
    else:
        VU, VJ = syn["VU_t"][t], syn["VJ_t"][t]
    Sigma_JU = VJ @ VU.T
    Sigma_UU = VU @ VU.T + syn["sigma2_u"] * np.eye(VU.shape[0])
    return Sigma_JU @ np.linalg.inv(Sigma_UU)


def closed_form_predictor(syn, t=None):
    """The paper's eq. (25): B* = 1/(1+sigma_U^2) V*_J V*_U^T.

    Equal to :func:`blp_predictor` only when V*_U^T V*_U = I_K.
    """
    VU = syn["Vstar_U"] if t is None else syn["VU_t"][t]
    VJ = syn["Vstar_J"] if t is None else syn["VJ_t"][t]
    return (1.0 / (1.0 + syn["sigma2_u"])) * (VJ @ VU.T)


def build_prior(syn):
    """Subspace prior from the first PRIOR_ROWS rows (mirrors C_full, 2010-2014)."""
    V0 = C.build_prior_vectors(syn["tickers"])
    train = syn["rcc"].values[:PRIOR_ROWS]
    Zt, _, _ = C.standardize_window(train)
    return C.build_C0(C.correlation_from_Z(Zt), V0)


def estimate_B(syn, C0, t, L, lam, K=3, mode="paper"):
    """Estimate B^(K)_t from a window ending at t (eq. 21)."""
    window = syn["rcc"].values[t - L:t]
    Z, _, _ = C.standardize_window(window)
    C_t = C.correlation_from_Z(Z)
    VK, evals = C.regularized_pca(C_t, C0, lam=lam, K=K)
    return C.predictor_matrix(VK, syn["us_idx"], syn["jp_idx"], mode=mode,
                              sigma2=C.residual_variance(evals, K))


def subspace_alignment(B_est, B_true):
    """Cosine similarity between the two predictor matrices (scale-free).

    B = V_J V_U^T is invariant to flipping the sign of any eigenvector
    column, so a negative value here is a genuine estimation failure and is
    reported as such rather than being flipped away.
    """
    a, b = B_est.ravel(), B_true.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def signal_rank_ic(B_est, B_true, z_U):
    """Spearman correlation between the books B_est and B* imply for one day.

    The strategy never sees B itself -- it sorts ``B z_{U,t}`` and trades the
    tails (eqs. 3-7).  Two matrices can be far apart entrywise and still rank
    the 17 Japanese sectors identically, which is why this is reported next
    to the matrix cosine: shrinkage buys a large drop in variance at the cost
    of an entrywise bias the cosine punishes and the book does not.
    """
    a = pd.Series(B_est @ z_U)
    b = pd.Series(B_true @ z_U)
    return float(a.corr(b, method="spearman"))


def accuracy_vs_window(syn, C0, mode="paper",
                       windows=(20, 30, 45, 60, 90, 120, 180), reps=40):
    """For each L, compare plain vs regularized estimates against B*.

    Two columns per estimator: the matrix cosine, and the rank correlation of
    the books the two predictors imply on the day in question.
    """
    T = len(syn["rcc"])
    rcc_v = syn["rcc"].values
    us_idx = syn["us_idx"]
    rng = np.random.default_rng(123)
    res = {"L": [], "plain": [], "sub": [], "plain_IC": [], "sub_IC": []}
    for L in windows:
        al_p, al_s, ic_p, ic_s = [], [], [], []
        for _ in range(reps):
            t = rng.integers(max(L, PRIOR_ROWS), T - 1)
            B_true = blp_predictor(syn, t)
            Bp = estimate_B(syn, C0, t, L, 0.0, mode=mode)
            Bs = estimate_B(syn, C0, t, L, 0.9, mode=mode)
            al_p.append(subspace_alignment(Bp, B_true))
            al_s.append(subspace_alignment(Bs, B_true))
            # the day's standardised U.S. shock, exactly as eq. (17) builds it
            _, mu, sig = C.standardize_window(rcc_v[t - L:t])
            z_U = (rcc_v[t, us_idx] - mu[us_idx]) / sig[us_idx]
            ic_p.append(signal_rank_ic(Bp, B_true, z_U))
            ic_s.append(signal_rank_ic(Bs, B_true, z_U))
        res["L"].append(L)
        res["plain"].append(np.mean(al_p))
        res["sub"].append(np.mean(al_s))
        res["plain_IC"].append(np.mean(ic_p))
        res["sub_IC"].append(np.mean(ic_s))
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# Claim 3: strategy backtest on synthetic data
# ---------------------------------------------------------------------------
def run_strategies(syn, C0, L=60, lam=0.9, mode="paper"):
    """Backtest on the post-prior slice only, so C0 is strictly out-of-sample."""
    rcc = syn["rcc"].iloc[PRIOR_ROWS:]
    roc = syn["roc_J"].iloc[PRIOR_ROWS:]
    rets = C.run_backtest(rcc, roc, syn["tickers"], C.JP_TICKERS, C0,
                          L=L, lam=lam, mode=mode)
    metrics = {m: C.performance_metrics(rets[m].values) for m in rets.columns}
    return rets, pd.DataFrame(metrics).T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(syn, C0, acc_df, rets, metrics, out_path, mode="paper"):
    # representative estimates at a short window, inside the evaluation slice
    t = (PRIOR_ROWS + len(syn["rcc"])) // 2
    B_true = blp_predictor(syn, t)
    B_sub = estimate_B(syn, C0, t, 60, 0.9, mode=mode)
    B_plain = estimate_B(syn, C0, t, 60, 0.0, mode=mode)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    vmax = np.abs(B_true).max()
    for ax, M, title in [
        (fig.add_subplot(gs[0, 0]), B_true, "True B*  (Prop. 2)"),
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
    ax2.plot(acc_df["L"], acc_df["sub"], "o-", color="#1f77b4",
             label="subspace-reg, cosine")
    ax2.plot(acc_df["L"], acc_df["plain"], "s--", color="#888",
             label="plain PCA, cosine")
    ax2.plot(acc_df["L"], acc_df["sub_IC"], "o-", color="#1f77b4", alpha=0.45,
             label="subspace-reg, book rank IC")
    ax2.plot(acc_df["L"], acc_df["plain_IC"], "s--", color="#888", alpha=0.45,
             label="plain PCA, book rank IC")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("estimation window L (days)")
    ax2.set_ylabel("agreement with B*")
    ax2.set_title(
        f"Recovery of the best linear predictor\n"
        f"(prior misspec={syn['prior_misspec']:.2f}, "
        f"contemp rho={syn['contemp_rho']:.2f}, "
        f"drift={syn['loading_drift']:g})", fontsize=10)
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
    ax4.axhline(0, color="k", lw=0.8)
    ax4.set_title("Risk-adjusted return (R/R)", fontsize=10)
    ax4.set_ylabel("AR / RISK")
    for i, v in enumerate(rr):
        ax4.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for lab in ax4.get_xticklabels():
        lab.set_rotation(20)

    fig.suptitle("Verification of subspace-regularized PCA lead-lag logic "
                 f"(synthetic idealized model, predictor={mode})", fontsize=13,
                 y=0.98)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[figure] written to {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--prior-misspec", type=float, default=0.15,
                    help="0 = the prior directions are exactly the truth, "
                         "1 = the prior carries no information (default 0.15)")
    ap.add_argument("--contemp-rho", type=float, default=1.0,
                    help="strength of same-day U.S./Japan co-movement, the "
                         "unstated assumption that identifies V*_J from the "
                         "contemporaneous C_t (default 1.0)")
    ap.add_argument("--loading-drift", type=float, default=0.01,
                    help="per-day random-walk step of the true loadings; 0 "
                         "freezes the factor structure, which removes the "
                         "instability the paper's shrinkage exists to fight "
                         "(default 0.01)")
    ap.add_argument("--predictor", choices=C.PREDICTOR_MODES, default="paper",
                    help="'paper' = eq. (21) verbatim, 'ridge' = the Prop. 2 "
                         "correction (default paper)")
    ap.add_argument("--out", default=os.path.join("output", "verify_logic.png"))
    args = ap.parse_args()

    print("=" * 70)
    print(" Subspace-regularized PCA lead-lag : LOGIC VERIFICATION")
    print(" (synthetic data from the idealized model of Prop. 1-2)")
    print("=" * 70)

    syn = make_synthetic(seed=args.seed, days=args.days,
                         prior_misspec=args.prior_misspec,
                         contemp_rho=args.contemp_rho,
                         loading_drift=args.loading_drift)
    C0 = build_prior(syn)
    print(f"\nUniverse: {len(C.US_TICKERS)} U.S. + {len(C.JP_TICKERS)} Japan "
          f"= {len(syn['tickers'])} assets, {args.days} trading days "
          f"({PRIOR_ROWS} reserved for the prior).")
    print(f"Generator: prior-misspec={args.prior_misspec}  "
          f"contemp-rho={args.contemp_rho}  "
          f"loading-drift={args.loading_drift}  predictor={args.predictor}")

    # --- Claim 0: does the DGP satisfy the premise of eq. (25)? ------------
    VU = syn["Vstar_U"]
    gram_err = np.abs(VU.T @ VU - np.eye(VU.shape[1])).max()
    B_true = blp_predictor(syn)
    B_closed = closed_form_predictor(syn)
    cos_closed = subspace_alignment(B_closed, B_true)
    scale_closed = np.linalg.norm(B_closed) / np.linalg.norm(B_true)
    print("\n[Premise] Prop. 2 assumes V*_U^T V*_U = I_K (used in the "
          "Woodbury step):")
    print(f"    max |V*_U^T V*_U - I| : {gram_err:.2e}")
    print(f"    eq.(25) vs Sigma_JU Sigma_UU^-1 : cos={cos_closed:.6f}  "
          f"norm ratio={scale_closed:.6f}")
    if cos_closed < 0.9999 or abs(scale_closed - 1.0) > 1e-3:
        print("    -> WARNING: the closed form is NOT the BLP for this DGP.")
    else:
        print("    -> eq. (25) holds exactly here; using it as the target.")

    # --- Claim 1: rank and one estimate -----------------------------------
    t = (PRIOR_ROWS + args.days) // 2
    B_true = blp_predictor(syn, t)
    B_sub = estimate_B(syn, C0, t, 60, 0.9, mode=args.predictor)
    B_plain = estimate_B(syn, C0, t, 60, 0.0, mode=args.predictor)
    print("\n[Claim 1-2] Predictor recovery (cosine similarity to B*, L=60):")
    print(f"    subspace-reg PCA : {subspace_alignment(B_sub, B_true):+.3f}")
    print(f"    plain PCA        : {subspace_alignment(B_plain, B_true):+.3f}")
    print(f"    rank(B*)         : {np.linalg.matrix_rank(B_true)}  "
          f"(<= K=3 as Prop. 1 requires)")

    # --- both predictor modes side by side --------------------------------
    print("\n[Prop. 2 correction] alignment to B* at L=60, lambda=0.9:")
    for m in C.PREDICTOR_MODES:
        a = subspace_alignment(estimate_B(syn, C0, t, 60, 0.9, mode=m), B_true)
        print(f"    predictor={m:<6s} : {a:+.3f}")

    # --- accuracy vs window ----------------------------------------------
    acc_df = accuracy_vs_window(syn, C0, mode=args.predictor)
    print("\n[Claim 2] Mean cosine similarity of B^(K) to B* vs window length:")
    print("  cosine = similarity of the whole matrix to B*;  "
          "IC = rank correlation of the book they imply")
    print(acc_df.to_string(index=False,
                           formatters={c: "{:+.3f}".format
                                       for c in acc_df.columns if c != "L"}))

    # --- Claim 3: strategy backtest --------------------------------------
    rets, metrics = run_strategies(syn, C0, mode=args.predictor)
    print("\n[Claim 3] Long/short performance on synthetic data "
          "(annualised, 252d):")
    show = metrics[["AR", "RISK", "RR", "MDD"]].copy()
    print(show.to_string(formatters={c: "{:.2f}".format for c in show.columns}))
    best = metrics["RR"].idxmax()
    print(f"\n  -> best risk-adjusted method: {best} "
          f"(paper's conclusion is PCA_SUB: {best == 'PCA_SUB'})")
    if args.prior_misspec >= 0.99:
        print("     note: with a fully misspecified prior there is no reason "
              "for PCA_SUB to win, and it should not be expected to.")

    make_figure(syn, C0, acc_df, rets, metrics, args.out,
                mode=args.predictor)
    print("\nDone.")


if __name__ == "__main__":
    main()
