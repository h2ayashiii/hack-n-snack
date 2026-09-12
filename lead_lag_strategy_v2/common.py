"""
common.py
=========
Lead-lag strategies for Japanese and U.S. sectors using *subspace
regularization PCA* (Nakagawa, Takemoto, Kubo, Kato; SIG-FIN-036-13).

This module holds the building blocks that are shared by both
`verify_logic.py` (idealized-model verification) and
`realtime_run.py` (point-in-time signal generation):

    * universe definition (11 U.S. + 17 Japanese sector ETFs)
    * return transforms        ........  rcc (close-to-close), roc (open-to-close)
    * rolling standardisation  ........  eqs. (8)-(9)
    * subspace prior C0        ........  eqs. (10)-(12)
    * regularised PCA          ........  eqs. (13)-(16)
    * lead-lag signal          ........  eqs. (17)-(21)
    * long/short portfolio     ........  eqs. (3)-(7)
    * performance metrics      ........  eqs. (27)-(30)
    * backtest loop driving every baseline (MOM, PCA_PLAIN, PCA_SUB, DOUBLE)

Everything operates on plain numpy / pandas so the two entry-point
scripts can feed it either real prices or synthetic factor-model data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Universe (Section 4.1)
# ---------------------------------------------------------------------------
# U.S.: the 11 Select Sector SPDR ETFs tracking the GICS sectors of the S&P 500.
US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI",
              "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

# Japan: the 17 NEXT FUNDS TOPIX-17 series ETFs.
JP_TICKERS = [f"{code}.T" for code in range(1617, 1634)]  # 1617.T .. 1633.T

# Human readable labels for the Japanese codes (TOPIX-17 sectors).
JP_LABELS = {
    "1617.T": "Foods",
    "1618.T": "Energy resources",
    "1619.T": "Construction & materials",
    "1620.T": "Raw materials & chemicals",
    "1621.T": "Pharmaceutical",
    "1622.T": "Automobiles & transportation",
    "1623.T": "Steel & nonferrous",
    "1624.T": "Machinery",
    "1625.T": "Electric appliances & precision",
    "1626.T": "IT & services, others",
    "1627.T": "Electric power & gas",
    "1628.T": "Transportation & logistics",
    "1629.T": "Commercial & wholesale",
    "1630.T": "Retail trade",
    "1631.T": "Banks",
    "1632.T": "Financials (ex banks)",
    "1633.T": "Real estate",
}

# Cyclical / defensive tags used to build the third subspace prior vector
# (Section 4.1, last paragraph).
US_CYCLICAL = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]


# ---------------------------------------------------------------------------
# 2. Return transforms (eqs. 1-2)
# ---------------------------------------------------------------------------
def close_to_close_returns(close: pd.DataFrame) -> pd.DataFrame:
    """rcc_{i,t} = P^close_{i,t} / P^close_{i,t-1} - 1   (eq. 1)."""
    return close.pct_change().dropna(how="all")


def open_to_close_returns(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """roc_{j,t} = P^close_{j,t} / P^open_{j,t} - 1      (eq. 2)."""
    return (close / open_ - 1.0).dropna(how="all")


# ---------------------------------------------------------------------------
# 3. Rolling standardisation (eqs. 8-9)
# ---------------------------------------------------------------------------
def standardize_window(rcc: np.ndarray):
    """Standardise a (L, N) window of close-to-close returns.

    Returns
    -------
    Z   : (L, N) standardised returns  z_{i,tau} = (rcc - mu)/sigma
    mu  : (N,)   window means
    sig : (N,)   window stds (population, /L as in eq. 8)
    """
    mu = rcc.mean(axis=0)
    sig = rcc.std(axis=0)  # numpy default ddof=0 -> divide by L (eq. 8)
    sig = np.where(sig < 1e-12, 1e-12, sig)
    Z = (rcc - mu) / sig
    return Z, mu, sig


def correlation_from_Z(Z: np.ndarray) -> np.ndarray:
    """C_t = (1/L) Z^T Z  -- correlation matrix of standardised returns."""
    L = Z.shape[0]
    C = (Z.T @ Z) / L
    # numerical clean-up: force unit diagonal & symmetry
    C = 0.5 * (C + C.T)
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    C = C / np.outer(d, d)
    return C


# ---------------------------------------------------------------------------
# 4. Subspace prior C0 (eqs. 10-12)
# ---------------------------------------------------------------------------
def _gram_schmidt(vectors):
    """Sequential Gram-Schmidt -> orthonormal columns (N, k)."""
    basis = []
    for v in vectors:
        w = v.astype(float).copy()
        for b in basis:
            w = w - (b @ w) * b
        nrm = np.linalg.norm(w)
        if nrm > 1e-12:
            basis.append(w / nrm)
    return np.column_stack(basis)


def build_prior_vectors(tickers) -> np.ndarray:
    """Construct the K0 = 3 orthonormal prior directions V0 (Section 3.1).

    1. global factor          v1 ~ 1
    2. U.S./Japan spread       v2 ~ (+1 on U.S., -1 on Japan), orth. to v1
    3. cyclical-defensive      v3 ~ (+1 cyclical, -1 defensive), orth. to v1,v2
    """
    tickers = list(tickers)
    N = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    v1 = np.ones(N)

    v2 = np.zeros(N)
    for t in tickers:
        v2[idx[t]] = 1.0 if t in US_TICKERS else -1.0

    v3 = np.zeros(N)
    cyc = set(US_CYCLICAL) | set(JP_CYCLICAL)
    deff = set(US_DEFENSIVE) | set(JP_DEFENSIVE)
    for t in tickers:
        if t in cyc:
            v3[idx[t]] = 1.0
        elif t in deff:
            v3[idx[t]] = -1.0

    return _gram_schmidt([v1, v2, v3])


def build_C0(C_full: np.ndarray, V0: np.ndarray) -> np.ndarray:
    """Low-rank, normalised subspace prior C0 (eqs. 10-12).

        D0     = diag( V0^T C_full V0 )          (eq. 10)
        C_raw  = V0 D0 V0^T                       (eq. 11)
        C0     = Delta^-1/2 C_raw Delta^-1/2      (eq. 12)  with Delta = diag(C_raw)

    ``C_full`` may contain NaN rows/columns for tickers with no quotes in
    the prior window (e.g. XLC before its 2018-06 inception when the prior
    is the paper's 2010-2014 period).  Each prior eigenvalue is then taken
    as the Rayleigh quotient of the prior direction restricted to the
    observed sub-block, which reduces to eq. (10) when nothing is missing.
    """
    C_full = np.asarray(C_full, dtype=float)
    K0 = V0.shape[1]
    avail = ~np.isnan(np.diag(C_full))
    C_sub = np.nan_to_num(C_full[np.ix_(avail, avail)], nan=0.0)
    d0 = np.empty(K0)
    for k in range(K0):
        v = V0[avail, k]
        nrm2 = v @ v
        d0[k] = (v @ C_sub @ v) / nrm2 if nrm2 > 1e-12 else 1.0
    d0 = np.clip(d0, 1e-8, None)
    C_raw = V0 @ np.diag(d0) @ V0.T
    d = np.sqrt(np.clip(np.diag(C_raw), 1e-12, None))
    C0 = C_raw / np.outer(d, d)
    # enforce exact unit diagonal
    np.fill_diagonal(C0, 1.0)
    return C0


# ---------------------------------------------------------------------------
# 5. Regularised PCA (eqs. 13-16)
# ---------------------------------------------------------------------------
def regularized_pca(C_t: np.ndarray, C0: np.ndarray, lam: float = 0.9, K: int = 3):
    """Shrink C_t toward the prior C0 and take the top-K eigenvectors.

        C^reg_t = (1 - lam) C_t + lam C0          (eq. 13)
        C^reg_t = V_t Lambda_t V_t^T              (eq. 14)

    Returns the top-K eigenvectors V^(K)_t (N, K) and the full eigenvalue
    spectrum (descending).
    """
    C_reg = (1.0 - lam) * C_t + lam * C0
    C_reg = 0.5 * (C_reg + C_reg.T)
    evals, evecs = np.linalg.eigh(C_reg)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    return evecs[:, :K], evals


# ---------------------------------------------------------------------------
# 6. Lead-lag signal (eqs. 17-21)
# ---------------------------------------------------------------------------
PREDICTOR_MODES = ("paper", "ridge")


def residual_variance(evals, K: int) -> float:
    """sigma_U^2 estimate from the regularised spectrum.

    The cross-section is standardised, so ``trace(C^reg_t) = N`` and the K
    retained eigenvalues account for ``sum(l_1..l_K)`` of it.  What is left
    over, per asset, is the idiosyncratic variance of the factor model of
    eqs. (23)-(24):

        sigma^2 ~= (N - sum_{k<=K} l_k) / N
    """
    evals = np.asarray(evals, dtype=float)
    N = len(evals)
    explained = float(np.clip(evals[:K].sum(), 0.0, N))
    return max((N - explained) / N, 1e-6)


def leadlag_signal(VK: np.ndarray, us_idx, jp_idx, z_U: np.ndarray,
                   mode: str = "paper", sigma2: float | None = None):
    """Map today's U.S. shock into a prediction of tomorrow's Japanese returns.

        f_t        = V^(K)_{U,t}^T z_{U,t}        (eq. 18)  factor scores
        zhat_{J}   = V^(K)_{J,t} f_t              (eq. 19)  prediction
                   = B^(K)_t z_{U,t},  B = V_J V_U^T (eqs. 20-21)

    ``mode`` selects which predictor is applied; see :func:`predictor_matrix`.
    ``"ridge"`` needs ``sigma2`` (estimate it with :func:`residual_variance`).
    """
    VKU = VK[us_idx, :]          # (NU, K)
    VKJ = VK[jp_idx, :]          # (NJ, K)
    proj = VKU.T @ z_U           # (K,)
    if mode == "paper":
        f = proj
    elif mode == "ridge":
        if sigma2 is None:
            raise ValueError("mode='ridge' requires sigma2")
        K = VK.shape[1]
        f = np.linalg.solve(VKU.T @ VKU + sigma2 * np.eye(K), proj)
    else:
        raise ValueError(f"unknown predictor mode {mode!r}")
    z_hat_J = VKJ @ f            # (NJ,)
    return z_hat_J, f


def predictor_matrix(VK: np.ndarray, us_idx, jp_idx,
                     mode: str = "paper",
                     sigma2: float | None = None) -> np.ndarray:
    """The rank<=K linear map from z_{U,t} to zhat_{J,t+1}.

    ``mode="paper"`` (default) is the paper's eq. (21) verbatim::

        B^(K)_t = V^(K)_{J,t} V^(K)_{U,t}^T

    ``mode="ridge"`` is the general best linear predictor of Prop. 2::

        B_t = V_J V_U^T (V_U V_U^T + sigma^2 I)^-1
            = V_J (V_U^T V_U + sigma^2 I_K)^-1 V_U^T     (push-through)

    The two agree only when ``V_U^T V_U = I_K``, which the proof of Prop. 2
    assumes (it is used in the Woodbury step) but which a *block* of a joint
    PCA eigenbasis never satisfies: for the paper's prior directions the U.S.
    block has Gram ``[[0.39, 0.49, 0], [0.49, 0.61, 0], [0, 0, 0.50]]``.
    Because the strategy ranks Japanese names cross-sectionally, the K x K
    reweighting is *not* absorbed as a monotone rescaling -- it changes the
    book.  ``"paper"`` stays the default so the reproduction is faithful;
    ``"ridge"`` is the variant that matches eq. (25).

    The second form is used below: it needs only a K x K solve, and adding
    sigma^2 keeps it stable where ``V_U^T V_U`` is near-singular (the plain
    whitening ``V_J (V_U^T V_U)^-1 V_U^T`` is not, with observed condition
    numbers of order 1e5).
    """
    VKU, VKJ = VK[us_idx, :], VK[jp_idx, :]
    if mode == "paper":
        return VKJ @ VKU.T
    if mode == "ridge":
        if sigma2 is None:
            raise ValueError("mode='ridge' requires sigma2")
        K = VK.shape[1]
        return VKJ @ np.linalg.solve(VKU.T @ VKU + sigma2 * np.eye(K), VKU.T)
    raise ValueError(f"unknown predictor mode {mode!r}")


# ---------------------------------------------------------------------------
# 7. Long/short portfolio (eqs. 3-7)
# ---------------------------------------------------------------------------
def long_short_weights(signal: np.ndarray, q: float = 0.3) -> np.ndarray:
    """Equal-weight long top-q / short bottom-q (eqs. 3-6).

    Weights satisfy  sum w = 0  and  sum |w| = 2.
    """
    n = len(signal)
    nq = max(1, int(np.floor(q * n)))
    order = np.argsort(signal)
    short = order[:nq]
    long = order[-nq:]
    w = np.zeros(n)
    w[long] = 1.0 / len(long)
    w[short] = -1.0 / len(short)
    return w


def double_sort_weights(sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
    """2x2 double sort (Section 4.3, DOUBLE).

    Median-split each signal into High/Low.  Long  High_a & High_b,
    short Low_a & Low_b, equal weight, dollar neutral.  If either
    intersection is empty the book stays flat so that sum(w) = 0 holds
    on every day, matching the neutrality of eq. (6).
    """
    ma, mb = np.median(sig_a), np.median(sig_b)
    high = (sig_a >= ma) & (sig_b >= mb)
    low = (sig_a < ma) & (sig_b < mb)
    w = np.zeros(len(sig_a))
    if high.any() and low.any():
        w[high] = 1.0 / high.sum()
        w[low] = -1.0 / low.sum()
    return w


# ---------------------------------------------------------------------------
# 8. Performance metrics (eqs. 27-30)
# ---------------------------------------------------------------------------
def performance_metrics(returns, periods_per_year: int = 252) -> dict:
    """Annualised return / risk / ratio / max-drawdown.

    The paper writes the annualisation factor as 12 (eqs. 27-28, monthly
    convention).  Here it is exposed as ``periods_per_year`` so a daily
    open-to-close backtest can use 252.  All values are returned in percent
    except R/R which is a pure ratio.
    """
    R = np.asarray(returns, dtype=float)
    R = R[~np.isnan(R)]
    if len(R) == 0:
        return dict(AR=np.nan, RISK=np.nan, RR=np.nan, MDD=np.nan, N=0)
    AR = periods_per_year * R.mean()
    RISK = np.sqrt(periods_per_year) * R.std(ddof=1)
    RR = AR / RISK if RISK > 0 else np.nan
    W = np.cumprod(1.0 + R)
    peak = np.maximum.accumulate(W)
    MDD = (W / peak - 1.0).min()
    return dict(AR=AR * 100, RISK=RISK * 100, RR=RR, MDD=MDD * 100, N=len(R))


# ---------------------------------------------------------------------------
# 9. Per-day estimation universe
# ---------------------------------------------------------------------------
def select_subuniverse(window: np.ndarray, today: np.ndarray,
                       us_idx, jp_idx, C0: np.ndarray, K: int = 3):
    """Restrict one day's joint PCA to the assets that actually have data.

    The Japanese block must have a complete estimation window -- without it
    there is nothing to predict.  U.S. tickers that lack one (XLRE before its
    2015-10 inception, XLC before 2018-06) are dropped from *that day's*
    joint PCA instead of killing the day outright, which is what the paper's
    per-ticker sample sizes imply (Table 1: 2590 rows for most tickers, 2409
    for XLRE, 1758 for XLC).

    Only the U.S. block of ``today`` is examined: the signal reads the window
    ``W_t = {t-L, ..., t-1}`` plus the U.S. close-to-close of day t, and never
    the Japanese return of day t itself.

    Returns ``(sel, us_idx_sub, jp_idx_sub, C0_sub)`` where ``sel`` indexes the
    kept assets in [U.S. block, Japan block] order.  Raises ``ValueError`` when
    the day is unusable.

    ``C0`` is sub-selected rather than rebuilt: it is diagonal-normalised, so a
    principal submatrix of ``Delta^-1/2 C_raw Delta^-1/2`` equals what eq. (12)
    would produce from the restricted ``C_raw`` -- unit diagonal and rank <= K0
    are both preserved.
    """
    us_idx = np.asarray(us_idx, dtype=int)
    jp_idx = np.asarray(jp_idx, dtype=int)

    win_ok = np.isfinite(window).all(axis=0)
    if not win_ok[jp_idx].all():
        raise ValueError("incomplete Japanese estimation window")

    us_keep = us_idx[win_ok[us_idx] & np.isfinite(today)[us_idx]]
    if len(us_keep) < K:
        raise ValueError(
            f"only {len(us_keep)} U.S. tickers available (need at least K={K})")

    sel = np.concatenate([us_keep, jp_idx])
    us_idx_sub = np.arange(len(us_keep))
    jp_idx_sub = np.arange(len(us_keep), len(sel))
    return sel, us_idx_sub, jp_idx_sub, C0[np.ix_(sel, sel)]


# ---------------------------------------------------------------------------
# 10. Single-day signal helper (used by both scripts)
# ---------------------------------------------------------------------------
def compute_signal_for_day(rcc_window: np.ndarray,
                           rcc_today_us: np.ndarray,
                           us_idx, jp_idx,
                           C0: np.ndarray,
                           lam: float = 0.9,
                           K: int = 3,
                           mode: str = "paper"):
    """Run the full PCA pipeline for one rebalancing day.

    Parameters
    ----------
    rcc_window   : (L, N) close-to-close returns of the estimation window W_t.
    rcc_today_us : (NU,)  close-to-close U.S. returns on the signal day t.
    us_idx,jp_idx: integer index arrays into the N-asset axis.
    C0           : subspace prior (set lam=0 to recover plain PCA).
    mode         : predictor variant, see :func:`predictor_matrix`.

    Returns a dict with the standardised prediction, factor scores,
    eigen-spectrum and the top-K eigenvectors.
    """
    Z, mu, sig = standardize_window(rcc_window)
    C_t = correlation_from_Z(Z)
    VK, evals = regularized_pca(C_t, C0, lam=lam, K=K)

    mu_us = mu[us_idx]
    sig_us = sig[us_idx]
    z_U = (rcc_today_us - mu_us) / sig_us             # eq. (17)
    sigma2 = residual_variance(evals, K)
    z_hat_J, f = leadlag_signal(VK, us_idx, jp_idx, z_U,
                                mode=mode, sigma2=sigma2)

    return dict(z_hat_J=z_hat_J, f=f, evals=evals, VK=VK,
                z_U=z_U, C_t=C_t, sigma2=sigma2, mode=mode)


# ---------------------------------------------------------------------------
# 11. Full backtest loop
# ---------------------------------------------------------------------------
def run_backtest(rcc: pd.DataFrame,
                 roc_J: pd.DataFrame,
                 tickers,
                 jp_tickers,
                 C0: np.ndarray,
                 L: int = 60,
                 lam: float = 0.9,
                 K: int = 3,
                 q: float = 0.3,
                 methods=("MOM", "PCA_PLAIN", "PCA_SUB", "DOUBLE"),
                 mode: str = "paper"):
    """Walk forward day by day and accumulate strategy returns.

    For each day t (with a full window behind it) the U.S. close-to-close
    shock observed at t is turned into a prediction of the Japanese
    open-to-close return realised at t+1; portfolios are formed from that
    prediction and marked to the realised roc_J at t+1.

    Days are kept whenever the Japanese block has a full window and at least
    K U.S. tickers do (:func:`select_subuniverse`), so the pre-inception NaN
    of XLRE and XLC shrinks that day's U.S. cross-section instead of dropping
    the day -- the same rule ``realtime_run.snapshot_at`` applies.

    Returns
    -------
    DataFrame indexed by the t+1 trade date with one column of daily
    strategy returns per method.
    """
    tickers = list(tickers)
    jp_tickers = list(jp_tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    us_idx = np.array([idx[t] for t in tickers if t in US_TICKERS])
    jp_idx = np.array([idx[t] for t in jp_tickers])

    rcc_v = rcc[tickers].values
    # align roc_J columns to jp order
    roc_v = roc_J[jp_tickers].values
    dates = rcc.index

    out = {m: [] for m in methods}
    trade_dates = []

    T = len(rcc_v)
    for t in range(L, T - 1):
        window = rcc_v[t - L:t]                 # W_t = {t-L,...,t-1}
        realised = roc_v[t + 1]                 # next-day open-to-close
        if not np.isfinite(realised).all():
            continue
        try:
            sel, us_sub, jp_sub, C0_sub = select_subuniverse(
                window, rcc_v[t], us_idx, jp_idx, C0, K=K)
        except ValueError:
            continue

        window = window[:, sel]
        today_us = rcc_v[t, sel[us_sub]]

        Z, mu, sig = standardize_window(window)
        C_t = correlation_from_Z(Z)

        # momentum signal (eq. 31): trailing-mean of Japanese cc returns
        mom = mu[jp_sub]

        # regularised / plain PCA signals
        z_U = (today_us - mu[us_sub]) / sig[us_sub]

        sig_reg = None
        if any(m in methods for m in ("PCA_PLAIN", "PCA_SUB", "DOUBLE")):
            if "PCA_PLAIN" in methods:
                VKp, ev_p = regularized_pca(C_t, C0_sub, lam=0.0, K=K)
                zhat_plain, _ = leadlag_signal(
                    VKp, us_sub, jp_sub, z_U, mode=mode,
                    sigma2=residual_variance(ev_p, K))
            if any(m in methods for m in ("PCA_SUB", "DOUBLE")):
                VKr, ev_r = regularized_pca(C_t, C0_sub, lam=lam, K=K)
                zhat_reg, _ = leadlag_signal(
                    VKr, us_sub, jp_sub, z_U, mode=mode,
                    sigma2=residual_variance(ev_r, K))
                sig_reg = zhat_reg

        trade_dates.append(dates[t + 1])
        for m in methods:
            if m == "MOM":
                w = long_short_weights(mom, q=q)
            elif m == "PCA_PLAIN":
                w = long_short_weights(zhat_plain, q=q)
            elif m == "PCA_SUB":
                w = long_short_weights(sig_reg, q=q)
            elif m == "DOUBLE":
                w = double_sort_weights(mom, sig_reg)
            else:
                raise ValueError(f"unknown method {m}")
            out[m].append(float(w @ realised))

    return pd.DataFrame(out, index=pd.Index(trade_dates, name="date"))
