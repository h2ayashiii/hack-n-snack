# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two independent Python implementations of the lead-lag strategy described in the paper *"Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA"* (Nakagawa et al., SIG-FIN-036-13).

- **`lead_lag_strategy/`** — Structured package implementation (v1)
- **`lead_lag_strategy_v2/`** — Flat-file implementation (v2)

Both implement the same algorithm but differ in architecture. Neither has a test suite.

---

## lead_lag_strategy/ (v1 — package)

### Setup

```bash
cd lead_lag_strategy
uv sync
source .venv/bin/activate
```

### Run

```bash
# Backtest (reproduces Table 2/3/4 and Figure 2 from the paper)
python -m lead_lag_strategy.backtest
python -m lead_lag_strategy.backtest --start 2015-01-01 --end 2025-12-31 --no-plot
python -m lead_lag_strategy.backtest --force-refresh   # re-download yfinance data

# Live signal (today's positions for next JP session)
python -m lead_lag_strategy.live
python -m lead_lag_strategy.live --date 2025-06-26 --strategy PCA_SUB
python -m lead_lag_strategy.live --json --propagation
```

Outputs go to `results/` (gitignored): `performance_summary.csv`, `ff3_regression.csv`, `carhart4_regression.csv`, `cumulative_returns.png`, `strategy_returns.csv`.

### Architecture

Data flows one-way through four layers:

```
data/fetcher.py          → downloads yfinance OHLCV, caches to data/cache/ohlcv.parquet
                           build_returns() → (us_cc, jp_cc, jp_oc, jp_oc_next)
model/pca.py             → SubspaceRegularisedPCA: fit_prior(), compute_signal()
                           builds C0 prior from 2010-2014, regularises C_t toward it
model/signal.py          → compute_all_signals(): rolls daily, returns DataFrame of
                           signals for MOM / PCA_PLAIN / PCA_SUB / DOUBLE
model/portfolio.py       → compute_all_strategy_returns(): signals → weights → returns
evaluation/metrics.py    → summary_table(), factor_regression_table() (FF3/Carhart4
                           with Newey-West HAC), cumulative_wealth()
```

`config.py` is the single source of truth for all constants (tickers, λ=0.9, K=3, L=60, q=0.3, date ranges).

**Key timing invariant**: signal on day *t* (US close-to-close) predicts JP open-to-close on day *t+1*. In `fetcher.py` this is enforced via `jp_oc_next = jp_oc.shift(-1)`.

---

## lead_lag_strategy_v2/ (v2 — flat files)

### Setup

```bash
cd lead_lag_strategy_v2
uv sync             # base packages
uv sync --extra live  # + yfinance for live data
source .venv/bin/activate
```

### Run

```bash
# Logic verification on synthetic data
python verify_logic.py                        # → output/verify_logic.png
python verify_logic.py --seed 1 --days 2000

# Real-time signal (single date or range)
python realtime_run.py                                     # latest date
python realtime_run.py --date 2024-11-01                   # specific date
python realtime_run.py --start-date 2024-10-01 --end-date 2024-11-30  # range → heatmap
python realtime_run.py --no-chart --offline                # text only, no network
python realtime_run.py --watch 300                         # refresh every 300s
```

Outputs go to `output/` (gitignored). Single-date: `output/realtime_signal_YYYY-MM-DD.png`. Range: `output/realtime_signal_START_END.png` (RdYlGn heatmap, ▲=LONG / ▼=SHORT).

### Architecture

Everything lives in two files:

- **`common.py`** — all math: return transforms, rolling standardisation, C0 prior construction (Gram-Schmidt), regularised PCA (`(1-λ)C_t + λC0`), lead-lag signal, long-short weights, performance metrics, full backtest loop (`run_backtest`)
- **`realtime_run.py`** — data acquisition (yfinance with synthetic fallback), prior estimation, `snapshot_at(date)` / `snapshot_range(start, end)`, text and chart output
- **`verify_logic.py`** — synthetic data generation from the idealized factor model; verifies Propositions 1–2 and reproduces Table 2

---

## Algorithm Summary

**Core idea**: US sectors close before Japan opens. US close-to-close return on day *t* predicts JP open-to-close return on day *t+1* via shared global factors.

**Key equation** (propagation matrix): `B_t^(K) = V_J^(K) (V_U^(K))^T`  
where `V^(K)` are the top-K eigenvectors of the regularised correlation matrix  
`C_reg_t = (1-λ)C_t + λC0` (λ=0.9 shrinks toward the 2010–2014 prior).

**Four strategies**: MOM (momentum), PCA_PLAIN (λ=0), PCA_SUB (λ=0.9, proposed), DOUBLE (2×2 median sort on MOM × PCA_SUB). Paper result: PCA_SUB best on R/R and MDD.
