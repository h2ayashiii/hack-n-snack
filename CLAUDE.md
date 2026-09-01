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

# Daily report: e-mail and/or Discord (see .github/workflows/daily-signal.yml)
python daily_report.py --dry-run                           # render only → .eml + .html and/or discord_payload.json
python daily_report.py --date 2024-11-01 --offline --dry-run --channels discord
python daily_report.py                                     # send via e-mail (needs SMTP_* env)
python daily_report.py --channels discord                  # post to Discord (needs DISCORD_WEBHOOK_URL)
python daily_report.py --channels email,discord             # both

# Post-close review: how the published book actually did (see .github/workflows/jp-close-review.yml)
python review_report.py --dry-run                          # render only → .eml + .html and/or discord_review.json
python review_report.py --date 2024-11-01 --offline --dry-run
python review_report.py --channels discord                 # post the review to Discord
```

Outputs go to `output/` (gitignored). Single-date: `output/realtime_signal_YYYY-MM-DD.png`. Range: `output/realtime_signal_START_END.png` (RdYlGn heatmap, ▲=LONG / ▼=SHORT). Daily report: `output/daily_report_YYYY-MM-DD.{eml,html}` and/or `output/discord_payload_YYYY-MM-DD.json`. Post-close review: `output/review_report_YYYY-MM-DD.{eml,html}`, `output/discord_review_YYYY-MM-DD.json`, `output/review_YYYY-MM-DD.png`, `output/review_calc_YYYY-MM-DD.html` (P&L calculator, see below).

### Architecture

- **`common.py`** — all math: return transforms, rolling standardisation, C0 prior construction (Gram-Schmidt), regularised PCA (`(1-λ)C_t + λC0`), lead-lag signal, long-short weights, performance metrics, full backtest loop (`run_backtest`)
- **`realtime_run.py`** — data acquisition (yfinance with synthetic fallback), prior estimation, `snapshot_at(date)` / `snapshot_range(start, end)`, text and chart output
- **`verify_logic.py`** — synthetic data generation from the idealized factor model; verifies Propositions 1–2 and reproduces Table 2
- **`daily_report.py`** — reuses `realtime_run`'s `get_data` / `build_prior` / `snapshot_at` (signal computed once regardless of channel), then dispatches to one or more independently-configured delivery channels selected via `--channels` / `$REPORT_CHANNELS` (default `email`):
  - **email** — multipart HTML+text (inline chart, trailing 20-session realised performance) via SMTP. Config: `SMTP_HOST/PORT/USER/PASSWORD/STARTTLS`, `REPORT_FROM/TO`; defaults are `@example.com` mock addresses, unset `SMTP_HOST` falls back to writing the message to disk.
  - **discord** — a single embed (top-q/bottom-q book, factor scores, trailing performance, chart attached) POSTed as `multipart/form-data` to a webhook URL, stdlib `urllib` only (no `requests` dependency). Config: `DISCORD_WEBHOOK_URL`; unset falls back to writing the JSON payload to disk. The webhook URL is a bearer credential — never logged in full (`mask_webhook`). `post_discord()` sets an explicit `User-Agent`: discord.com sits behind Cloudflare, which blocks urllib's default `Python-urllib/x.y` UA as a bot signature (`403`, Cloudflare error 1010) before the request reaches Discord.
- **`review_report.py`** — post-close counterpart to `daily_report.py`: scores the published book against the JP session that just closed. Reuses `daily_report`'s delivery plumbing verbatim (`resolve_channels`, `smtp_config`, `assemble_message`, `post_discord`, the dry-run writers), so it needs no configuration of its own. Per predicted ETF it reports open, close, the move in yen, the open-to-close %, the contribution `w_j × roc_j`, and whether the direction was right; at book level the return (= long leg − short leg), direction hit rate, Spearman rank IC vs realised, the TOPIX-17 equal-weight mean, and a trailing `--history-days` (default 20) summary re-scored the same way.

**P&L calculator** (`build_calculator_html` / `save_calculator_html`, `--no-calculator` to disable): a standalone, self-contained HTML+vanilla-JS page (no external assets, no network calls) that lets whoever opens it type a principal and see it scaled against two already-computed numbers — the session's *actual* book return (exact, since everything has already closed) and, for context, the trailing `--history-days` mean daily book return from `review_history()`, explicitly labelled as a track-record estimate rather than a forecast (the model ranks direction, not the size of the move, so there is no per-session return magnitude to scale against). Delivered as `output/review_calc_YYYY-MM-DD.html`: attached to the e-mail as a real MIME attachment rather than inlined into the HTML body (mail clients strip `<script>` from HTML e-mails), and attached to the Discord message as an extra `files[]` part alongside the chart (`post_discord`'s `extra_files` parameter — plain downloads, not referenced by the embed) with a footer note pointing at it. Either way the page only works once downloaded and opened in a browser; Discord/e-mail cannot execute JS inline.

**Review timing/pairing**: the session being reviewed is *not yet* a day both markets traded (the U.S. has not closed), so `fetch_prices`'s common-day filter drops it. `review_report.jp_session_frames()` lays a Japan-only download (`realtime_run.fetch_jp_prices`, same `threads=False` + retry treatment) over the joint frames to bring it back, degrading to the common-day frames if that fetch fails. The book being scored is the one from the last common day *strictly before* the session — exactly what `daily_report.py` publishes off `rcc.index[-1]`. When the U.S. was shut the day before (~8×/year) no new book was published, so `stale_book` is set and every renderer says the previous book was carried.

**Real-data handling** (`realtime_run.fetch_prices` / `build_prior`): only days both markets traded are kept; pre-inception NaN is preserved (XLRE 2015-10, XLC 2018-06) so the 2010–2014 prior window survives; U.S. tickers lacking a full estimation window are dropped from that day's joint PCA. `fetch_prices` calls `yf.download(..., threads=False)` — yfinance's default threaded mode has concurrent workers write to a shared sqlite tz-cache (`~/.cache/py-yfinance/`), which occasionally raises `database is locked` for a single ticker without the caller seeing an exception (it just shows up as a missing-data gap downstream); serial fetching plus a tail-completeness check + retry (3 attempts, backoff) closes that failure mode.

### Scheduled run

`.github/workflows/daily-signal.yml` runs `daily_report.py` at `30 22 * * 1-5` UTC — after the U.S. close, ~1.5h before the JP open. Scheduled runs pass `--require-live --skip-if-stale 0` so a synthetic or stale book is never sent; `workflow_dispatch` exposes `date` / `dry_run` / `offline` / `channels` inputs. Channel selection comes from the `REPORT_CHANNELS` repo variable, but **this workflow's own default (when that variable is unset) is `discord`**, not `daily_report.py`'s script-level default of `email` — so setting only the `DISCORD_WEBHOOK_URL` secret is enough to turn on daily Discord delivery. Rendered output is always uploaded as an artifact.

`.github/workflows/jp-close-review.yml` runs `review_report.py` at `0 7 * * 1-5` UTC — 16:00 JST, 1h after the JP close (slack for Yahoo to settle the daily bars) and well before the next U.S. open. Same secrets/variables and same `discord` default as the signal job, so enabling it needs no extra configuration; scheduled runs pass `--require-live --skip-if-stale 0`, and at 07:00 UTC the runner's date already equals the JST session date so that check is exact.

Note GitHub disables cron workflows after 60 days of repo inactivity.

---

## Algorithm Summary

**Core idea**: US sectors close before Japan opens. US close-to-close return on day *t* predicts JP open-to-close return on day *t+1* via shared global factors.

**Key equation** (propagation matrix): `B_t^(K) = V_J^(K) (V_U^(K))^T`  
where `V^(K)` are the top-K eigenvectors of the regularised correlation matrix  
`C_reg_t = (1-λ)C_t + λC0` (λ=0.9 shrinks toward the 2010–2014 prior).

**Four strategies**: MOM (momentum), PCA_PLAIN (λ=0), PCA_SUB (λ=0.9, proposed), DOUBLE (2×2 median sort on MOM × PCA_SUB). Paper result: PCA_SUB best on R/R and MDD.
