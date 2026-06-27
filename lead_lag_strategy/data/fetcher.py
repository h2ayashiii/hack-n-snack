"""
Data acquisition: download and cache US/Japan sector ETF OHLCV data.
Uses yfinance; results are cached as parquet files under data/cache/.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from lead_lag_strategy.config import ALL_TICKERS, JP_TICKERS, US_TICKERS

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_OHLCV_CACHE = CACHE_DIR / "ohlcv.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Raw download
# ─────────────────────────────────────────────────────────────────────────────

def download_ohlcv(
    start: str = "2009-12-01",
    end: str = "2025-12-31",
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily Open/Close prices for all tickers.

    Returns a MultiIndex-column DataFrame: (field, ticker)
    where field ∈ {'Open', 'Close'}.
    """
    if use_cache and _OHLCV_CACHE.exists() and not force_refresh:
        df = pd.read_parquet(_OHLCV_CACHE)
        # Extend cache if needed
        last_date = df.index[-1]
        if pd.Timestamp(end) > last_date + pd.Timedelta(days=3):
            df = _merge_new(df, start=str(last_date.date()), end=end)
            df.to_parquet(_OHLCV_CACHE)
        return df

    df = _download(start=start, end=end)
    if use_cache:
        df.to_parquet(_OHLCV_CACHE)
    return df


def _download(start: str, end: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ALL_TICKERS,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    # Keep only Open and Close
    df = raw[["Open", "Close"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


def _merge_new(existing: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    new = _download(start=start, end=end)
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Return construction
# ─────────────────────────────────────────────────────────────────────────────

def build_returns(
    ohlcv: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build the two return series required by the paper.

    Returns a dict with keys:
      'us_cc'  – US sector close-to-close returns  (eq. 1, used for estimation)
      'jp_cc'  – JP sector close-to-close returns  (used for Cfull estimation)
      'jp_oc'  – JP sector open-to-close returns   (eq. 2, used for strategy eval)
    """
    close = ohlcv["Close"]
    open_ = ohlcv["Open"]

    us_cc = close[US_TICKERS].pct_change()
    jp_cc = close[JP_TICKERS].pct_change()
    jp_oc = close[JP_TICKERS] / open_[JP_TICKERS] - 1

    if start is not None:
        us_cc = us_cc.loc[start:]
        jp_cc = jp_cc.loc[start:]
        jp_oc = jp_oc.loc[start:]
    if end is not None:
        us_cc = us_cc.loc[:end]
        jp_cc = jp_cc.loc[:end]
        jp_oc = jp_oc.loc[:end]

    # Drop days where ALL tickers are NaN (weekends etc. that slipped through)
    all_cc = pd.concat([us_cc, jp_cc], axis=1)
    valid = all_cc.dropna(how="all").index

    return {
        "us_cc": us_cc.loc[valid],
        "jp_cc": jp_cc.loc[valid],
        "jp_oc": jp_oc.loc[valid],
    }


def align_us_jp(
    us_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    jp_cc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align dates so that US close-to-close return on day t maps to
    JP open-to-close return on day t+1.

    Returns (us_cc_aligned, jp_oc_next, jp_cc_aligned) on the
    intersection of trading dates. jp_oc_next[t] = JP OC return
    on the business day following t.
    """
    # Common dates where both US and JP have at least some data
    common = us_cc.index.intersection(jp_cc.index)
    us_cc = us_cc.loc[common].copy()
    jp_cc = jp_cc.loc[common].copy()
    jp_oc = jp_oc.loc[common].copy()

    # Build shifted JP OC: jp_oc_next[t] is the OC return on day t+1
    jp_oc_next = jp_oc.shift(-1)

    # Drop last row (no next-day OC available) and any date with all NaN
    us_cc     = us_cc.iloc[:-1]
    jp_cc     = jp_cc.iloc[:-1]
    jp_oc_next = jp_oc_next.iloc[:-1]

    return us_cc, jp_oc_next, jp_cc
