"""
review_report.py
================
Post-close counterpart to ``daily_report.py``.

``daily_report.py`` runs *before* the Japanese open and publishes the
book the model wants for the coming session.  This script runs *after*
the Japanese close and reports what that book actually did: for every
predicted ETF, the open, the close, the move in yen and in percent, and
whether the predicted direction was right.

Timing (the paper's convention, eqs. 1-2):

    day t     U.S. close-to-close  ->  signal, published by daily_report.py
    day t+1   Japanese open-to-close  ->  realised, reviewed here

so the natural schedule is just after 15:00 JST (06:00 UTC); see
``.github/workflows/jp-close-review.yml``.

Delivery is shared with ``daily_report.py`` -- same channels, same
configuration, same secrets:

    --channels email            e-mail only
    --channels discord          Discord webhook only
    --channels email,discord    both

    REPORT_CHANNELS   env var equivalent of --channels (CLI wins)
    SMTP_*, REPORT_FROM, REPORT_TO      see daily_report.py
    DISCORD_WEBHOOK_URL                 see daily_report.py

With no SMTP_HOST / DISCORD_WEBHOOK_URL the script falls back to a dry
run and writes the rendered report to the output directory.

Run
---
    # Render only -- writes output/review_report_YYYY-MM-DD.{eml,html}
    # and/or output/discord_review_YYYY-MM-DD.json
    python review_report.py --dry-run

    # Review a specific past Japanese session, no network
    python review_report.py --date 2024-11-01 --offline --dry-run

    # Post the review of the session that just closed to Discord
    python review_report.py --channels discord
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
import daily_report as D
from realtime_run import (build_prior, ensure_output_dir, fetch_jp_prices,
                          get_data, snapshot_at)


# ---------------------------------------------------------------------------
# Japanese session prices
# ---------------------------------------------------------------------------
def jp_session_frames(open_, close, allow_network=True):
    """Open/close frames for the Japanese ETFs, as fresh as possible.

    The joint frames coming out of :func:`realtime_run.get_data` are
    filtered down to the days both markets traded, which is right for the
    joint PCA but drops the session this script exists to review: at
    15:00 JST on day t+1 the U.S. has not closed yet, so day t+1 is not a
    common day.  When the network is available a Japan-only download is
    laid over the joint frames to bring that session back.

    Returns ``(jp_open, jp_close, extended)``.
    """
    jp_open = open_[C.JP_TICKERS].copy()
    jp_close = close[C.JP_TICKERS].copy()
    if not allow_network:
        return jp_open, jp_close, False
    try:
        fresh_open, fresh_close = fetch_jp_prices()
    except Exception as e:
        print(f"[warn] Japan-only refresh failed ({type(e).__name__}: {e}); "
              "falling back to the common-day frames, which may not yet "
              "include today's session.", file=sys.stderr)
        return jp_open, jp_close, False
    # the fresh download wins wherever it has data, and contributes the
    # trailing session the joint frames are missing
    jp_open = fresh_open.combine_first(jp_open).sort_index()
    jp_close = fresh_close.combine_first(jp_close).sort_index()
    return jp_open[C.JP_TICKERS], jp_close[C.JP_TICKERS], True


# ---------------------------------------------------------------------------
# Review computation
# ---------------------------------------------------------------------------
def review_at(rcc, C0, jp_open, jp_close, review_date, source,
              L=60, lam=0.9, K=3, q=0.3):
    """Score the published book against one realised Japanese session.

    ``review_date`` is the Japanese session that has just closed (day
    t+1).  The book being scored is the one implied by the most recent
    U.S. close *strictly before* it -- exactly what ``daily_report.py``
    would have sent out, since it publishes off ``rcc.index[-1]``.
    """
    jp_dates = jp_close.index
    review_ts = pd.Timestamp(review_date)
    pos = jp_dates.searchsorted(review_ts, side="right") - 1
    if pos < 0:
        raise ValueError(f"No Japanese session on or before {review_date}")
    review_ts = jp_dates[pos]

    spos = rcc.index.searchsorted(review_ts, side="left") - 1
    if spos < 0:
        raise ValueError(f"No U.S. close before {review_ts.date()}")
    signal_ts = rcc.index[spos]

    s = snapshot_at(rcc, C0, signal_ts, source, L=L, lam=lam, K=K, q=q)

    o = jp_open.loc[review_ts, C.JP_TICKERS].to_numpy(dtype=float)
    c = jp_close.loc[review_ts, C.JP_TICKERS].to_numpy(dtype=float)
    if not (np.isfinite(o).all() and np.isfinite(c).all()):
        missing = [t for t, ok in zip(C.JP_TICKERS,
                                      np.isfinite(o) & np.isfinite(c)) if not ok]
        raise ValueError(f"Incomplete Japanese prices on {review_ts.date()} "
                         f"for {', '.join(missing)}")
    roc = c / o - 1.0                                   # eq. (2)

    w, z = s["w"], s["z_hat"]
    contrib = w * roc                                   # per-name P&L share
    contrib[contrib == 0] = 0.0                         # kill signed zeros
    longs = np.where(w > 0)[0]
    shorts = np.where(w < 0)[0]
    book = np.concatenate([longs, shorts])
    hits = np.sign(w[book]) == np.sign(roc[book])

    # The book is only "fresh" if this session is the first Japanese one
    # after the signal.  When the U.S. was shut the day before (~8 times a
    # year) no new book was published and yesterday's is simply carried.
    later = jp_dates[jp_dates > signal_ts]
    stale_book = bool(len(later)) and later[0] != review_ts

    return dict(
        review_date=review_ts, signal_date=s["signal_date"], source=source,
        stale_book=stale_book,
        z_hat=z, w=w, roc=roc, open_px=o, close_px=c, change=c - o,
        contrib=contrib,
        port_return=float(contrib.sum()),
        long_leg=float(roc[longs].mean()) if len(longs) else float("nan"),
        short_leg=float(roc[shorts].mean()) if len(shorts) else float("nan"),
        market=float(roc.mean()),
        hit_rate=float(hits.mean()) if len(book) else float("nan"),
        n_hits=int(hits.sum()), n_book=int(len(book)),
        ic=float(spearmanr(z, roc)[0]),
        longs=longs, shorts=shorts,
        K=K, lam=lam, L=L, q=q, us_used=s.get("us_used", C.US_TICKERS),
    )


def review_history(rcc, C0, jp_open, jp_close, review_ts, source, days=20,
                   **kw):
    """Re-score the last ``days`` Japanese sessions up to ``review_ts``.

    Uses the same live pairing as :func:`review_at` (each session against
    the book that was actually in force for it) rather than the
    backtest's common-day pairing, so the series ends on the session
    being reported instead of on the last day both markets traded.
    """
    dates = [d for d in jp_close.index if d <= pd.Timestamp(review_ts)]
    dates = dates[-days:]
    rows = []
    for d in dates:
        try:
            r = review_at(rcc, C0, jp_open, jp_close, d, source, **kw)
        except (ValueError, KeyError):
            continue
        rows.append((d, r["port_return"], r["hit_rate"]))
    if not rows:
        return None
    idx = pd.DatetimeIndex([r[0] for r in rows])
    ret = pd.Series([r[1] for r in rows], index=idx)
    hit = pd.Series([r[2] for r in rows], index=idx)
    return dict(n=len(ret), returns=ret,
                cumulative=float((1.0 + ret).prod() - 1.0),
                mean=float(ret.mean()),
                win_rate=float((ret > 0).mean()),
                hit_rate=float(hit.mean()),
                first=idx[0], last=idx[-1])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _order(r):
    """Book first (longs by conviction, then shorts), then the rest."""
    longs = sorted(r["longs"], key=lambda i: -r["z_hat"][i])
    shorts = sorted(r["shorts"], key=lambda i: r["z_hat"][i])
    return list(longs), list(shorts)


def _side(i, r):
    return "LONG" if r["w"][i] > 0 else ("SHORT" if r["w"][i] < 0 else "-")


def _hit(i, r):
    if r["w"][i] == 0:
        return ""
    return "OK" if np.sign(r["w"][i]) == np.sign(r["roc"][i]) else "MISS"


def subject_line(r) -> str:
    date = pd.Timestamp(r["review_date"]).date()
    prefix = "[SYNTHETIC] " if not r["live"] else ""
    return (f"{prefix}[Lead-Lag PCA_SUB] {date} JP close review"
            f"  |  book {100 * r['port_return']:+.2f}%"
            f"  hit {r['n_hits']}/{r['n_book']}")


def render_text(r, hist=None) -> str:
    line = "=" * 78
    out = [line,
           " SUBSPACE-REGULARIZED PCA LEAD-LAG  --  JP CLOSE REVIEW",
           line,
           f" reviewed session : {pd.Timestamp(r['review_date']).date()}"
           f"  (Japanese open-to-close)",
           f" from signal day  : {pd.Timestamp(r['signal_date']).date()}"
           f"  (U.S. close-to-close)",
           f" data source      : {r['source']}",
           f" params           : L={r['L']}  lambda={r['lam']}"
           f"  K={r['K']}  q={r['q']}",
           line]

    if not r["live"]:
        out += ["",
                " *** WARNING: built on SYNTHETIC fallback data, not live"
                " prices. ***"]
    if r["stale_book"]:
        out += ["",
                " NOTE: no new signal was published for this session (the"
                " U.S. market",
                "       was shut the day before), so the previous book was"
                " carried."]

    out += ["", " RESULT",
            f"   book return   : {100 * r['port_return']:+.3f}%"
            f"   (long leg {100 * r['long_leg']:+.3f}%"
            f" / short leg {100 * r['short_leg']:+.3f}%)",
            f"   TOPIX-17 mean : {100 * r['market']:+.3f}%"
            f"   (equal weight, all 17 sectors)",
            f"   direction hit : {r['n_hits']}/{r['n_book']}"
            f"  ({100 * r['hit_rate']:.0f}%)",
            f"   rank IC       : {r['ic']:+.3f}"
            f"   (Spearman, zhat vs realised)"]

    hdr = (f"   {'ETF':<8s} {'sector':<30s} {'zhat':>7s} {'open':>9s}"
           f" {'close':>9s} {'chg':>8s} {'%':>8s} {'contrib':>8s}  hit")
    longs, shorts = _order(r)
    for title, group in (("LONG (top-q)", longs), ("SHORT (bottom-q)", shorts)):
        out += ["", f" {title}:", hdr]
        for i in group:
            tk = C.JP_TICKERS[i]
            out.append(
                f"   {tk:<8s} {C.JP_LABELS.get(tk, ''):<30s}"
                f" {r['z_hat'][i]:+7.3f} {r['open_px'][i]:9.1f}"
                f" {r['close_px'][i]:9.1f} {r['change'][i]:+8.1f}"
                f" {100 * r['roc'][i]:+7.2f}% {100 * r['contrib'][i]:+7.3f}%"
                f"  {_hit(i, r)}")

    out += ["", " All 17 sectors, ranked by realised open-to-close:", hdr]
    for i in np.argsort(r["roc"])[::-1]:
        tk = C.JP_TICKERS[i]
        out.append(
            f"   {tk:<8s} {C.JP_LABELS.get(tk, ''):<30s}"
            f" {r['z_hat'][i]:+7.3f} {r['open_px'][i]:9.1f}"
            f" {r['close_px'][i]:9.1f} {r['change'][i]:+8.1f}"
            f" {100 * r['roc'][i]:+7.2f}% {100 * r['contrib'][i]:+7.3f}%"
            f"  {_side(i, r):<5s} {_hit(i, r)}")

    if hist:
        out += ["", f" Trailing {hist['n']} reviewed sessions"
                    f" ({hist['first'].date()} .. {hist['last'].date()}):",
                f"   cumulative    : {100 * hist['cumulative']:+.2f}%",
                f"   mean daily    : {100 * hist['mean']:+.3f}%",
                f"   positive days : {100 * hist['win_rate']:.1f}%",
                f"   direction hit : {100 * hist['hit_rate']:.1f}%"]

    out += ["", line,
            " Research output only -- no investment advice."
            "  Costs and slippage are not modelled.",
            line]
    return "\n".join(out)


_TD = "padding:4px 10px;border-bottom:1px solid #eee"


def _html_rows(indices, r, with_side=False):
    rows = []
    for i in indices:
        tk = C.JP_TICKERS[i]
        roc = r["roc"][i]
        colour = "#1a7f37" if roc > 0 else ("#c11d2b" if roc < 0 else "#57606a")
        # the sector move is coloured by its own sign, the contribution by
        # whether it made the book money -- for a short those disagree
        con = r["contrib"][i]
        con_colour = ("#1a7f37" if con > 0 else
                      ("#c11d2b" if con < 0 else "#57606a"))
        hit = _hit(i, r)
        badge = ""
        if hit == "OK":
            badge = ('<span style="color:#1a7f37;font-weight:600">&#10003;</span>')
        elif hit == "MISS":
            badge = ('<span style="color:#c11d2b;font-weight:600">&#10007;</span>')
        side = ""
        if with_side:
            side = f'<td style="{_TD};font-size:12px;color:#57606a">{_side(i, r)}</td>'
        rows.append(
            f'<tr><td style="{_TD};font-family:monospace">{tk}</td>'
            f'<td style="{_TD}">{C.JP_LABELS.get(tk, "")}</td>'
            f'{side}'
            f'<td style="{_TD};text-align:right;font-family:monospace">'
            f'{r["z_hat"][i]:+.3f}</td>'
            f'<td style="{_TD};text-align:right;font-family:monospace">'
            f'{r["open_px"][i]:,.1f}</td>'
            f'<td style="{_TD};text-align:right;font-family:monospace">'
            f'{r["close_px"][i]:,.1f}</td>'
            f'<td style="{_TD};text-align:right;font-family:monospace;'
            f'color:{colour}">{r["change"][i]:+,.1f}</td>'
            f'<td style="{_TD};text-align:right;font-family:monospace;'
            f'color:{colour};font-weight:600">{100 * roc:+.2f}%</td>'
            f'<td style="{_TD};text-align:right;font-family:monospace;'
            f'color:{con_colour}">{100 * con:+.3f}%</td>'
            f'<td style="{_TD};text-align:center">{badge}</td></tr>')
    return "".join(rows)


def render_html(r, hist=None, chart_cid=None) -> str:
    date = pd.Timestamp(r["review_date"]).date()
    sig = pd.Timestamp(r["signal_date"]).date()
    head = ('font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;'
            'color:#1f2328;max-width:860px')
    th = ('padding:6px 10px;text-align:left;border-bottom:2px solid #d0d7de;'
          'font-size:12px;letter-spacing:.04em;color:#57606a')
    pnl = r["port_return"]
    pnl_colour = "#1a7f37" if pnl > 0 else ("#c11d2b" if pnl < 0 else "#57606a")

    banner = ""
    if not r["live"]:
        banner += ('<div style="background:#fff3cd;border:1px solid #f0c36d;'
                   'padding:12px;border-radius:6px;margin-bottom:16px">'
                   '<b>SYNTHETIC fallback data.</b> Live prices were '
                   'unavailable, so these are simulated results.</div>')
    if r["stale_book"]:
        banner += ('<div style="background:#eef4ff;border:1px solid #b6cff5;'
                   'padding:12px;border-radius:6px;margin-bottom:16px">'
                   'No new signal was published for this session &mdash; the '
                   'U.S. market was shut the day before, so the previous '
                   'book was carried into it.</div>')

    def stat(label, value, colour="#1f2328", note=""):
        return (f'<td style="padding:10px 16px;border:1px solid #d0d7de;'
                f'border-radius:6px">'
                f'<div style="font-size:11px;letter-spacing:.04em;'
                f'color:#57606a">{label}</div>'
                f'<div style="font-size:20px;font-weight:600;color:{colour};'
                f'font-family:monospace">{value}</div>'
                f'<div style="font-size:11px;color:#57606a">{note}</div></td>')

    headline = (
        '<table style="border-collapse:separate;border-spacing:8px 0;'
        'margin:8px 0 20px"><tr>'
        + stat("BOOK RETURN", f"{100 * pnl:+.2f}%", pnl_colour,
               "long &minus; short, equal weight")
        + stat("DIRECTION HIT", f"{r['n_hits']}/{r['n_book']}",
               note=f"{100 * r['hit_rate']:.0f}% of the book")
        + stat("RANK IC", f"{r['ic']:+.3f}", note="Spearman vs realised")
        + stat("TOPIX-17 MEAN", f"{100 * r['market']:+.2f}%",
               note="all 17, equal weight")
        + '</tr></table>')

    legs = (f'<p style="margin:0 0 16px;color:#57606a;font-size:13px">'
            f'Long leg <code>{100 * r["long_leg"]:+.3f}%</code> &nbsp;&nbsp; '
            f'Short leg <code>{100 * r["short_leg"]:+.3f}%</code> &nbsp;&nbsp; '
            f'Book return = long &minus; short.</p>')

    hist_html = ""
    if hist:
        hist_html = (
            f'<h3 style="margin:24px 0 8px">Trailing {hist["n"]} reviewed '
            f'sessions</h3>'
            f'<p style="margin:0;color:#57606a;font-size:13px">'
            f'{hist["first"].date()} &ndash; {hist["last"].date()}</p>'
            f'<ul style="line-height:1.6">'
            f'<li>cumulative: <code>{100 * hist["cumulative"]:+.2f}%</code></li>'
            f'<li>mean daily: <code>{100 * hist["mean"]:+.3f}%</code></li>'
            f'<li>positive days: <code>{100 * hist["win_rate"]:.1f}%</code></li>'
            f'<li>direction hit: <code>{100 * hist["hit_rate"]:.1f}%</code></li>'
            f'</ul>')

    chart_html = ""
    if chart_cid:
        chart_html = (f'<h3 style="margin:24px 0 8px">Predicted vs realised</h3>'
                      f'<img src="cid:{chart_cid}" style="max-width:100%" '
                      f'alt="predicted signal versus realised returns">')

    longs, shorts = _order(r)
    cols = ('<tr>'
            f'<th style="{th}">ETF</th><th style="{th}">SECTOR</th>'
            f'<th style="{th};text-align:right">&#7825;<sub>J</sub></th>'
            f'<th style="{th};text-align:right">OPEN</th>'
            f'<th style="{th};text-align:right">CLOSE</th>'
            f'<th style="{th};text-align:right">CHG</th>'
            f'<th style="{th};text-align:right">%</th>'
            f'<th style="{th};text-align:right">CONTRIB</th>'
            f'<th style="{th};text-align:center">HIT</th></tr>')
    cols_all = cols.replace(
        f'<th style="{th}">SECTOR</th>',
        f'<th style="{th}">SECTOR</th><th style="{th}">SIDE</th>')
    ranked = np.argsort(r["roc"])[::-1]

    return f"""<div style="{head}">
<h2 style="margin:0 0 4px">JP close review &mdash; {date}</h2>
<p style="margin:0 0 4px;color:#57606a">
Japanese open-to-close realised on <b>{date}</b>, against the book implied
by the U.S. close on <b>{sig}</b>.<br>
Model: subspace-regularized PCA (PCA_SUB), L={r['L']},
&lambda;={r['lam']}, K={r['K']}, q={r['q']}.<br>
Data source: {r['source']}
</p>
{banner}
{headline}
{legs}
<h3 style="margin:24px 0 8px">The book</h3>
<table style="border-collapse:collapse;width:100%;font-size:14px">
{cols}
<tr><td colspan="9" style="padding:8px 10px;background:#eaf6ec;
font-weight:600;color:#1a7f37">LONG</td></tr>
{_html_rows(longs, r)}
<tr><td colspan="9" style="padding:8px 10px;background:#fdecee;
font-weight:600;color:#c11d2b">SHORT</td></tr>
{_html_rows(shorts, r)}
</table>
<p style="color:#57606a;font-size:13px">
CHG is the move in yen per unit (close &minus; open); CONTRIB is the
name's share of the book return, w<sub>j</sub> &times; roc<sub>j</sub>.</p>
{hist_html}
{chart_html}
<h3 style="margin:24px 0 8px">All 17 sectors, by realised return</h3>
<table style="border-collapse:collapse;width:100%;font-size:14px">
{cols_all}
{_html_rows(ranked, r, with_side=True)}
</table>
<hr style="border:none;border-top:1px solid #d0d7de;margin:24px 0">
<p style="color:#57606a;font-size:12px">
Generated by <code>review_report.py</code>. Research output only &mdash;
not investment advice. Transaction costs and slippage are not modelled.
</p>
</div>"""


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------
DISCORD_COLOR_UP = 0x1A7F37
DISCORD_COLOR_DOWN = 0xC11D2B
DISCORD_COLOR_SYNTHETIC = 0xF0A020


def _discord_side_field(indices, r):
    if len(indices) == 0:
        return "—"
    lines = []
    for i in indices:
        tk = C.JP_TICKERS[i]
        mark = "✅" if _hit(i, r) == "OK" else "❌"
        lines.append(f"{mark} `{tk:<7s}` `{100 * r['roc'][i]:+6.2f}%` "
                     f"`{r['change'][i]:+7.1f}` {C.JP_LABELS.get(tk, '')}")
    return "\n".join(lines)


def build_discord_payload(r, hist=None, attach_chart=False) -> dict:
    date = pd.Timestamp(r["review_date"]).date()
    sig = pd.Timestamp(r["signal_date"]).date()
    longs, shorts = _order(r)
    pnl = r["port_return"]

    description = (
        f"Japanese open-to-close realised on **{date}**, scored against the "
        f"book from the U.S. close on **{sig}**.\n"
        f"**Book return `{100 * pnl:+.2f}%`**  "
        f"(long `{100 * r['long_leg']:+.2f}%` / "
        f"short `{100 * r['short_leg']:+.2f}%`)\n"
        f"Direction hit `{r['n_hits']}/{r['n_book']}`  ·  "
        f"rank IC `{r['ic']:+.3f}`  ·  "
        f"TOPIX-17 mean `{100 * r['market']:+.2f}%`"
    )
    if not r["live"]:
        description = ("⚠️ **SYNTHETIC fallback data — not a real "
                       "result.**\n\n" + description)
    if r["stale_book"]:
        description += ("\n\nℹ️ No new signal was published for this session "
                        "(U.S. market shut the day before); the previous "
                        "book was carried.")

    fields = [
        {"name": f"\U0001F7E2 LONG ({len(longs)})",
         "value": _discord_side_field(longs, r), "inline": False},
        {"name": f"\U0001F534 SHORT ({len(shorts)})",
         "value": _discord_side_field(shorts, r), "inline": False},
    ]
    if hist:
        fields.append({
            "name": f"Trailing {hist['n']} reviewed sessions",
            "value": (f"cumulative `{100 * hist['cumulative']:+.2f}%`  "
                      f"mean `{100 * hist['mean']:+.3f}%`  "
                      f"positive `{100 * hist['win_rate']:.0f}%`  "
                      f"hit `{100 * hist['hit_rate']:.0f}%`"),
            "inline": False,
        })

    if not r["live"]:
        colour = DISCORD_COLOR_SYNTHETIC
    else:
        colour = DISCORD_COLOR_UP if pnl >= 0 else DISCORD_COLOR_DOWN

    embed = {
        "title": f"JP close review — {date}  ({100 * pnl:+.2f}%)",
        "description": description,
        "color": colour,
        "fields": fields,
        "footer": {"text": "% is open-to-close; the second column is the "
                           "move in yen. Research output only -- not "
                           "investment advice."},
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if attach_chart:
        embed["image"] = {"url": "attachment://signal.png"}
    return {"username": "Lead-Lag PCA Bot", "embeds": [embed]}


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def save_review_chart(r, path, hist=None):
    """Predicted signal beside the realised move, plus the trailing curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.style.use("seaborn-v0_8-dark")

    order = np.argsort(r["z_hat"])          # weakest at the bottom
    labels = [f"{C.JP_TICKERS[i].replace('.T', '')}  "
              f"{C.JP_LABELS.get(C.JP_TICKERS[i], '')}" for i in order]
    colours = ["#d62728" if r["w"][i] < 0 else
               ("#2ca02c" if r["w"][i] > 0 else "#bbbbbb") for i in order]

    nrows = 2 if hist is not None and hist["n"] > 1 else 1
    fig = plt.figure(figsize=(12, 7.5 if nrows == 1 else 10))
    # explicit margins rather than tight_layout: the two top panels share a
    # y axis whose tick labels only exist on the left one, which tight_layout
    # cannot balance.
    gs = fig.add_gridspec(nrows, 2, height_ratios=[3, 1][:nrows], hspace=0.32,
                          wspace=0.05, left=0.20, right=0.98, top=0.90,
                          bottom=0.06 if nrows == 1 else 0.10)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.barh(range(len(order)), r["z_hat"][order], color=colours)
    ax0.set_yticks(range(len(order)))
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.axvline(0, color="k", lw=0.8)
    ax0.set_xlabel("Predicted  $\\hat{z}_{J,t+1}$")
    ax0.set_title("Predicted (signal day "
                  f"{pd.Timestamp(r['signal_date']).date()})", fontsize=10)
    ax0.grid(axis="x", alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    realised = 100 * r["roc"][order]
    ax1.barh(range(len(order)), realised, color=colours)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("Realised open-to-close (%)")
    ax1.set_title(f"Realised (JP session "
                  f"{pd.Timestamp(r['review_date']).date()})", fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    plt.setp(ax1.get_yticklabels(), visible=False)

    pad = max(0.06 * max(np.abs(realised).max(), 0.1), 0.02)
    for row, i in enumerate(order):
        if r["w"][i] == 0:
            continue
        ok = np.sign(r["w"][i]) == np.sign(r["roc"][i])
        x = realised[row]
        ax1.text(x + (pad if x >= 0 else -pad), row, "✓" if ok else "✗",
                 va="center", ha="left" if x >= 0 else "right", fontsize=9,
                 color="#1a7f37" if ok else "#c11d2b", fontweight="bold")
    ax1.margins(x=0.14)

    fig.suptitle(
        f"Lead-lag book review — {pd.Timestamp(r['review_date']).date()}"
        f"   book {100 * r['port_return']:+.2f}%"
        f"   hit {r['n_hits']}/{r['n_book']}"
        f"   IC {r['ic']:+.3f}   [{r['source'].split()[0]}]",
        fontsize=11)
    ax0.legend(handles=[Patch(facecolor="#2ca02c", label="LONG"),
                        Patch(facecolor="#d62728", label="SHORT"),
                        Patch(facecolor="#bbbbbb", label="neutral")],
               loc="lower right", fontsize=8)

    if nrows == 2:
        ax2 = fig.add_subplot(gs[1, :])
        curve = 100 * ((1.0 + hist["returns"]).cumprod() - 1.0)
        ax2.plot(curve.index, curve.values, color="#2f6feb", lw=1.6,
                 marker="o", ms=3)
        ax2.axhline(0, color="k", lw=0.8)
        ax2.fill_between(curve.index, 0, curve.values,
                         where=curve.values >= 0, color="#2ca02c", alpha=0.15)
        ax2.fill_between(curve.index, 0, curve.values,
                         where=curve.values < 0, color="#d62728", alpha=0.15)
        ax2.set_ylabel("cumulative (%)", fontsize=9)
        ax2.set_title(f"Trailing {hist['n']} reviewed sessions "
                      f"({100 * hist['cumulative']:+.2f}%)", fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.tick_params(axis="x", labelrotation=45, labelsize=8)

    fig.savefig(path, dpi=120)
    print(f"[chart] written to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Post-close review of the lead-lag book")
    ap.add_argument("--date", default=None,
                    help="Japanese session to review (YYYY-MM-DD); "
                         "default: the latest one available")
    ap.add_argument("--channels", default=None,
                    help="comma-separated delivery channels: email,discord "
                         "(default: $REPORT_CHANNELS or 'email')")
    ap.add_argument("--to", default=None,
                    help=f"comma-separated recipients (default: $REPORT_TO or "
                         f"the mock address {D.MOCK_ADDRESS})")
    ap.add_argument("--sender", "--from", dest="sender", default=None,
                    help=f"From address (default: $REPORT_FROM or "
                         f"{D.MOCK_SENDER})")
    ap.add_argument("--discord-webhook", default=None,
                    help="Discord webhook URL (default: $DISCORD_WEBHOOK_URL)")
    ap.add_argument("--dry-run", action="store_true",
                    help="render only; never open an SMTP connection or "
                         "post to Discord")
    ap.add_argument("--require-live", action="store_true",
                    help="exit non-zero instead of falling back to synthetic "
                         "data (use in production so no simulated result is "
                         "ever sent out)")
    ap.add_argument("--offline", action="store_true",
                    help="skip the network and use the synthetic fallback")
    ap.add_argument("--no-chart", action="store_true",
                    help="do not render or attach the chart")
    ap.add_argument("--skip-if-stale", type=int, default=-1, metavar="DAYS",
                    help="exit 0 without sending when the latest Japanese "
                         "session is more than DAYS calendar days old "
                         "(0 = must be today, in JST).  Ignored with --date. "
                         "Default: -1 (never skip)")
    ap.add_argument("--history-days", type=int, default=20,
                    help="reviewed sessions to summarise (0=off)")
    ap.add_argument("--L", type=int, default=60, help="estimation window")
    ap.add_argument("--lam", type=float, default=0.9, help="shrinkage lambda")
    ap.add_argument("--K", type=int, default=3, help="number of factors")
    ap.add_argument("--q", type=float, default=0.3, help="long/short quantile")
    ap.add_argument("--prior-start", default="2010-01-01",
                    help="download start and beginning of the prior window")
    ap.add_argument("--prior-end", default="2014-12-31",
                    help="end of prior training window")
    ap.add_argument("--output-dir", default="output",
                    help="directory for charts and dry-run output")
    args = ap.parse_args()

    out_dir = ensure_output_dir(args.output_dir)

    # --- data -------------------------------------------------------------
    open_, close, source = get_data(args.prior_start,
                                    allow_network=not args.offline)
    live = source.startswith("yfinance")
    if args.require_live and not live:
        print("[error] --require-live was set but live data is unavailable "
              f"(source: {source}); refusing to send a synthetic review.",
              file=sys.stderr)
        return 1

    tickers = C.US_TICKERS + C.JP_TICKERS
    rcc, C0 = build_prior(close, tickers, prior_start=args.prior_start,
                          prior_end=args.prior_end)
    jp_open, jp_close, extended = jp_session_frames(
        open_, close, allow_network=live and not args.offline)
    if extended:
        print(f"[info] Japanese sessions available through "
              f"{jp_close.index[-1].date()}.")

    # --- review -----------------------------------------------------------
    kw = dict(L=args.L, lam=args.lam, K=args.K, q=args.q)
    review_date = args.date or jp_close.index[-1]
    r = review_at(rcc, C0, jp_open, jp_close, review_date, source, **kw)
    r["live"] = live

    # The job runs at ~15:00 JST, which is still the same calendar date in
    # UTC (06:00 UTC), so today's UTC date is the session date to expect.
    stale = (pd.Timestamp(dt.date.today())
             - pd.Timestamp(r["review_date"])).days
    if args.date is None and stale > 0:
        print(f"[warn] latest Japanese session is "
              f"{pd.Timestamp(r['review_date']).date()}, {stale} day(s) old.",
              file=sys.stderr)
        if args.skip_if_stale >= 0 and stale > args.skip_if_stale:
            print(f"[skip] no Japanese session newer than "
                  f"{args.skip_if_stale} day(s); the market did not trade "
                  "today, or the close has not been published yet. "
                  "Nothing sent.")
            return 0

    hist = None
    if args.history_days > 0:
        try:
            hist = review_history(rcc, C0, jp_open, jp_close, r["review_date"],
                                  source, days=args.history_days, **kw)
        except Exception as e:  # never let the extras block the report
            print(f"[warn] trailing review history unavailable "
                  f"({type(e).__name__}: {e})", file=sys.stderr)

    chart_path = None
    if not args.no_chart:
        d_str = str(pd.Timestamp(r["review_date"]).date())
        chart_path = os.path.join(out_dir, f"review_{d_str}.png")
        save_review_chart(r, chart_path, hist)

    # --- deliver ----------------------------------------------------------
    channels = D.resolve_channels(args)

    if "email" in channels:
        cfg = D.smtp_config(args)
        msg = D.assemble_message(
            subject_line(r), cfg["sender"], cfg["recipients"],
            render_text(r, hist), lambda cid: render_html(r, hist, cid),
            chart_path, filename="review.png")

        dry = args.dry_run or not cfg["host"]
        if dry and not args.dry_run:
            print("[info] SMTP_HOST is not set -- falling back to a dry run.")

        if dry:
            eml, html = D.write_dry_run(msg, r, out_dir,
                                        prefix="review_report",
                                        date_key="review_date")
            print(f"[dry-run] email would send to: "
                  f"{', '.join(cfg['recipients'])}")
            print(f"[dry-run] email subject: {msg['Subject']}")
            print(f"[dry-run] message written to {eml}")
            print(f"[dry-run] html preview written to {html}")
        else:
            D.send_message(msg, cfg)
            print(f"[sent] email: {msg['Subject']}")
            print(f"[sent] email to {', '.join(cfg['recipients'])} via "
                  f"{cfg['host']}:{cfg['port']}")

    if "discord" in channels:
        webhook = (args.discord_webhook
                   or os.environ.get("DISCORD_WEBHOOK_URL", "").strip())
        payload = build_discord_payload(r, hist, attach_chart=bool(chart_path))

        dry = args.dry_run or not webhook
        if dry and not args.dry_run:
            print("[info] DISCORD_WEBHOOK_URL is not set -- falling back "
                  "to a dry run.")

        if dry:
            path = D.write_discord_dry_run(payload, r, out_dir,
                                           prefix="discord_review",
                                           date_key="review_date")
            print(f"[dry-run] discord payload written to {path}")
        else:
            D.post_discord(webhook, payload, chart_path)
            print(f"[sent] discord: {payload['embeds'][0]['title']} "
                  f"-> {D.mask_webhook(webhook)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
