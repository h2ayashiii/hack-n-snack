"""
daily_report.py
===============
Daily driver for the subspace-regularized PCA lead-lag model: build the
signal for the latest U.S. close, render it, and deliver it.

Designed to be run once per day from CI (see
``.github/workflows/daily-signal.yml``) after the U.S. close and before
the Japanese open, which is exactly the window the paper's timing
convention targets: the U.S. close-to-close return on day *t* predicts
the Japanese open-to-close return on day *t+1*.

Two delivery channels are supported, selected independently of each
other and of the underlying signal computation:

    --channels email            (default) e-mail only
    --channels discord          Discord webhook only
    --channels email,discord    both

    REPORT_CHANNELS   env var equivalent of --channels (CLI wins)

E-mail configuration is read from the environment so that CI can supply
it through secrets:

    SMTP_HOST       SMTP server host        (unset -> dry run)
    SMTP_PORT       default 587 (587/25 -> STARTTLS, 465 -> implicit TLS)
    SMTP_USER       login user              (optional)
    SMTP_PASSWORD   login password          (optional)
    SMTP_STARTTLS   "0" to disable STARTTLS on non-465 ports
    REPORT_FROM     envelope/From address   (default: the mock address)
    REPORT_TO       comma-separated list    (default: the mock address)

The defaults point at ``@example.com``, a domain RFC 2606 reserves for
documentation, so an unconfigured deployment can never deliver mail to a
real mailbox.  With no SMTP_HOST the script falls back to a dry run and
writes the rendered message to the output directory instead of sending.

Discord configuration:

    DISCORD_WEBHOOK_URL   the target channel's webhook URL (unset -> dry run)

With no DISCORD_WEBHOOK_URL the script falls back to writing the embed
payload as JSON instead of posting it, exactly like the e-mail path.

Run
---
    # Render only -- writes output/daily_report_YYYY-MM-DD.{eml,html}
    # and/or output/discord_payload_YYYY-MM-DD.json depending on --channels
    python daily_report.py --dry-run

    # Render and send by e-mail (needs SMTP_HOST etc. in the environment)
    python daily_report.py

    # Post to Discord instead (needs DISCORD_WEBHOOK_URL in the environment)
    python daily_report.py --channels discord

    # Both channels at once
    python daily_report.py --channels email,discord

    # Backfill / test a specific signal date, no network
    python daily_report.py --date 2024-11-01 --offline --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import smtplib
import sys
import urllib.error
import urllib.request
import uuid
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

import numpy as np
import pandas as pd

import common as C
from realtime_run import build_prior, ensure_output_dir, get_data, save_chart_single

MOCK_ADDRESS = "lead-lag-signals@example.com"
MOCK_SENDER = "lead-lag-bot@example.com"

FACTOR_NAMES = ["global", "US-Japan spread", "cyclical-defensive"]

VALID_CHANNELS = {"email", "discord"}


# ---------------------------------------------------------------------------
# Trailing realised performance
# ---------------------------------------------------------------------------
def trailing_performance(rcc, roc_J, C0, days=20, L=60, lam=0.9, K=3, q=0.3):
    """Realised PCA_SUB returns over the last ``days`` tradeable sessions.

    Runs the same walk-forward loop as the backtest (eqs. 3-7, 17-21) on
    the tail of the sample, so the e-mail can show how the book that was
    published on previous days actually did.  Returns ``None`` when there
    is not enough clean history behind the tail.
    """
    need = days + L + 2
    if len(rcc) < need:
        return None
    tail = rcc.iloc[-need:]
    roc_tail = roc_J.reindex(tail.index)
    rets = C.run_backtest(tail, roc_tail, list(tail.columns), C.JP_TICKERS,
                          C0, L=L, lam=lam, K=K, q=q, methods=("PCA_SUB",))
    r = rets["PCA_SUB"].dropna()
    if len(r) == 0:
        return None
    r = r.iloc[-days:]
    cum = float((1.0 + r).prod() - 1.0)
    return dict(n=len(r), cumulative=cum, mean=float(r.mean()),
                hit_rate=float((r > 0).mean()),
                first=r.index[0], last=r.index[-1])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _book(s):
    """Split the snapshot into (longs, shorts), each ranked by conviction."""
    longs = [i for i in np.argsort(s["z_hat"])[::-1] if s["w"][i] > 0]
    shorts = [i for i in np.argsort(s["z_hat"]) if s["w"][i] < 0]
    return longs, shorts


def _row(i, s):
    tk = C.JP_TICKERS[i]
    return tk, C.JP_LABELS.get(tk, ""), float(s["z_hat"][i]), float(s["w"][i])


def subject_line(s, perf=None) -> str:
    longs, shorts = _book(s)
    date = pd.Timestamp(s["signal_date"]).date()
    prefix = "[SYNTHETIC] " if not s["live"] else ""
    top_long = C.JP_TICKERS[longs[0]].replace(".T", "") if longs else "-"
    top_short = C.JP_TICKERS[shorts[0]].replace(".T", "") if shorts else "-"
    return (f"{prefix}[Lead-Lag PCA_SUB] {date} signal -> next JP session"
            f"  |  L:{len(longs)} S:{len(shorts)}"
            f"  top +{top_long} / -{top_short}")


def render_text(s, perf=None) -> str:
    """Plain-text alternative -- mirrors the console snapshot layout."""
    line = "=" * 68
    out = [line,
           " SUBSPACE-REGULARIZED PCA LEAD-LAG  --  DAILY SIGNAL",
           line,
           f" signal day (t)   : {pd.Timestamp(s['signal_date']).date()}"
           f"  (U.S. close-to-close)",
           f" predicting       : next Japanese open-to-close (t+1)",
           f" data source      : {s['source']}",
           f" params           : L={s['L']}  lambda={s['lam']}"
           f"  K={s['K']}  q={s['q']}"]

    us_used = s.get("us_used", C.US_TICKERS)
    if len(us_used) < len(C.US_TICKERS):
        missing = ", ".join(sorted(set(C.US_TICKERS) - set(us_used)))
        out.append(f" U.S. universe    : {len(us_used)}/{len(C.US_TICKERS)}"
                   f"  (no data yet for {missing})")
    out.append(line)

    if not s["live"]:
        out += ["",
                " *** WARNING: built on SYNTHETIC fallback data, not live"
                " prices. ***",
                " *** Do not trade on this book.                        ***"]

    longs, shorts = _book(s)
    out += ["", " LONG  (top-q):"]
    for i in longs:
        tk, lbl, z, w = _row(i, s)
        out.append(f"   {tk:<8s} {lbl:<34s} zhat={z:+.3f}  w={w:+.3f}")
    out += ["", " SHORT (bottom-q):"]
    for i in shorts:
        tk, lbl, z, w = _row(i, s)
        out.append(f"   {tk:<8s} {lbl:<34s} zhat={z:+.3f}  w={w:+.3f}")
    out.append(f"\n   sum(w) = {s['w'].sum():+.3f}"
               f"   sum|w| = {np.abs(s['w']).sum():.3f}   (target: 0 and 2)")

    out += ["", " Common-factor scores f_t (eq. 18):"]
    for k in range(s["K"]):
        nm = FACTOR_NAMES[k] if k < len(FACTOR_NAMES) else f"factor {k+1}"
        out.append(f"   f_{k+1} ({nm:<18s}) = {s['f'][k]:+.3f}")

    ev = s["evals"]
    tot = ev[ev > 0].sum()
    out += ["", " Regularized correlation spectrum (eq. 14), top 6:"]
    for k in range(min(6, len(ev))):
        tag = "  <- retained" if k < s["K"] else ""
        out.append(f"   lambda_{k+1:<2d} = {ev[k]:7.3f}"
                   f"  ({100 * ev[k] / tot:5.1f}%){tag}")

    out += ["", " Full ranking of zhat_{J,t+1}:"]
    for rk, i in enumerate(np.argsort(s["z_hat"])[::-1], 1):
        tk, lbl, z, w = _row(i, s)
        book = "LONG" if w > 0 else ("SHORT" if w < 0 else "")
        out.append(f"   {rk:>3d}  {tk:<8s} {lbl:<34s} {z:+.3f}  {book}")

    if perf:
        out += ["", f" Realised PCA_SUB performance, last {perf['n']} sessions"
                    f" ({perf['first'].date()} .. {perf['last'].date()}):",
                f"   cumulative : {100 * perf['cumulative']:+.2f}%",
                f"   mean daily : {100 * perf['mean']:+.3f}%",
                f"   hit rate   : {100 * perf['hit_rate']:.1f}%"]

    out += ["", line,
            " Research output only -- no investment advice."
            "  Costs and slippage are not modelled.",
            line]
    return "\n".join(out)


def _html_rows(indices, s):
    rows = []
    for i in indices:
        tk, lbl, z, w = _row(i, s)
        colour = "#1a7f37" if w > 0 else ("#c11d2b" if w < 0 else "#57606a")
        rows.append(
            f'<tr><td style="padding:4px 10px;border-bottom:1px solid #eee;'
            f'font-family:monospace">{tk}</td>'
            f'<td style="padding:4px 10px;border-bottom:1px solid #eee">{lbl}</td>'
            f'<td style="padding:4px 10px;border-bottom:1px solid #eee;'
            f'text-align:right;font-family:monospace">{z:+.3f}</td>'
            f'<td style="padding:4px 10px;border-bottom:1px solid #eee;'
            f'text-align:right;font-family:monospace;color:{colour}">'
            f'{w:+.3f}</td></tr>')
    return "".join(rows)


def render_html(s, perf=None, chart_cid=None) -> str:
    longs, shorts = _book(s)
    date = pd.Timestamp(s["signal_date"]).date()
    head = ('font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;'
            'color:#1f2328;max-width:760px')
    # no text-transform here: it would uppercase the "t+1" subscript too
    th = ('padding:6px 10px;text-align:left;border-bottom:2px solid #d0d7de;'
          'font-size:12px;letter-spacing:.04em;color:#57606a')

    banner = ""
    if not s["live"]:
        banner = ('<div style="background:#fff3cd;border:1px solid #f0c36d;'
                  'padding:12px;border-radius:6px;margin-bottom:16px">'
                  '<b>SYNTHETIC fallback data.</b> Live prices were '
                  'unavailable, so this book is model output on simulated '
                  'returns. Do not trade on it.</div>')

    us_used = s.get("us_used", C.US_TICKERS)
    universe = ""
    if len(us_used) < len(C.US_TICKERS):
        missing = ", ".join(sorted(set(C.US_TICKERS) - set(us_used)))
        universe = (f'<br>U.S. universe: {len(us_used)}/{len(C.US_TICKERS)} '
                    f'(no data yet for {missing})')

    factors = "".join(
        f'<li>f<sub>{k+1}</sub> '
        f'({FACTOR_NAMES[k] if k < len(FACTOR_NAMES) else f"factor {k+1}"}) '
        f'= <code>{s["f"][k]:+.3f}</code></li>'
        for k in range(s["K"]))

    perf_html = ""
    if perf:
        perf_html = (
            f'<h3 style="margin:24px 0 8px">Realised performance '
            f'(last {perf["n"]} sessions)</h3>'
            f'<p style="margin:0;color:#57606a;font-size:13px">'
            f'{perf["first"].date()} &ndash; {perf["last"].date()}</p>'
            f'<ul style="line-height:1.6"><li>cumulative: '
            f'<code>{100 * perf["cumulative"]:+.2f}%</code></li>'
            f'<li>mean daily: <code>{100 * perf["mean"]:+.3f}%</code></li>'
            f'<li>hit rate: <code>{100 * perf["hit_rate"]:.1f}%</code></li></ul>')

    chart_html = ""
    if chart_cid:
        chart_html = (f'<h3 style="margin:24px 0 8px">Predicted returns</h3>'
                      f'<img src="cid:{chart_cid}" style="max-width:100%" '
                      f'alt="predicted standardised Japanese returns">')

    ranking = _html_rows(np.argsort(s["z_hat"])[::-1], s)

    return f"""<div style="{head}">
<h2 style="margin:0 0 4px">Lead-lag signal &mdash; {date}</h2>
<p style="margin:0 0 16px;color:#57606a">
U.S. close-to-close on <b>{date}</b> &rarr; predicted Japanese
open-to-close for the <b>next JP session</b>.<br>
Model: subspace-regularized PCA (PCA_SUB), L={s['L']},
&lambda;={s['lam']}, K={s['K']}, q={s['q']}.<br>
Data source: {s['source']}{universe}
</p>
{banner}
<h3 style="margin:24px 0 8px">Book for the next session</h3>
<table style="border-collapse:collapse;width:100%;font-size:14px">
<tr><th style="{th}">ETF</th><th style="{th}">SECTOR</th>
<th style="{th};text-align:right">&#7825;<sub>J,t+1</sub></th>
<th style="{th};text-align:right">weight</th></tr>
<tr><td colspan="4" style="padding:8px 10px;background:#eaf6ec;
font-weight:600;color:#1a7f37">LONG</td></tr>
{_html_rows(longs, s)}
<tr><td colspan="4" style="padding:8px 10px;background:#fdecee;
font-weight:600;color:#c11d2b">SHORT</td></tr>
{_html_rows(shorts, s)}
</table>
<p style="color:#57606a;font-size:13px">
&sum;w = {s['w'].sum():+.3f}, &sum;|w| = {np.abs(s['w']).sum():.3f}
(equal weight, dollar neutral &mdash; eq. 6)</p>
<h3 style="margin:24px 0 8px">Common-factor scores</h3>
<ul style="line-height:1.6">{factors}</ul>
{perf_html}
{chart_html}
<h3 style="margin:24px 0 8px">Full ranking</h3>
<table style="border-collapse:collapse;width:100%;font-size:14px">
<tr><th style="{th}">ETF</th><th style="{th}">SECTOR</th>
<th style="{th};text-align:right">&#7825;<sub>J,t+1</sub></th>
<th style="{th};text-align:right">weight</th></tr>
{ranking}
</table>
<hr style="border:none;border-top:1px solid #d0d7de;margin:24px 0">
<p style="color:#57606a;font-size:12px">
Generated by <code>daily_report.py</code>. Research output only &mdash;
not investment advice. Transaction costs and slippage are not modelled.
</p>
</div>"""


# ---------------------------------------------------------------------------
# Message assembly and delivery
# ---------------------------------------------------------------------------
def build_message(s, sender, recipients, perf=None, chart_path=None):
    """Assemble a multipart/alternative message with the chart inlined."""
    msg = EmailMessage()
    msg["Subject"] = subject_line(s, perf)
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1])

    chart_cid = None
    chart_bytes = None
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as fh:
            chart_bytes = fh.read()
        chart_cid = make_msgid(domain="lead-lag.local")[1:-1]  # strip <>

    msg.set_content(render_text(s, perf))
    msg.add_alternative(render_html(s, perf, chart_cid), subtype="html")

    if chart_bytes:
        html_part = msg.get_payload()[-1]
        html_part.add_related(chart_bytes, maintype="image", subtype="png",
                              cid=f"<{chart_cid}>", filename="signal.png")
    return msg


def smtp_config(args):
    """Resolve SMTP settings from the environment (CLI --to/--from win)."""
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587") or 587)
    sender = (args.sender or os.environ.get("REPORT_FROM", "").strip()
              or MOCK_SENDER)
    raw_to = args.to or os.environ.get("REPORT_TO", "").strip() or MOCK_ADDRESS
    recipients = [a.strip() for a in raw_to.split(",") if a.strip()]
    return dict(
        host=host,
        port=port,
        user=os.environ.get("SMTP_USER", "").strip(),
        password=os.environ.get("SMTP_PASSWORD", ""),
        starttls=os.environ.get("SMTP_STARTTLS", "1") != "0" and port != 465,
        ssl=port == 465,
        sender=sender,
        recipients=recipients,
    )


def send_message(msg, cfg):
    """Deliver via SMTP.  Raises on failure so CI surfaces the problem."""
    cls = smtplib.SMTP_SSL if cfg["ssl"] else smtplib.SMTP
    with cls(cfg["host"], cfg["port"], timeout=60) as srv:
        srv.ehlo()
        if cfg["starttls"]:
            srv.starttls()
            srv.ehlo()
        if cfg["user"]:
            srv.login(cfg["user"], cfg["password"])
        srv.send_message(msg, from_addr=cfg["sender"],
                         to_addrs=cfg["recipients"])


def write_dry_run(msg, s, out_dir):
    """Persist the rendered message so CI can upload it as an artifact."""
    stamp = pd.Timestamp(s["signal_date"]).date()
    eml = os.path.join(out_dir, f"daily_report_{stamp}.eml")
    html = os.path.join(out_dir, f"daily_report_{stamp}.html")
    with open(eml, "wb") as fh:
        fh.write(bytes(msg))
    body = msg.get_body(preferencelist=("html",))
    if body is not None:
        with open(html, "w", encoding="utf-8") as fh:
            fh.write(body.get_content())
    return eml, html


# ---------------------------------------------------------------------------
# Discord delivery
# ---------------------------------------------------------------------------
DISCORD_COLOR_LIVE = 0x2F6FEB       # blue -- live data
DISCORD_COLOR_SYNTHETIC = 0xF0A020  # amber -- synthetic fallback warning


def _discord_side_field(indices, s):
    if not indices:
        return "—"
    lines = []
    for i in indices:
        tk, lbl, z, w = _row(i, s)
        lines.append(f"`{tk:<8s}` {lbl}  `{z:+.3f}`")
    return "\n".join(lines)


def build_discord_payload(s, perf=None, attach_chart=False) -> dict:
    """Build a Discord webhook payload (a single embed) for the snapshot.

    Kept shorter than the e-mail on purpose: Discord embed fields cap at
    1024 characters and 25 fields, so this shows the tradeable book
    (top-q / bottom-q) rather than the full 17-sector ranking -- the
    attached chart covers the rest visually.
    """
    longs, shorts = _book(s)
    date = pd.Timestamp(s["signal_date"]).date()
    live = s["live"]

    factor_lines = []
    for k in range(s["K"]):
        nm = FACTOR_NAMES[k] if k < len(FACTOR_NAMES) else f"factor {k+1}"
        factor_lines.append(f"f_{k+1} ({nm}): `{s['f'][k]:+.3f}`")

    fields = [
        {"name": f"\U0001F7E2 LONG ({len(longs)})",
         "value": _discord_side_field(longs, s), "inline": True},
        {"name": f"\U0001F534 SHORT ({len(shorts)})",
         "value": _discord_side_field(shorts, s), "inline": True},
        {"name": "Common-factor scores (eq. 18)",
         "value": "\n".join(factor_lines) or "—", "inline": False},
    ]

    if perf:
        fields.append({
            "name": f"Realised PCA_SUB performance (last {perf['n']} sessions)",
            "value": (f"cumulative `{100 * perf['cumulative']:+.2f}%`  "
                      f"mean `{100 * perf['mean']:+.3f}%`  "
                      f"hit rate `{100 * perf['hit_rate']:.1f}%`"),
            "inline": False,
        })

    us_used = s.get("us_used", C.US_TICKERS)
    universe_note = ""
    if len(us_used) < len(C.US_TICKERS):
        missing = ", ".join(sorted(set(C.US_TICKERS) - set(us_used)))
        universe_note = (f"\nU.S. universe: {len(us_used)}/{len(C.US_TICKERS)} "
                         f"(no data yet for {missing})")

    description = (
        f"U.S. close-to-close on **{date}** → predicted Japanese "
        f"open-to-close for the **next JP session**.\n"
        f"Model: PCA_SUB, L={s['L']}, λ={s['lam']}, K={s['K']}, "
        f"q={s['q']}\n"
        f"Data source: {s['source']}{universe_note}"
    )
    if not live:
        description = ("⚠️ **SYNTHETIC fallback data — do not "
                       "trade on this.**\n\n" + description)

    embed = {
        "title": f"Lead-lag signal — {date}",
        "description": description,
        "color": DISCORD_COLOR_LIVE if live else DISCORD_COLOR_SYNTHETIC,
        "fields": fields,
        "footer": {"text": "Research output only -- not investment advice. "
                           "Costs/slippage not modelled."},
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if attach_chart:
        embed["image"] = {"url": "attachment://signal.png"}

    return {"username": "Lead-Lag PCA Bot", "embeds": [embed]}


def _encode_multipart(payload: dict, chart_path: str | None):
    """Minimal multipart/form-data encoder (stdlib only, no ``requests``).

    Discord's webhook endpoint expects the JSON payload under a
    ``payload_json`` part and, when an image is attached, the file under
    ``files[0]`` with the embed referencing it via ``attachment://name``.
    """
    boundary = f"leadlag-{uuid.uuid4().hex}"
    nl = "\r\n"
    chunks = [
        (f"--{boundary}{nl}"
         f'Content-Disposition: form-data; name="payload_json"{nl}'
         f"Content-Type: application/json{nl}{nl}"
         f"{json.dumps(payload)}{nl}").encode("utf-8")
    ]
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as fh:
            img = fh.read()
        chunks.append(
            (f"--{boundary}{nl}"
             f'Content-Disposition: form-data; name="files[0]"; '
             f'filename="signal.png"{nl}'
             f"Content-Type: image/png{nl}{nl}").encode("utf-8")
            + img + nl.encode("utf-8")
        )
    chunks.append(f"--{boundary}--{nl}".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


# https://discord.com/api/webhooks/<snowflake id>/<token>, tolerating the
# ptb./canary. subdomains, the legacy discordapp.com domain, a trailing
# slash, and an existing query string (which post_discord() extends).
_WEBHOOK_URL_RE = re.compile(
    r"^https://(?:ptb\.|canary\.)?discord(?:app)?\.com"
    r"/api/webhooks/(?P<id>\d+)/(?P<token>[^/?\s]+)/?(?:\?.*)?$")


def _validate_webhook_url(webhook_url: str) -> None:
    """Reject an obviously-malformed URL before spending a network call.

    ``404 Unknown Webhook`` from Discord itself just means "no webhook
    with this id+token exists right now" -- could be a genuinely deleted
    /regenerated webhook (nothing this script can fix), but it's the same
    error you'd get from a URL mangled while pasting it into the GitHub
    secret (stray newline/space, wrong domain, a truncated token). This
    catches the second class with a specific, actionable message instead
    of a bare 404.
    """
    if re.search(r"\s", webhook_url):
        raise ValueError(
            "DISCORD_WEBHOOK_URL contains whitespace/newline characters -- "
            "it was likely mangled when pasted into the GitHub secret. "
            "Re-copy the URL from Discord (channel settings -> "
            "Integrations -> Webhooks -> Copy Webhook URL) and re-paste it "
            "without a trailing newline.")
    if not _WEBHOOK_URL_RE.match(webhook_url):
        raise ValueError(
            "DISCORD_WEBHOOK_URL doesn't look like a Discord webhook URL "
            "(expected https://discord.com/api/webhooks/<id>/<token>). "
            f"Got a value of length {len(webhook_url)} starting with "
            f"{webhook_url[:30]!r}. Check for truncation or an extra "
            "character introduced when it was copied into the secret.")


def post_discord(webhook_url: str, payload: dict, chart_path=None, timeout=30):
    """POST the embed (with the chart attached) to a Discord webhook.

    Raises on failure so CI surfaces the problem, matching send_message().
    ``?wait=true`` makes Discord return the created message (or a detailed
    error body) instead of a bare 204, which is worth the extra latency
    for a once-a-day job.
    """
    _validate_webhook_url(webhook_url)
    content_type, body = _encode_multipart(payload, chart_path)
    sep = "&" if "?" in webhook_url else "?"
    req = urllib.request.Request(f"{webhook_url}{sep}wait=true", data=body,
                                 method="POST")
    req.add_header("Content-Type", content_type)
    # discord.com sits behind Cloudflare, which blocks urllib's default
    # "Python-urllib/x.y" User-Agent as a bot signature (HTTP 403,
    # Cloudflare error 1010) before the request ever reaches Discord.
    # A descriptive UA -- Discord's own convention for API clients --
    # gets past that check.
    req.add_header(
        "User-Agent",
        "lead-lag-strategy-v2-daily-report/1.0 "
        "(+https://github.com/h2ayashiii/hack-n-snack)")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"Discord webhook POST failed ({e.code}): {detail}") from e


def write_discord_dry_run(payload: dict, s, out_dir):
    """Persist the rendered payload so CI can upload it as an artifact."""
    stamp = pd.Timestamp(s["signal_date"]).date()
    path = os.path.join(out_dir, f"discord_payload_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def _mask_webhook(url: str) -> str:
    """Never print the webhook token in full -- it's a bearer credential
    that lets anyone post to the channel."""
    tail = url.rsplit("/", 1)[-1]
    return url[: len(url) - len(tail)] + tail[:6] + "…(redacted)"


def resolve_channels(args) -> list[str]:
    """Delivery channels from --channels, falling back to $REPORT_CHANNELS
    and then to e-mail-only, so existing deployments keep working
    unchanged."""
    raw = args.channels or os.environ.get("REPORT_CHANNELS", "").strip() or "email"
    channels = [c.strip().lower() for c in raw.split(",") if c.strip()]
    invalid = [c for c in channels if c not in VALID_CHANNELS]
    if invalid:
        raise SystemExit(f"unknown channel(s): {', '.join(invalid)} "
                         f"(valid: {', '.join(sorted(VALID_CHANNELS))})")
    return channels or ["email"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Daily lead-lag signal e-mail report")
    ap.add_argument("--date", default=None,
                    help="signal date (YYYY-MM-DD); default: latest available")
    ap.add_argument("--channels", default=None,
                    help="comma-separated delivery channels: email,discord "
                         "(default: $REPORT_CHANNELS or 'email')")
    ap.add_argument("--to", default=None,
                    help=f"comma-separated recipients (default: $REPORT_TO or "
                         f"the mock address {MOCK_ADDRESS})")
    ap.add_argument("--sender", "--from", dest="sender", default=None,
                    help=f"From address (default: $REPORT_FROM or {MOCK_SENDER})")
    ap.add_argument("--discord-webhook", default=None,
                    help="Discord webhook URL "
                         "(default: $DISCORD_WEBHOOK_URL)")
    ap.add_argument("--dry-run", action="store_true",
                    help="render only; never open an SMTP connection or "
                         "post to Discord")
    ap.add_argument("--require-live", action="store_true",
                    help="exit non-zero instead of falling back to synthetic "
                         "data (use in production so no simulated book is "
                         "ever mailed out)")
    ap.add_argument("--offline", action="store_true",
                    help="skip the network and use the synthetic fallback")
    ap.add_argument("--no-chart", action="store_true",
                    help="do not render or attach the chart")
    ap.add_argument("--skip-if-stale", type=int, default=-1, metavar="DAYS",
                    help="exit 0 without sending when the latest signal date "
                         "is more than DAYS calendar days old (0 = must be "
                         "today).  There is no new U.S. shock on a day the "
                         "two markets did not both trade, so re-mailing the "
                         "previous book would be misleading.  Ignored with "
                         "--date.  Default: -1 (never skip)")
    ap.add_argument("--trailing-days", type=int, default=20,
                    help="sessions of realised performance to include (0=off)")
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
              f"(source: {source}); refusing to send a synthetic book.",
              file=sys.stderr)
        return 1

    tickers = C.US_TICKERS + C.JP_TICKERS
    rcc, C0 = build_prior(close, tickers, prior_start=args.prior_start,
                          prior_end=args.prior_end)

    # --- signal -----------------------------------------------------------
    from realtime_run import snapshot_at

    signal_date = args.date or rcc.index[-1]
    s = snapshot_at(rcc, C0, signal_date, source, L=args.L, lam=args.lam,
                    K=args.K, q=args.q)
    s["live"] = live

    stale = (pd.Timestamp(dt.date.today()) - pd.Timestamp(s["signal_date"])).days
    if args.date is None and stale > 0:
        print(f"[warn] latest available signal date is "
              f"{pd.Timestamp(s['signal_date']).date()}, {stale} day(s) old.",
              file=sys.stderr)
        if args.skip_if_stale >= 0 and stale > args.skip_if_stale:
            print(f"[skip] no signal newer than {args.skip_if_stale} day(s); "
                  "the U.S. and Japanese markets did not both trade today. "
                  "Nothing sent.")
            return 0

    perf = None
    if args.trailing_days > 0:
        roc_J = C.open_to_close_returns(open_[C.JP_TICKERS],
                                        close[C.JP_TICKERS])
        try:
            perf = trailing_performance(rcc, roc_J, C0,
                                        days=args.trailing_days, L=args.L,
                                        lam=args.lam, K=args.K, q=args.q)
        except Exception as e:  # never let the extras block the report
            print(f"[warn] trailing performance unavailable "
                  f"({type(e).__name__}: {e})", file=sys.stderr)

    chart_path = None
    if not args.no_chart:
        d_str = str(pd.Timestamp(s["signal_date"]).date())
        chart_path = os.path.join(out_dir, f"realtime_signal_{d_str}.png")
        save_chart_single(s, chart_path)

    # --- deliver ----------------------------------------------------------
    channels = resolve_channels(args)

    if "email" in channels:
        cfg = smtp_config(args)
        msg = build_message(s, cfg["sender"], cfg["recipients"], perf, chart_path)

        dry = args.dry_run or not cfg["host"]
        if dry and not args.dry_run:
            print("[info] SMTP_HOST is not set -- falling back to a dry run.")

        if dry:
            eml, html = write_dry_run(msg, s, out_dir)
            print(f"[dry-run] email would send to: {', '.join(cfg['recipients'])}")
            print(f"[dry-run] email subject: {msg['Subject']}")
            print(f"[dry-run] message written to {eml}")
            print(f"[dry-run] html preview written to {html}")
        else:
            send_message(msg, cfg)
            print(f"[sent] email: {msg['Subject']}")
            print(f"[sent] email to {', '.join(cfg['recipients'])} via "
                  f"{cfg['host']}:{cfg['port']}")

    if "discord" in channels:
        webhook = ((args.discord_webhook or "").strip()
                  or os.environ.get("DISCORD_WEBHOOK_URL", "").strip())
        payload = build_discord_payload(s, perf, attach_chart=bool(chart_path))

        dry = args.dry_run or not webhook
        if dry and not args.dry_run:
            print("[info] DISCORD_WEBHOOK_URL is not set -- falling back "
                  "to a dry run.")

        if dry:
            path = write_discord_dry_run(payload, s, out_dir)
            print(f"[dry-run] discord payload written to {path}")
        else:
            post_discord(webhook, payload, chart_path)
            print(f"[sent] discord: {payload['embeds'][0]['title']} "
                  f"-> {_mask_webhook(webhook)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
