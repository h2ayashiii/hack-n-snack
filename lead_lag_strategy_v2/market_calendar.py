"""
market_calendar.py
==================
Exchange-calendar logic for the scheduled daily signal: which Japanese
session the next delivery is for, which U.S. close it must be built from,
and whether that close has settled yet.

The paper's timing convention -- the U.S. close-to-close return on day t
predicts the Japanese open-to-close return on day t+1 -- is applied to
the two real exchange calendars (``XNYS`` and ``XTKS`` from
``exchange_calendars``) rather than to weekdays:

* the target is the next Tokyo session whose open (09:00 JST) is still
  ahead, so nothing is ever addressed to a day Tokyo is shut;
* the U.S. input is the last NYSE session that closed before that open,
  so after a U.S. holiday the latest available U.S. close is used, and
  after a Japanese holiday the U.S. session(s) Tokyo missed are not
  skipped;
* the run is "ready" only once that U.S. close is ``settle`` old, so a
  still-trading intraday bar is never mistaken for the daily close.

DST and early closes come from the calendars themselves.

The post-close review (``review_report.py --auto``) uses the same
calendars the other way round: it reviews the latest Tokyo session that
has closed, and only if a book was actually delivered for it (the books
are saved here by the signal job, see :func:`save_book`).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass

import pandas as pd

JP_CALENDAR = "XTKS"
US_CALENDAR = "XNYS"
DEFAULT_SETTLE = dt.timedelta(minutes=15)
REVIEW_SETTLE = dt.timedelta(minutes=30)
BOOKS_KEPT = 60


@dataclass(frozen=True)
class DeliveryPlan:
    now: pd.Timestamp           # UTC
    jp_session: pd.Timestamp    # Tokyo session the book is for
    jp_open: pd.Timestamp       # its open, UTC
    us_session: pd.Timestamp    # U.S. session whose close is the signal day t
    us_close: pd.Timestamp      # its close, UTC
    ready_at: pd.Timestamp      # us_close + settle

    @property
    def ready(self) -> bool:
        return self.now >= self.ready_at

    def describe(self) -> str:
        jst = "Asia/Tokyo"
        return (f"JP session {self.jp_session.date()} (opens "
                f"{self.jp_open.tz_convert(jst):%Y-%m-%d %H:%M} JST) <- "
                f"U.S. close {self.us_session.date()} "
                f"({self.us_close:%Y-%m-%d %H:%M} UTC)")


def _calendars():
    import exchange_calendars as xc
    return xc.get_calendar(JP_CALENDAR), xc.get_calendar(US_CALENDAR)


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def plan_delivery(now=None, settle: dt.timedelta = DEFAULT_SETTLE
                  ) -> DeliveryPlan:
    """Work out what a delivery made at ``now`` (UTC) is for."""
    now = _utc(now if now is not None else dt.datetime.now(dt.timezone.utc))
    jp, us = _calendars()

    day = now.tz_convert(None).normalize()
    jp_sessions = jp.sessions_in_range(day - pd.Timedelta(days=1),
                                       day + pd.Timedelta(days=21))
    jp_session = next(s for s in jp_sessions if jp.session_open(s) > now)
    jp_open = jp.session_open(jp_session)

    us_sessions = us.sessions_in_range(day - pd.Timedelta(days=21),
                                       day + pd.Timedelta(days=21))
    us_session = [s for s in us_sessions if us.session_close(s) < jp_open][-1]
    us_close = us.session_close(us_session)

    return DeliveryPlan(now=now, jp_session=jp_session, jp_open=jp_open,
                        us_session=us_session, us_close=us_close,
                        ready_at=us_close + settle)


def latest_closed_jp_session(now=None, settle: dt.timedelta = REVIEW_SETTLE
                             ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The latest Tokyo session closed at least ``settle`` before ``now``.

    Returns ``(session, close_utc)``.
    """
    now = _utc(now if now is not None else dt.datetime.now(dt.timezone.utc))
    jp, _ = _calendars()
    day = now.tz_convert(None).normalize()
    sessions = jp.sessions_in_range(day - pd.Timedelta(days=21), day)
    session = [s for s in sessions if jp.session_close(s) + settle <= now][-1]
    return session, jp.session_close(session)


# ---------------------------------------------------------------------------
# Delivery / review state, persisted between runs by the workflow cache
# ---------------------------------------------------------------------------
def last_session(path: str) -> pd.Timestamp | None:
    """The ``jp_session`` recorded in a state file, if any."""
    try:
        with open(path, encoding="utf-8") as fh:
            return pd.Timestamp(json.load(fh)["jp_session"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def record_session(path: str, jp_session, **extra) -> None:
    """Record ``jp_session`` (plus ``extra`` fields) as done in ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {"jp_session": str(pd.Timestamp(jp_session).date())}
    rec.update({k: str(v) for k, v in extra.items()})
    rec["at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)


def record_sent(path: str, plan: DeliveryPlan, signal_date) -> None:
    record_session(path, plan.jp_session,
                   us_session=plan.us_session.date(),
                   signal_date=pd.Timestamp(signal_date).date())


# The delivered books themselves, one JSON file per Tokyo session, so the
# post-close review scores exactly what was sent rather than recomputing.
def _book_path(book_dir: str, jp_session) -> str:
    return os.path.join(book_dir, f"{pd.Timestamp(jp_session).date()}.json")


def save_book(book_dir: str, jp_session, s: dict, tickers) -> str:
    """Save the delivered snapshot ``s`` as the book for ``jp_session``."""
    os.makedirs(book_dir, exist_ok=True)
    book = {"jp_session": str(pd.Timestamp(jp_session).date()),
            "signal_date": str(pd.Timestamp(s["signal_date"]).date()),
            "tickers": list(tickers),
            "w": [float(x) for x in s["w"]],
            "z_hat": [float(x) for x in s["z_hat"]],
            "us_used": list(s.get("us_used", [])),
            "source": s["source"],
            "L": s["L"], "lam": s["lam"], "K": s["K"], "q": s["q"]}
    path = _book_path(book_dir, jp_session)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(book, fh, indent=2)
    for old in list_books(book_dir)[:-BOOKS_KEPT]:
        os.remove(_book_path(book_dir, old))
    return path


def load_book(book_dir: str, jp_session) -> dict | None:
    try:
        with open(_book_path(book_dir, jp_session), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def list_books(book_dir: str) -> list[pd.Timestamp]:
    """Sessions with a saved book, oldest first."""
    try:
        names = os.listdir(book_dir)
    except OSError:
        return []
    out = []
    for n in names:
        if n.endswith(".json"):
            try:
                out.append(pd.Timestamp(n[:-5]))
            except ValueError:
                pass
    return sorted(out)


def jp_sessions_between(after, before) -> list[pd.Timestamp]:
    """Tokyo sessions strictly between two session dates."""
    after, before = pd.Timestamp(after), pd.Timestamp(before)
    if before - after <= pd.Timedelta(days=1):
        return []
    jp, _ = _calendars()
    return list(jp.sessions_in_range(after + pd.Timedelta(days=1),
                                     before - pd.Timedelta(days=1)))
