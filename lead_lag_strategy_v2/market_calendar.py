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


# ---------------------------------------------------------------------------
# "Already sent" state, persisted between runs by the workflow's cache
# ---------------------------------------------------------------------------
def last_sent_session(path: str) -> pd.Timestamp | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return pd.Timestamp(json.load(fh)["jp_session"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def record_sent(path: str, plan: DeliveryPlan, signal_date) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"jp_session": str(plan.jp_session.date()),
                   "us_session": str(plan.us_session.date()),
                   "signal_date": str(pd.Timestamp(signal_date).date()),
                   "sent_at": dt.datetime.now(dt.timezone.utc).isoformat()},
                  fh, indent=2)


def jp_sessions_between(after, before) -> list[pd.Timestamp]:
    """Tokyo sessions strictly between two session dates."""
    after, before = pd.Timestamp(after), pd.Timestamp(before)
    if before - after <= pd.Timedelta(days=1):
        return []
    jp, _ = _calendars()
    return list(jp.sessions_in_range(after + pd.Timedelta(days=1),
                                     before - pd.Timedelta(days=1)))
