"""
Market-calendar helpers.

GitHub's cron only sets the *earliest* time a workflow may start; runs can be
delayed by hours. So the bot never trusts "the time it woke up". It asks
Alpaca's market calendar which trading session it is working on, and checks
the clock against that session's real open/close (which already accounts for
US daylight saving, holidays and half days).
"""
import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Session:
    date: dt.date
    open: dt.datetime   # timezone-aware, America/New_York
    close: dt.datetime  # timezone-aware, America/New_York (13:00 on half days)


def now_ny() -> dt.datetime:
    return dt.datetime.now(NY)


def get_sessions(trading_client, start: dt.date, end: dt.date) -> list:
    """Trading sessions between start and end (inclusive), oldest first."""
    days = trading_client.get_calendar(GetCalendarRequest(start=start, end=end))
    # Alpaca returns naive datetimes in New York local time; attach the zone.
    return [
        Session(date=d.date, open=d.open.replace(tzinfo=NY), close=d.close.replace(tzinfo=NY))
        for d in days
    ]


def session_on(trading_client, day: dt.date):
    """The session on `day`, or None if the market is closed that day."""
    sessions = get_sessions(trading_client, day, day)
    return sessions[0] if sessions else None


def previous_session(trading_client, day: dt.date) -> Session:
    """The last session strictly before `day`."""
    sessions = get_sessions(trading_client, day - dt.timedelta(days=10), day - dt.timedelta(days=1))
    if not sessions:
        raise RuntimeError(f"No trading session found in the 10 days before {day}")
    return sessions[-1]
