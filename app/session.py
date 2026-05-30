from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

from app.models import Market


@dataclass(frozen=True)
class SessionWindow:
    open_day: int
    open_time: time
    close_day: int
    close_time: time


FOREX_SESSION = SessionWindow(open_day=6, open_time=time(22, 0), close_day=4, close_time=time(22, 0))
CME_GLOBEX_SESSION = SessionWindow(open_day=6, open_time=time(23, 0), close_day=4, close_time=time(22, 0))


def _in_weekly_session(now: datetime, window: SessionWindow) -> bool:
    current_day = now.weekday()
    current_time = now.time()

    if window.open_day < window.close_day:
        if current_day < window.open_day or current_day > window.close_day:
            return False
        if current_day == window.open_day and current_time < window.open_time:
            return False
        if current_day == window.close_day and current_time >= window.close_time:
            return False
        return True

    if current_day == window.open_day:
        return current_time >= window.open_time
    if current_day == window.close_day:
        return current_time < window.close_time
    return current_day > window.open_day or current_day < window.close_day


def market_is_open(market: Market, now: datetime | None = None) -> tuple[bool, str | None]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    if market.session == "forex":
        if _in_weekly_session(current, FOREX_SESSION):
            return True, None
        return False, "Forex markets are closed for the weekend"

    if market.session == "cme_globex":
        if _in_weekly_session(current, CME_GLOBEX_SESSION):
            return True, None
        return False, "Gold futures are outside the Globex session"

    return True, None


def attach_market_status(market: Market, now: datetime | None = None) -> Market:
    is_open, reason = market_is_open(market, now=now)
    return market.model_copy(update={"is_open": is_open, "closed_reason": reason})
