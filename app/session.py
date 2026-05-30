from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import get_settings
from app.models import Market, SessionSignal


@dataclass(frozen=True)
class SessionWindow:
    open_day: int
    open_time: time
    close_day: int
    close_time: time


FOREX_SESSION = SessionWindow(open_day=6, open_time=time(22, 0), close_day=4, close_time=time(22, 0))
CME_GLOBEX_SESSION = SessionWindow(open_day=6, open_time=time(23, 0), close_day=4, close_time=time(22, 0))

SESSION_DISPLAY = {
    "asia": "Asia",
    "london": "London",
    "new_york": "New York",
}


def _session_timezone() -> ZoneInfo:
    settings = get_settings()
    try:
        return ZoneInfo(settings.session_timezone)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _parse_session_time(value: str, fallback: time) -> time:
    try:
        hour, minute = value.strip().split(":", maxsplit=1)
        return time(int(hour), int(minute))
    except (AttributeError, TypeError, ValueError):
        return fallback


def session_windows() -> dict[str, tuple[time, time]]:
    settings = get_settings()
    return {
        "asia": (
            _parse_session_time(settings.asia_session_open, time(0, 0)),
            _parse_session_time(settings.asia_session_close, time(8, 0)),
        ),
        "london": (
            _parse_session_time(settings.london_session_open, time(7, 0)),
            _parse_session_time(settings.london_session_close, time(16, 0)),
        ),
        "new_york": (
            _parse_session_time(settings.new_york_session_open, time(13, 0)),
            _parse_session_time(settings.new_york_session_close, time(21, 0)),
        ),
    }


def _current_session_time(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(_session_timezone())


def _time_in_range(current: time, open_time: time, close_time: time) -> bool:
    if open_time <= close_time:
        return open_time <= current < close_time
    return current >= open_time or current < close_time


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


def active_currency_sessions(now: datetime | None = None) -> list[str]:
    current = _current_session_time(now)

    active: list[str] = []
    for session, (open_time, close_time) in session_windows().items():
        if _time_in_range(current.time(), open_time, close_time):
            active.append(session)
    return active


def session_label(sessions: list[str]) -> str:
    if not sessions:
        return "quiet"
    if len(sessions) == 1:
        return sessions[0]
    return "+".join(sessions)


def display_session(session: str) -> str:
    if session in SESSION_DISPLAY:
        return SESSION_DISPLAY[session]
    if session == "quiet":
        return "Quiet"
    if "+" in session:
        return " / ".join(SESSION_DISPLAY.get(part, part.title()) for part in session.split("+"))
    return session.title()


def preferred_sessions_for_market(market: Market) -> list[str]:
    if market.preferred_sessions:
        return market.preferred_sessions
    if market.code.endswith("JPY") or market.code.startswith("AUD") or market.code.startswith("NZD"):
        return ["asia", "london"]
    if market.code.startswith("EUR") or market.code.startswith("GBP"):
        return ["london", "new_york"]
    if market.code.startswith("USD") or market.code.endswith("CAD"):
        return ["new_york", "london"]
    if market.category == "metal":
        return ["london", "new_york"]
    return ["london"]


def _next_session_start(preferred: list[str], now: datetime | None = None) -> tuple[str | None, datetime | None]:
    current = _current_session_time(now)
    windows = session_windows()
    candidates: list[tuple[str, datetime]] = []

    for session in preferred:
        if session not in windows:
            continue
        open_time, _ = windows[session]
        candidate = current.replace(
            hour=open_time.hour,
            minute=open_time.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= current:
            candidate += timedelta(days=1)
        candidates.append((session, candidate))

    if not candidates:
        return None, None
    return min(candidates, key=lambda item: item[1])


def _format_session_start(session_start: datetime) -> str:
    label = session_start.strftime("%H:%M %Z").strip()
    return label or session_start.isoformat()


def recommend_session_entry(market: Market, now: datetime | None = None) -> SessionSignal:
    settings = get_settings()
    active = active_currency_sessions(now)
    preferred = preferred_sessions_for_market(market)
    active_matches = [session for session in active if session in preferred]
    current_session = session_label(active)
    next_session, next_start = _next_session_start(preferred, now)

    if active_matches:
        alignment = "aligned"
        match = session_label(active_matches)
        suggestion = f"Best entry window is {display_session(match)} for this market."
        score_adjustment = settings.session_alignment_boost
        confidence = min(95, 70 + len(active_matches) * 10)
        reasons = [f"{display_session(match)} liquidity aligns with this setup"]
    elif active:
        alignment = "off_session"
        session_name = next_session or preferred[0]
        if next_start:
            suggestion = f"Wait for {display_session(session_name)} at {_format_session_start(next_start)} for a cleaner entry."
        else:
            suggestion = f"Wait for {display_session(session_name)} for a cleaner entry."
        score_adjustment = -settings.session_offsession_penalty
        confidence = 55
        reasons = [f"Current session is {display_session(current_session)}, but this market prefers {display_session(session_name)}"]
    else:
        alignment = "quiet"
        session_name = next_session or preferred[0]
        if next_start:
            suggestion = f"Market is in a quiet window; wait for {display_session(session_name)} at {_format_session_start(next_start)}."
        else:
            suggestion = f"Market is in a quiet window; wait for {display_session(session_name)}."
        score_adjustment = -settings.session_offsession_penalty
        confidence = 50
        reasons = ["No major FX session is currently active"]

    return SessionSignal(
        current_session=current_session,
        active_sessions=active,
        preferred_sessions=preferred,
        next_session=next_session,
        next_session_start=next_start.isoformat() if next_start else None,
        alignment=alignment,
        suggestion=suggestion,
        score_adjustment=round(score_adjustment, 4),
        confidence=confidence,
        reasons=reasons,
    )
