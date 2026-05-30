from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

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

SESSION_WINDOWS: dict[str, tuple[time, time]] = {
    "asia": (time(0, 0), time(8, 0)),
    "london": (time(7, 0), time(16, 0)),
    "new_york": (time(13, 0), time(21, 0)),
}

SESSION_DISPLAY = {
    "asia": "Asia",
    "london": "London",
    "new_york": "New York",
}


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
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    active: list[str] = []
    for session, (open_time, close_time) in SESSION_WINDOWS.items():
        if open_time <= current.time() < close_time:
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


def recommend_session_entry(market: Market, now: datetime | None = None) -> SessionSignal:
    settings = get_settings()
    active = active_currency_sessions(now)
    preferred = preferred_sessions_for_market(market)
    active_matches = [session for session in active if session in preferred]
    current_session = session_label(active)

    if active_matches:
        alignment = "aligned"
        match = session_label(active_matches)
        suggestion = f"Best entry window is {display_session(match)} for this market."
        score_adjustment = settings.session_alignment_boost
        confidence = min(95, 70 + len(active_matches) * 10)
        reasons = [f"{display_session(match)} liquidity aligns with this setup"]
    elif active:
        alignment = "off_session"
        next_session = preferred[0]
        suggestion = f"Wait for {display_session(next_session)} for a cleaner entry."
        score_adjustment = -settings.session_offsession_penalty
        confidence = 55
        reasons = [f"Current session is {display_session(current_session)}, but this market prefers {display_session(next_session)}"]
    else:
        alignment = "quiet"
        next_session = preferred[0]
        suggestion = f"Market is in a quiet window; wait for {display_session(next_session)}."
        score_adjustment = -settings.session_offsession_penalty
        confidence = 50
        reasons = ["No major FX session is currently active"]

    return SessionSignal(
        current_session=current_session,
        active_sessions=active,
        preferred_sessions=preferred,
        alignment=alignment,
        suggestion=suggestion,
        score_adjustment=round(score_adjustment, 4),
        confidence=confidence,
        reasons=reasons,
    )
