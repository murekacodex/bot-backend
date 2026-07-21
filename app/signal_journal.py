from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.market_data import fetch_candles
from app.markets import get_market
from app.models import Signal, SignalLogEntry, SignalOutcome, SignalOutcomeStats


_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path() -> Path:
    return Path(get_settings().signal_log_path)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _default_state() -> dict[str, list[dict[str, Any]]]:
    return {"signals": []}


def _read_state() -> dict[str, list[dict[str, Any]]]:
    path = _state_path()
    if not path.exists():
        return _default_state()
    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError):
        return _default_state()
    signals = state.get("signals") if isinstance(state, dict) else None
    return {"signals": signals if isinstance(signals, list) else []}


def _write_state(state: dict[str, list[dict[str, Any]]]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True)
    temporary.replace(path)


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(entry.get("source")),
        str(entry.get("market_code")),
        str(entry.get("interval")),
        str(entry.get("period")),
        str(entry.get("candle_time")),
    )


def _signal_to_entry(signal: Signal, source: str) -> SignalLogEntry:
    return SignalLogEntry(
        id=str(uuid4()),
        source=source,
        generated_at=_now().isoformat(),
        candle_time=signal.timestamp,
        market_code=signal.market.code,
        market_name=signal.market.name,
        category=signal.market.category,
        interval=signal.interval,
        period=signal.period,
        direction=signal.direction,
        confidence=signal.confidence,
        score=signal.score,
        strategy=signal.strategy,
        entry_price=signal.last_candle.close,
        risk=signal.risk,
        features=signal.features,
        reasons=signal.reasons,
        warnings=signal.warnings,
        outcome=SignalOutcome(status="pending"),
    )


def record_signal(signal: Signal, source: str) -> SignalLogEntry:
    entry = _signal_to_entry(signal, source)
    entry_payload = entry.model_dump()
    with _lock:
        state = _read_state()
        key = _entry_key(entry_payload)
        for existing in state["signals"]:
            if _entry_key(existing) == key:
                return SignalLogEntry(**existing)
        state["signals"].append(entry_payload)
        _write_state(state)
    return entry


def record_signals(signals: list[Signal], source: str) -> list[SignalLogEntry]:
    return [record_signal(signal, source) for signal in signals]


def list_signal_log(market_code: str | None = None, limit: int = 100, include_pending: bool = True) -> list[SignalLogEntry]:
    normalized_market = market_code.upper().replace("/", "") if market_code else None
    with _lock:
        entries = list(_read_state()["signals"])
    if normalized_market:
        entries = [entry for entry in entries if entry.get("market_code") == normalized_market]
    if not include_pending:
        entries = [entry for entry in entries if entry.get("outcome", {}).get("status") == "resolved"]
    entries.sort(key=lambda entry: str(entry.get("generated_at", "")), reverse=True)
    return [SignalLogEntry(**entry) for entry in entries[:limit]]


def _resolve_entry(entry: dict[str, Any], now: datetime) -> bool:
    outcome = entry.get("outcome") or {}
    if outcome.get("status") == "resolved":
        return False

    settings = get_settings()
    try:
        generated_at = _parse_time(str(entry["generated_at"]))
    except (KeyError, ValueError):
        generated_at = now
    horizon = timedelta(hours=settings.signal_outcome_horizon_hours)
    if now - generated_at < horizon:
        return False

    market = get_market(str(entry["market_code"]))
    frame = fetch_candles(market, interval=str(entry["interval"]), period=str(entry["period"]))
    actual_price = float(frame.iloc[-1]["close"])
    entry_price = float(entry["entry_price"])
    move = (actual_price - entry_price) / max(entry_price, 1e-12)
    threshold = float(settings.signal_outcome_min_move_pct)
    direction = str(entry["direction"])

    if direction == "bullish":
        success = move >= threshold
    elif direction == "bearish":
        success = move <= -threshold
    else:
        success = abs(move) < threshold

    if abs(move) < threshold:
        label = "flat"
    elif move > 0:
        label = "up"
    else:
        label = "down"

    entry["outcome"] = SignalOutcome(
        status="resolved",
        resolved_at=now.isoformat(),
        actual_price=round(actual_price, 5),
        move_pct=round(move * 100, 4),
        label=label,
        success=success,
        reason=f"Resolved after {settings.signal_outcome_horizon_hours}h horizon",
    ).model_dump()
    return True


def resolve_signal_outcomes() -> int:
    current = _now()
    updated = 0
    with _lock:
        state = _read_state()
        for entry in state["signals"]:
            try:
                if _resolve_entry(entry, current):
                    updated += 1
            except Exception as exc:
                entry["outcome"] = SignalOutcome(status="pending", reason=f"Resolution failed: {exc}").model_dump()
        if updated:
            _write_state(state)
    return updated


def signal_outcome_stats() -> SignalOutcomeStats:
    with _lock:
        entries = list(_read_state()["signals"])

    total = len(entries)
    resolved_entries = [entry for entry in entries if entry.get("outcome", {}).get("status") == "resolved"]
    successes = sum(1 for entry in resolved_entries if entry.get("outcome", {}).get("success") is True)
    failures = sum(1 for entry in resolved_entries if entry.get("outcome", {}).get("success") is False)

    by_market: dict[str, dict[str, int | float | None]] = {}
    for entry in entries:
        market = str(entry.get("market_code"))
        bucket = by_market.setdefault(market, {"total": 0, "pending": 0, "resolved": 0, "successes": 0, "failures": 0, "accuracy": None})
        bucket["total"] = int(bucket["total"] or 0) + 1
        outcome = entry.get("outcome", {})
        if outcome.get("status") == "resolved":
            bucket["resolved"] = int(bucket["resolved"] or 0) + 1
            if outcome.get("success") is True:
                bucket["successes"] = int(bucket["successes"] or 0) + 1
            else:
                bucket["failures"] = int(bucket["failures"] or 0) + 1
        else:
            bucket["pending"] = int(bucket["pending"] or 0) + 1

    for bucket in by_market.values():
        resolved = int(bucket["resolved"] or 0)
        bucket["accuracy"] = round(int(bucket["successes"] or 0) / resolved, 4) if resolved else None

    return SignalOutcomeStats(
        total=total,
        pending=total - len(resolved_entries),
        resolved=len(resolved_entries),
        successes=successes,
        failures=failures,
        accuracy=round(successes / len(resolved_entries), 4) if resolved_entries else None,
        by_market=by_market,
    )
