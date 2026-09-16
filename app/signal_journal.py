from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.config import get_settings
from app.market_data import fetch_candles
from app.markets import get_market
from app.models import HistoricalEdge, Signal, SignalLogEntry, SignalOutcome, SignalOutcomeStats
from app.persistence import atomic_write_json, file_lock


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
    atomic_write_json(path, state)


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(entry.get("market_code")),
        str(entry.get("interval")),
        str(entry.get("period")),
        str(entry.get("candle_time")),
        str(entry.get("direction")),
    )


def _unique_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda item: str(item.get("generated_at", ""))):
        key = _entry_key(entry)
        existing = unique.get(key)
        if existing is None or entry.get("outcome", {}).get("status") == "resolved":
            unique[key] = entry
    return list(unique.values())


def calibrated_confidence(market_code: str, interval: str, direction: str, fallback: int) -> int:
    """Return a conservative empirical hit-rate estimate for this exact regime."""
    try:
        with _lock, file_lock(_state_path()):
            entries = _unique_entries(list(_read_state()["signals"]))
    except Exception:
        return min(fallback, 65)
    resolved = [
        entry for entry in entries
        if entry.get("market_code") == market_code
        and entry.get("interval") == interval
        and entry.get("direction") == direction
        and entry.get("outcome", {}).get("status") == "resolved"
    ]
    if len(resolved) < 10:
        return min(fallback, 65)
    successes = sum(entry.get("outcome", {}).get("success") is True for entry in resolved)
    return max(5, min(95, round(((successes + 5) / (len(resolved) + 10)) * 100)))


def _entry_r_multiple(entry: dict[str, Any]) -> float:
    outcome = entry.get("outcome") or {}
    label = outcome.get("label")
    risk = entry.get("risk") or {}
    if label == "stop_loss":
        return -1.0
    if label == "take_profit_1":
        return float(risk.get("risk_reward") or 1.5)
    entry_price = float(entry.get("entry_price") or 0.0)
    actual_price = float(outcome.get("actual_price") or entry_price)
    stop_loss = risk.get("stop_loss")
    if not entry_price or stop_loss is None:
        return 1.0 if outcome.get("success") is True else -1.0
    risk_distance = abs(entry_price - float(stop_loss))
    if risk_distance <= 0:
        return 1.0 if outcome.get("success") is True else -1.0
    move = actual_price - entry_price
    if entry.get("direction") == "bearish":
        move *= -1
    reward_cap = float(risk.get("risk_reward") or 1.5)
    return max(-1.0, min(reward_cap, move / risk_distance))


def rolling_performance_edge(
    market_code: str,
    direction: str,
    *,
    strategy: str | None = None,
    regime: str | None = None,
) -> HistoricalEdge:
    settings = get_settings()
    try:
        with _lock, file_lock(_state_path()):
            entries = _unique_entries(list(_read_state()["signals"]))
    except Exception:
        entries = []
    resolved = [
        entry for entry in entries
        if entry.get("market_code") == market_code
        and entry.get("direction") == direction
        and entry.get("outcome", {}).get("status") == "resolved"
    ]
    resolved.sort(key=lambda entry: str(entry.get("resolved_at") or entry.get("generated_at") or ""), reverse=True)
    scope = "market_direction"
    specific = [
        entry for entry in resolved
        if (not strategy or entry.get("selected_strategy") == strategy)
        and (not regime or entry.get("market_regime") == regime)
    ]
    if len(specific) >= settings.edge_min_samples:
        resolved = specific
        scope = "market_direction_strategy_regime"
    resolved = resolved[: max(1, settings.edge_max_samples)]
    sufficient = len(resolved) >= settings.edge_min_samples
    if not sufficient:
        return HistoricalEdge(
            scope=scope, samples=len(resolved), effective_samples=0.0, expectancy_r=0.0,
            win_rate=0.0, score_adjustment=0.0, sufficient_evidence=False,
        )
    decay = max(0.5, min(1.0, settings.edge_decay))
    weights = [decay ** index for index in range(len(resolved))]
    effective_samples = sum(weights)
    weighted_r = sum(_entry_r_multiple(entry) * weight for entry, weight in zip(resolved, weights))
    weighted_wins = sum(
        weight for entry, weight in zip(resolved, weights)
        if entry.get("outcome", {}).get("success") is True
    )
    expectancy = weighted_r / max(effective_samples + max(settings.edge_prior_samples, 0.0), 1e-12)
    adjustment = math.tanh(expectancy) * max(settings.edge_max_score_adjustment, 0.0)
    return HistoricalEdge(
        scope=scope,
        samples=len(resolved),
        effective_samples=round(effective_samples, 2),
        expectancy_r=round(expectancy, 4),
        win_rate=round(weighted_wins / max(effective_samples, 1e-12), 4),
        score_adjustment=round(adjustment, 4),
        sufficient_evidence=True,
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
        selected_strategy=signal.selected_strategy,
        market_regime=signal.regime.trend if signal.regime else None,
        volatility_regime=signal.regime.volatility if signal.regime else None,
        factor_composite=signal.factors.composite if signal.factors else None,
        entry_price=signal.risk.entry if signal.risk else signal.last_candle.close,
        risk=signal.risk,
        features=signal.features,
        reasons=signal.reasons,
        warnings=signal.warnings,
        outcome=SignalOutcome(status="pending"),
    )


def record_signal(signal: Signal, source: str) -> SignalLogEntry:
    entry = _signal_to_entry(signal, source)
    entry_payload = entry.model_dump()
    with _lock, file_lock(_state_path()):
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
    with _lock, file_lock(_state_path()):
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
        signal_time = _parse_time(str(entry.get("candle_time") or entry["generated_at"]))
    except (KeyError, ValueError):
        signal_time = now
    horizon_hours = {
        "1m": 1,
        "5m": 3,
        "15m": 6,
        "30m": 12,
        "1h": 24,
        "4h": 72,
        "1d": 168,
    }.get(str(entry.get("interval")), settings.signal_outcome_horizon_hours)
    horizon = timedelta(hours=horizon_hours)
    target_time = signal_time + horizon

    market = get_market(str(entry["market_code"]))
    frame = fetch_candles(market, interval=str(entry["interval"]), period=str(entry["period"]))
    timestamps = pd.to_datetime(frame.index, utc=True)
    evaluation = frame.loc[(timestamps >= signal_time) & (timestamps <= min(now, target_time))]
    risk = entry.get("risk") or {}
    direction = str(entry["direction"])
    stop_loss = risk.get("stop_loss")
    take_profit = risk.get("take_profit_1")
    for candle_time, candle in evaluation.iterrows():
        if direction == "bullish" and stop_loss is not None and float(candle["low"]) <= float(stop_loss):
            return _set_path_outcome(entry, now, float(stop_loss), False, "stop_loss", candle_time)
        if direction == "bearish" and stop_loss is not None and float(candle["high"]) >= float(stop_loss):
            return _set_path_outcome(entry, now, float(stop_loss), False, "stop_loss", candle_time)
        if direction == "bullish" and take_profit is not None and float(candle["high"]) >= float(take_profit):
            return _set_path_outcome(entry, now, float(take_profit), True, "take_profit_1", candle_time)
        if direction == "bearish" and take_profit is not None and float(candle["low"]) <= float(take_profit):
            return _set_path_outcome(entry, now, float(take_profit), True, "take_profit_1", candle_time)

    if now < target_time:
        return False
    eligible = frame.loc[timestamps >= target_time]
    if eligible.empty:
        raise ValueError(f"No candle is available at the {horizon_hours}h outcome horizon")
    actual_price = float(eligible.iloc[0]["close"])
    entry_price = float(entry["entry_price"])
    move = (actual_price - entry_price) / max(entry_price, 1e-12)
    threshold = float(settings.signal_outcome_min_move_pct)

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
        reason=f"Neither stop nor target was hit; resolved at the {horizon_hours}h timeframe horizon",
    ).model_dump()
    return True


def _set_path_outcome(entry: dict[str, Any], now: datetime, actual_price: float, success: bool, label: str, candle_time: Any) -> bool:
    entry_price = float(entry["entry_price"])
    move = (actual_price - entry_price) / max(entry_price, 1e-12)
    entry["outcome"] = SignalOutcome(
        status="resolved",
        resolved_at=now.isoformat(),
        actual_price=round(actual_price, 5),
        move_pct=round(move * 100, 4),
        label=label,
        success=success,
        reason=f"{label.replace('_', ' ').title()} reached first at {pd.Timestamp(candle_time).isoformat()}",
    ).model_dump()
    return True


def resolve_signal_outcomes() -> int:
    current = _now()
    updated = 0
    with _lock, file_lock(_state_path()):
        state = _read_state()
        for entry in state["signals"]:
            try:
                if _resolve_entry(entry, current):
                    updated += 1
            except Exception as exc:
                entry["outcome"] = SignalOutcome(status="pending", reason=f"Resolution failed: {exc}").model_dump()
        if updated or any(
            str(entry.get("outcome", {}).get("reason") or "").startswith("Resolution failed:")
            for entry in state["signals"]
        ):
            _write_state(state)
    return updated


def signal_outcome_stats(now: datetime | None = None) -> SignalOutcomeStats:
    with _lock, file_lock(_state_path()):
        entries = _unique_entries(list(_read_state()["signals"]))

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

    cutoff = (now or _now()) - timedelta(hours=24)
    recent_by_market: dict[str, dict[str, int]] = {}
    for entry in resolved_entries:
        outcome = entry.get("outcome", {})
        try:
            resolved_at = _parse_time(str(outcome.get("resolved_at")))
        except (TypeError, ValueError):
            continue
        if resolved_at < cutoff:
            continue
        market = str(entry.get("market_code"))
        bucket = recent_by_market.setdefault(market, {"resolved": 0, "successes": 0})
        bucket["resolved"] += 1
        if outcome.get("success") is True:
            bucket["successes"] += 1

    best_market_24h = None
    if recent_by_market:
        code, bucket = max(
            recent_by_market.items(),
            key=lambda item: (
                item[1]["successes"] / max(item[1]["resolved"], 1),
                item[1]["successes"],
                item[1]["resolved"],
                item[0],
            ),
        )
        best_market_24h = {
            "code": code,
            "accuracy": round(bucket["successes"] / bucket["resolved"], 4),
            "successes": bucket["successes"],
            "resolved": bucket["resolved"],
            "window_hours": 24,
        }

    return SignalOutcomeStats(
        total=total,
        pending=total - len(resolved_entries),
        resolved=len(resolved_entries),
        successes=successes,
        failures=failures,
        accuracy=round(successes / len(resolved_entries), 4) if resolved_entries else None,
        by_market=by_market,
        best_market_24h=best_market_24h,
    )
