from __future__ import annotations

import json
import hashlib
import hmac
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import parse, request

from app.config import get_settings
from app.models import Signal
from app.persistence import atomic_write_json, file_lock
from app.telegram_assistant import reset_conversation, start_assistant_task


_poller_started = False


def _api_call(method: str, payload: dict) -> dict:
    settings = get_settings()
    endpoint = f"https://api.telegram.org/bot{settings.telegram_bot_token}/{method}"
    encoded = parse.urlencode(payload).encode()
    with request.urlopen(request.Request(endpoint, data=encoded), timeout=35) as response:
        return json.loads(response.read())


def _auth_state() -> dict:
    path = Path(get_settings().telegram_auth_state_path)
    try:
        state = json.loads(path.read_text()) if path.exists() else {}
        state.setdefault("authorized", [])
        state.setdefault("pending", {})
        state.setdefault("offset", 0)
        state.setdefault("login_attempts", {})
        return state
    except (OSError, json.JSONDecodeError):
        return {"authorized": [], "pending": {}, "offset": 0, "login_attempts": {}}


def _save_auth_state(state: dict) -> None:
    path = Path(get_settings().telegram_auth_state_path)
    with file_lock(path):
        atomic_write_json(path, state)


def _authorized_chat_ids() -> list[str]:
    settings = get_settings()
    ids = {str(value) for value in _auth_state().get("authorized", [])}
    if settings.telegram_chat_id:
        ids.add(str(settings.telegram_chat_id))
    return sorted(ids)


def _password_matches(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def _send_text(chat_id: str, text: str) -> None:
    _api_call("sendMessage", {"chat_id": chat_id, "text": text})


def _handle_update(update: dict, state: dict) -> None:
    message = update.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    text = str(message.get("text") or "").strip()
    message_id = message.get("message_id")
    settings = get_settings()
    if not chat_id or not text:
        return
    if _password_matches(text, settings.telegram_access_keyword_hash):
        if message_id is not None:
            try:
                _api_call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
            except Exception:
                pass
        if settings.telegram_chat_id and hmac.compare_digest(chat_id, str(settings.telegram_chat_id)):
            _send_text(chat_id, "✅ Owner identity confirmed.")
        return
    if chat_id in _authorized_chat_ids():
        if text.startswith("/start"):
            _send_text(
                chat_id,
                "📈 Trade Alerts Bot\n\n"
                "Your access is active. You will receive viable-entry alerts with direction, timeframe, "
                "session, entry, stop loss, take profits, confidence, and score.\n\n"
                "Use /status at any time to confirm access.",
            )
        elif text.startswith("/status"):
            _send_text(chat_id, "✅ Trade-alert access is active on this device.")
        elif (
            text.startswith("/reset")
            and settings.telegram_chat_id
            and hmac.compare_digest(chat_id, str(settings.telegram_chat_id))
        ):
            reset_conversation(chat_id)
            _send_text(chat_id, "✅ Assistant conversation reset.")
        elif settings.telegram_chat_id and hmac.compare_digest(chat_id, str(settings.telegram_chat_id)):
            if not settings.openai_api_key:
                _send_text(chat_id, "⚠️ Telegram assistant is not configured yet. Add OPENAI_API_KEY on the server.")
            elif not start_assistant_task(chat_id, text, _send_text):
                _send_text(chat_id, "⏳ I’m still working on your previous request. Please wait for the result.")
        return
    attempts = state.setdefault("login_attempts", {})
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, settings.telegram_login_lockout_minutes))
    recent = []
    for value in attempts.get(chat_id, []):
        try:
            parsed = datetime.fromisoformat(str(value))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if parsed >= cutoff:
                recent.append(parsed.isoformat())
        except ValueError:
            continue
    attempts[chat_id] = recent
    if len(recent) >= max(1, settings.telegram_login_max_attempts):
        _send_text(chat_id, "Access temporarily locked after repeated failed attempts. Try again later.")
        return
    pending = state.setdefault("pending", {})
    if text.startswith("/start") or text.startswith("/login"):
        pending[chat_id] = "username"
        _send_text(
            chat_id,
            "📈 Trade Alerts Bot\n\n"
            "To activate alerts, enter your username and password when prompted. "
            "Your username and password messages are deleted after they are checked.\n\n"
            "Enter your username:",
        )
    elif pending.get(chat_id) == "username":
        if message_id is not None:
            try:
                _api_call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
            except Exception:
                pass
        if hmac.compare_digest(text, settings.telegram_access_username or ""):
            pending[chat_id] = "password"
            _send_text(chat_id, "Enter your password (this message will be deleted after checking):")
        else:
            attempts[chat_id].append(datetime.now(timezone.utc).isoformat())
            pending.pop(chat_id, None)
            _send_text(chat_id, "Access denied. Send /login to try again.")
    elif pending.get(chat_id) == "password":
        if message_id is not None:
            try:
                _api_call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
            except Exception:
                pass
        if _password_matches(text, settings.telegram_access_password_hash):
            authorized = {str(value) for value in state.setdefault("authorized", [])}
            authorized.add(chat_id)
            state["authorized"] = sorted(authorized)
            attempts.pop(chat_id, None)
            _send_text(chat_id, "✅ Access approved. Viable trade alerts are enabled.")
        else:
            attempts[chat_id].append(datetime.now(timezone.utc).isoformat())
            _send_text(chat_id, "Access denied. Send /login to try again.")
        pending.pop(chat_id, None)


def poll_telegram_updates() -> None:
    settings = get_settings()
    if not all((settings.telegram_bot_token, settings.telegram_access_username, settings.telegram_access_password_hash, settings.telegram_access_keyword_hash)):
        return
    while True:
        state = _auth_state()
        try:
            result = _api_call("getUpdates", {"offset": int(state.get("offset", 0)), "timeout": 25})
            for update in result.get("result", []):
                state["offset"] = int(update["update_id"]) + 1
                _handle_update(update, state)
                _save_auth_state(state)
        except Exception:
            time.sleep(5)


def start_telegram_poller() -> None:
    global _poller_started
    if _poller_started:
        return
    _poller_started = True
    threading.Thread(target=poll_telegram_updates, name="telegram-auth", daemon=True).start()


def _alert_key(signal: Signal, tier: str = "entry_ready") -> str:
    return ":".join((tier, signal.market.code, signal.interval, signal.direction, signal.timestamp))


def _alert_scope(signal: Signal) -> str:
    return f"{signal.market.code}:{signal.interval}"


def _read_alert_state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        state = {}
    state["sent"] = list(state.get("sent") or [])
    state["active_alerts"] = dict(state.get("active_alerts") or {})
    return state


def alert_was_sent(signal: Signal, tier: str) -> bool:
    """Check whether this candle has already emitted a lifecycle-stage alert."""
    state_path = Path(get_settings().telegram_alert_state_path)
    with file_lock(state_path):
        state = _read_alert_state(state_path)
    return _alert_key(signal, tier=tier) in state["sent"]


def _delete_alert_messages(messages: list[dict]) -> None:
    for message in messages:
        chat_id = message.get("chat_id")
        message_id = message.get("message_id")
        if chat_id is None or message_id is None:
            continue
        try:
            _api_call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
        except Exception as exc:
            print(json.dumps({"telegram_alert_delete_error": str(exc)}), flush=True)


def remove_outdated_alerts(active_keys: set[str], protected_scopes: set[str] | None = None) -> int:
    """Delete bot-owned trade alerts that are no longer viable in the current scan."""
    settings = get_settings()
    if not settings.telegram_bot_token:
        return 0
    state_path = Path(settings.telegram_alert_state_path)
    with file_lock(state_path):
        state = _read_alert_state(state_path)
        protected = protected_scopes or set()
        outdated = [
            (scope, alert) for scope, alert in state["active_alerts"].items()
            if str(alert.get("key")) not in active_keys and scope not in protected
        ]
        market_updates = list(state.get("market_update_messages") or []) if active_keys else []
    for _, alert in outdated:
        _delete_alert_messages(list(alert.get("messages") or []))
    _delete_alert_messages(market_updates)
    if not outdated and not market_updates:
        return 0
    with file_lock(state_path):
        state = _read_alert_state(state_path)
        for scope, alert in outdated:
            current = state["active_alerts"].get(scope)
            if current and current.get("key") == alert.get("key"):
                state["active_alerts"].pop(scope, None)
        if market_updates:
            state["market_update_messages"] = []
        atomic_write_json(state_path, state)
    return len(outdated)


def _message(signal: Signal, tier: str = "entry_ready") -> str:
    risk = signal.risk
    if risk is None:
        return ""
    heading = "🚨 TRADE RIPE — ENTRY READY" if tier == "entry_ready" else "👀 SWING SETUP — WATCHING TARGET"
    direction_icon = "🟢" if signal.direction == "bullish" else "🔴"
    strategy = getattr(signal, "strategy", None) or "Multi-factor setup"
    regime = getattr(signal, "regime", None)
    edge = getattr(signal, "historical_edge", None)
    sessions = " / ".join(part.replace("_", " ").title() for part in signal.session.active_sessions) if signal.session else "Unknown"
    live = getattr(signal, "live_quote", None)
    retest_target = (getattr(signal, "indicators", {}) or {}).get("retest_target")
    return "\n".join(
        (
            f"{heading}",
            f"{direction_icon} {signal.market.code} · {signal.direction.upper()} · {signal.interval.upper()}",
            "",
            "━━━━━━━━ TRADE LEVELS ━━━━━━━━",
            f"➡️  ENTRY   {risk.entry}",
            f"📍  RETEST TARGET   {retest_target if retest_target is not None else risk.entry}",
            f"🛑  STOP LOSS   {risk.stop_loss}",
            f"🎯  TAKE PROFIT 1   {risk.take_profit_1}",
            f"🎯  TAKE PROFIT 2   {risk.take_profit_2}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"Strategy: {strategy}",
            f"Regime: {regime.trend.title() if regime else 'Unknown'} / {regime.volatility.title() if regime else 'Unknown'} volatility",
            f"Session: {sessions}",
            f"Confidence: {signal.confidence}%",
            f"Score: {signal.score}",
            (
                f"Live timing: {live.source}, 1m {live.micro_direction}, spread {live.spread_bps:.1f} bps"
                if live else "Live timing: unavailable"
            ),
            (
                f"Recent edge: {edge.expectancy_r:+.2f}R from {edge.samples} samples"
                if edge and edge.sufficient_evidence
                else "Recent edge: collecting evidence"
            ),
            (
                "Live timing is confirmed. Verify price and spread with your broker before entering."
                if tier == "entry_ready"
                else (
                    f"Displacement through structure detected. Watching the retracement back to "
                    f"{retest_target if retest_target is not None else risk.entry}; wait for directional resumption "
                    "and the TRADE RIPE alert before entering."
                )
            ),
        )
    )


def _copy_keyboard(signal: Signal) -> str:
    risk = signal.risk
    if risk is None:
        return ""
    return json.dumps(
        {
            "inline_keyboard": [
                [
                    {"text": "📋 Copy Entry", "copy_text": {"text": str(risk.entry)}},
                    {"text": "📋 Copy Stop Loss", "copy_text": {"text": str(risk.stop_loss)}},
                ],
                [
                    {"text": "📋 Copy TP1", "copy_text": {"text": str(risk.take_profit_1)}},
                    {"text": "📋 Copy TP2", "copy_text": {"text": str(risk.take_profit_2)}},
                ],
            ]
        }
    )


def send_viable_entry_alert(signal: Signal, tier: str = "entry_ready") -> bool:
    settings = get_settings()
    if not settings.telegram_bot_token or signal.risk is None:
        return False

    state_path = Path(settings.telegram_alert_state_path)
    key = _alert_key(signal, tier=tier)
    scope = _alert_scope(signal)
    with file_lock(state_path):
        state = _read_alert_state(state_path)
        if key in state["sent"] or state["active_alerts"].get(scope, {}).get("key") == key:
            return False

    try:
        chat_ids = _authorized_chat_ids()
        if not chat_ids:
            return False
        delivered = False
        delivered_messages = []
        for chat_id in chat_ids:
            result = _api_call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": _message(signal, tier=tier),
                    "reply_markup": _copy_keyboard(signal),
                },
            )
            if result.get("ok"):
                delivered = True
                message_id = (result.get("result") or {}).get("message_id")
                if message_id is not None:
                    delivered_messages.append({"chat_id": chat_id, "message_id": message_id})
        if not delivered:
            return False
    except Exception as exc:
        print(json.dumps({"telegram_alert_error": str(exc)}), flush=True)
        return False

    with file_lock(state_path):
        state = _read_alert_state(state_path)
        previous = state["active_alerts"].get(scope)
        sent = state["sent"]
        if key not in sent:
            sent.append(key)
        state["sent"] = sent[-500:]
        state["active_alerts"][scope] = {
            "key": key,
            "market_code": signal.market.code,
            "interval": signal.interval,
            "messages": delivered_messages,
        }
        atomic_write_json(state_path, state)
    if previous and previous.get("key") != key:
        _delete_alert_messages(list(previous.get("messages") or []))
    return True


def send_market_update(signals: list[Signal]) -> bool:
    settings = get_settings()
    if not settings.telegram_bot_token or settings.telegram_market_update_hours <= 0 or not signals:
        return False
    state_path = Path(settings.telegram_alert_state_path)
    now = datetime.now(timezone.utc)
    with file_lock(state_path):
        state = _read_alert_state(state_path)
        last_value = state.get("last_market_update")
        if last_value:
            try:
                last_update = datetime.fromisoformat(str(last_value))
                if now - last_update < timedelta(hours=settings.telegram_market_update_hours):
                    return False
            except ValueError:
                pass

    strongest = sorted(signals, key=lambda item: abs(item.score), reverse=True)[:3]
    summary = "\n".join(
        f"• {signal.market.code}: {signal.direction}, score {signal.score}, confidence {signal.confidence}%"
        for signal in strongest
    )
    text = (
        "ℹ️ MARKET UPDATE\n\n"
        "The scanner is running, but no setup currently meets the entry or watchlist rules.\n\n"
        f"Strongest readings:\n{summary}"
    )
    try:
        chat_ids = _authorized_chat_ids()
        delivered = False
        delivered_messages = []
        for chat_id in chat_ids:
            result = _api_call("sendMessage", {"chat_id": chat_id, "text": text})
            if result.get("ok"):
                delivered = True
                message_id = (result.get("result") or {}).get("message_id")
                if message_id is not None:
                    delivered_messages.append({"chat_id": chat_id, "message_id": message_id})
        if not delivered:
            return False
    except Exception as exc:
        print(json.dumps({"telegram_market_update_error": str(exc)}), flush=True)
        return False

    with file_lock(state_path):
        state = _read_alert_state(state_path)
        previous_messages = list(state.get("market_update_messages") or [])
        state["last_market_update"] = now.isoformat()
        state["market_update_messages"] = delivered_messages
        atomic_write_json(state_path, state)
    _delete_alert_messages(previous_messages)
    return True
