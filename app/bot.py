import json
import time
from datetime import datetime, timezone

from app.analysis import analyze_market, trade_candidate_tier
from app.config import ANALYSIS_TIMEFRAMES, get_settings
from app.learning import AdaptiveSignalModel
from app.live_prices import apply_live_entry
from app.market_data import fetch_candles
from app.markets import MARKETS
from app.news import fetch_news_sentiment
from app.session import attach_market_status
from app.signal_journal import record_signal, resolve_signal_outcomes
from app.telegram_alerts import (
    _alert_key,
    _alert_scope,
    alert_was_sent,
    remove_outdated_alerts,
    send_market_update,
    send_viable_entry_alert,
)
from app.timeframes import timeframe_contexts


learner = AdaptiveSignalModel()
worker_status = {
    "running": False,
    "last_started_at": None,
    "last_completed_at": None,
    "last_error": None,
    "generated": 0,
    "errors": 0,
}


def run_once() -> list[dict]:
    worker_status.update(running=True, last_started_at=datetime.now(timezone.utc).isoformat(), last_error=None)
    settings = get_settings()
    signals = []
    analyzed_signals = []
    candidate_count = 0
    active_alert_keys: set[str] = set()
    protected_alert_scopes: set[str] = set()
    for market in [attach_market_status(market) for market in MARKETS.values()]:
        if settings.filter_closed_markets and not market.is_open:
            continue
        try:
            market_news = fetch_news_sentiment(market) if settings.enable_news_analysis else None
        except Exception as exc:
            market_news = None
            signals.append({"market": market.code, "source": "news", "error": str(exc)})
        for interval, period in ANALYSIS_TIMEFRAMES.items():
            try:
                frame = fetch_candles(market, interval=interval, period=period)
                current_close = float(frame.iloc[-1]["close"])
                current_time = frame.index[-1].to_pydatetime()
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)
                else:
                    current_time = current_time.astimezone(timezone.utc)
                learner.update_from_price(market.code, current_close, now=current_time)
                contexts, timeframe_warnings = timeframe_contexts(market, interval)
                signal = analyze_market(
                    market, frame, interval=interval, period=period,
                    news=market_news, learner=learner,
                    timeframes=contexts,
                )
                signal.warnings.extend(timeframe_warnings)
                learner.register_prediction(
                    market_code=market.code,
                    direction=signal.direction,
                    entry_price=signal.risk.entry if signal.risk else signal.last_candle.close,
                    features=signal.features or {},
                    interval=interval,
                    period=period,
                    timestamp=datetime.fromisoformat(signal.timestamp),
                )
                analyzed_signals.append(signal)
                tier = trade_candidate_tier(signal)
                delivered = False
                if tier == "entry_ready":
                    setup_sent = alert_was_sent(signal, tier="watchlist")
                    if not setup_sent:
                        candidate_count += 1
                        active_alert_keys.add(_alert_key(signal, tier="watchlist"))
                        delivered = send_viable_entry_alert(signal, tier="watchlist")
                    else:
                        live_valid, live_reason = apply_live_entry(signal)
                        if live_valid:
                            candidate_count += 1
                            active_alert_keys.add(_alert_key(signal, tier=tier))
                            delivered = send_viable_entry_alert(signal, tier=tier)
                        else:
                            candidate_count += 1
                            active_alert_keys.add(_alert_key(signal, tier="watchlist"))
                            signal.warnings.append(live_reason or "Live entry timing did not confirm")
                            if live_reason and ("unavailable" in live_reason.lower() or "stale" in live_reason.lower()):
                                protected_alert_scopes.add(_alert_scope(signal))
                elif tier == "watchlist":
                    candidate_count += 1
                    active_alert_keys.add(_alert_key(signal, tier=tier))
                    delivered = send_viable_entry_alert(signal, tier=tier)
                record_signal(
                    signal,
                    source="worker",
                    alert_tier=tier,
                    notification_delivered=delivered,
                )
                signals.append(signal.model_dump())
            except Exception as exc:
                signals.append({"market": market.code, "interval": interval, "error": str(exc)})
                protected_alert_scopes.add(f"{market.code}:{interval}")
    resolve_signal_outcomes()
    remove_outdated_alerts(active_alert_keys, protected_scopes=protected_alert_scopes)
    if analyzed_signals and candidate_count == 0:
        send_market_update(analyzed_signals)
    errors = sum(1 for result in signals if "error" in result)
    worker_status.update(
        running=False,
        last_completed_at=datetime.now(timezone.utc).isoformat(),
        generated=len(signals) - errors,
        errors=errors,
    )
    return signals


def main() -> None:
    settings = get_settings()
    while True:
        try:
            results = run_once()
            errors = [result for result in results if "error" in result]
            print(json.dumps({"generated": len(results) - len(errors), "errors": errors}), flush=True)
        except Exception as exc:
            worker_status.update(running=False, last_error=str(exc))
            print(json.dumps({"error": str(exc)}), flush=True)
        time.sleep(settings.bot_poll_seconds)


if __name__ == "__main__":
    main()
