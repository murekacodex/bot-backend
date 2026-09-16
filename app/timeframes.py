from app.analysis import summarize_timeframe
from app.market_data import fetch_candles
from app.models import Market


TIMEFRAME_MAP = {
    "1m": {"lower": None, "higher": ("5m", "1d")},
    "5m": {"lower": None, "higher": ("15m", "1d")},
    "15m": {"lower": ("5m", "1d"), "higher": ("1h", "5d")},
    "30m": {"lower": ("15m", "1d"), "higher": ("1h", "5d")},
    "1h": {"lower": ("15m", "1d"), "higher": ("1d", "3mo")},
    "4h": {"lower": ("1h", "5d"), "higher": ("1d", "3mo")},
    "1d": {"lower": ("1h", "5d"), "higher": ("1wk", "1y")},
}


def timeframe_contexts(market: Market, interval: str) -> tuple[dict, list[str]]:
    contexts = {}
    warnings = []
    selected = TIMEFRAME_MAP.get(interval, TIMEFRAME_MAP["1h"])
    for label in ("higher", "lower"):
        selection = selected.get(label)
        if not selection:
            continue
        selected_interval, selected_period = selection
        try:
            frame = fetch_candles(market, interval=selected_interval, period=selected_period)
            contexts[label] = summarize_timeframe(frame, interval=selected_interval, period=selected_period)
        except Exception as exc:
            warnings.append(f"{label.title()} timeframe {selected_interval} unavailable: {exc}")
    return contexts, warnings
