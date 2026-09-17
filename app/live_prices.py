from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib import parse, request

import yfinance as yf

from app.config import get_settings
from app.models import LiveQuote, Market, Signal


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_metaapi_quote(symbol: str) -> LiveQuote:
    settings = get_settings()
    if not settings.metaapi_token or not settings.metaapi_account_id:
        raise RuntimeError("MetaApi live pricing is not configured")
    region = settings.metaapi_region.strip().lower()
    base = (
        f"https://mt-client-api-v1.{region}.agiliumtrade.ai/users/current/accounts/"
        f"{parse.quote(settings.metaapi_account_id, safe='')}/symbols/"
        f"{parse.quote(symbol, safe='')}"
    )
    headers = {"Accept": "application/json", "auth-token": settings.metaapi_token}
    call = request.Request(f"{base}/current-price?keepSubscription=true", headers=headers)
    with request.urlopen(call, timeout=12) as response:
        payload = json.loads(response.read())
    candle_call = request.Request(f"{base}/current-candles/1m?keepSubscription=true", headers=headers)
    with request.urlopen(candle_call, timeout=12) as response:
        candle = json.loads(response.read())
    bid = float(payload["bid"])
    ask = float(payload["ask"])
    midpoint = (bid + ask) / 2.0
    if bid <= 0 or ask <= 0 or ask < bid or midpoint <= 0:
        raise ValueError("Live quote contains invalid bid/ask values")
    candle_open = float(candle["open"])
    candle_close = float(candle["close"])
    micro_direction = "bullish" if candle_close > candle_open else "bearish" if candle_close < candle_open else "neutral"
    return LiveQuote(
        symbol=str(payload.get("symbol") or symbol),
        bid=bid,
        ask=ask,
        time=str(payload["time"]),
        spread_bps=round(((ask - bid) / midpoint) * 10_000.0, 3),
        micro_direction=micro_direction,
        micro_candle_time=str(candle.get("time") or ""),
    )


def _fetch_yahoo_quote(market: Market) -> LiveQuote:
    """Near-real-time fallback used only for timing, never as a broker spread claim."""
    settings = get_settings()
    frame = yf.download(
        market.symbol,
        interval="1m",
        period="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError("Yahoo returned no live 1m data")
    if getattr(frame.columns, "nlevels", 1) > 1:
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.lower)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    latest = frame.dropna(subset=["open", "close"]).iloc[-1]
    price = float(latest["close"])
    opened = float(latest["open"])
    estimated_bps = settings.execution_cost_bps_metal if market.category == "metal" else settings.execution_cost_bps_forex
    half_spread = price * estimated_bps / 20_000.0
    timestamp = frame.index[-1]
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    direction = "bullish" if price > opened else "bearish" if price < opened else "neutral"
    return LiveQuote(
        symbol=market.code,
        bid=round(price - half_spread, 8),
        ask=round(price + half_spread, 8),
        time=timestamp.isoformat(),
        spread_bps=estimated_bps,
        source="Yahoo 1m fallback",
        micro_direction=direction,
        micro_candle_time=timestamp.isoformat(),
    )


def fetch_live_quote(market: Market) -> LiveQuote:
    settings = get_settings()
    metaapi_error: Exception | None = None
    if settings.metaapi_token and settings.metaapi_account_id:
        try:
            return _fetch_metaapi_quote(market.code)
        except Exception as exc:
            metaapi_error = exc
    if settings.enable_yahoo_live_fallback:
        return _fetch_yahoo_quote(market)
    if metaapi_error:
        raise metaapi_error
    raise RuntimeError("No live-price source is configured")


def apply_live_entry(signal: Signal, now: datetime | None = None) -> tuple[bool, str | None]:
    """Validate timing against broker bid/ask and translate the plan to the executable price."""
    settings = get_settings()
    if signal.risk is None:
        return False, "Signal has no risk plan"
    try:
        quote = fetch_live_quote(signal.market)
    except Exception as exc:
        if settings.require_live_price_for_alerts:
            return False, f"Live price unavailable: {exc}"
        signal.warnings.append(f"Live price unavailable; using candle close: {exc}")
        return True, None

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    quote_time = _parse_time(quote.time)
    age = max(0.0, (current.astimezone(timezone.utc) - quote_time).total_seconds())
    max_age = max(120, settings.live_price_max_age_seconds) if quote.source == "Yahoo 1m fallback" else max(1, settings.live_price_max_age_seconds)
    if age > max_age:
        return False, f"Live quote is stale ({age:.0f}s old)"
    max_spread = (
        settings.live_price_max_spread_bps_metal
        if signal.market.category == "metal"
        else settings.live_price_max_spread_bps_forex
    )
    if quote.spread_bps > max_spread:
        return False, f"Live spread is too wide ({quote.spread_bps:.1f} bps)"
    if quote.micro_direction not in {None, "neutral", signal.direction}:
        return False, f"Live 1m timing is {quote.micro_direction}, against the {signal.direction} setup"

    executable = quote.ask if signal.direction == "bullish" else quote.bid
    atr = float(signal.indicators.get("atr") or 0.0)
    allowed_deviation = atr * max(0.0, settings.live_price_max_deviation_atr)
    deviation = abs(executable - signal.risk.entry)
    if atr <= 0 or deviation > allowed_deviation:
        return False, f"Live price moved too far from setup ({deviation:.5f})"

    shift = executable - signal.risk.entry
    signal.risk.entry = round(executable, 5)
    signal.risk.stop_loss = round(signal.risk.stop_loss + shift, 5)
    signal.risk.take_profit_1 = round(signal.risk.take_profit_1 + shift, 5)
    signal.risk.take_profit_2 = round(signal.risk.take_profit_2 + shift, 5)
    signal.live_quote = quote
    signal.reasons.append(
        f"{quote.source} confirmed entry timing with {quote.spread_bps:.1f} bps spread"
    )
    return True, None
