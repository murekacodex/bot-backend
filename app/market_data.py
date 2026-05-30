from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from app.config import get_settings
from app.models import Candle, Market

_cache: dict[str, tuple[datetime, pd.DataFrame]] = {}


def _cache_key(market: Market, interval: str, period: str) -> str:
    return f"{market.symbol}:{interval}:{period}"


def fetch_candles(market: Market, interval: str | None = None, period: str | None = None) -> pd.DataFrame:
    settings = get_settings()
    selected_interval = interval or settings.default_interval
    selected_period = period or settings.default_period
    key = _cache_key(market, selected_interval, selected_period)
    now = datetime.now(timezone.utc)

    cached = _cache.get(key)
    if cached and now - cached[0] < timedelta(seconds=settings.cache_ttl_seconds):
        return cached[1].copy()

    frame = yf.download(
        market.symbol,
        interval=selected_interval,
        period=selected_period,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if frame.empty:
        raise ValueError(f"No candle data returned for {market.code}")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.rename(columns=str.lower)
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame.index = pd.to_datetime(frame.index)

    _cache[key] = (now, frame)
    return frame.copy()


def dataframe_to_candles(frame: pd.DataFrame) -> list[Candle]:
    candles: list[Candle] = []
    for timestamp, row in frame.iterrows():
        candles.append(
            Candle(
                time=timestamp.isoformat(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else None,
            )
        )
    return candles
