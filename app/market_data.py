from datetime import datetime, timedelta, timezone
import warnings

import pandas as pd
import yfinance as yf

from app.config import get_settings
from app.models import Candle, Market

_cache: dict[str, tuple[datetime, pd.DataFrame]] = {}

_INTERVAL_DURATION = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1wk": timedelta(days=7),
}


def interval_duration(interval: str) -> timedelta:
    return _INTERVAL_DURATION.get(interval, timedelta(hours=1))

warnings.filterwarnings(
    "ignore",
    message="The 'generic' unit for NumPy timedelta is deprecated.*",
    category=DeprecationWarning,
    module=r"yfinance\.utils",
)


def _cache_key(market: Market, interval: str, period: str) -> str:
    return f"{market.symbol}:{interval}:{period}"


def completed_candles(frame: pd.DataFrame, interval: str, now: datetime | None = None) -> pd.DataFrame:
    """Exclude a provider's still-forming final bar so signals cannot repaint."""
    if frame.empty:
        return frame
    duration = _INTERVAL_DURATION.get(interval)
    if duration is None:
        return frame
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    final_open = pd.Timestamp(frame.index[-1])
    if final_open.tzinfo is None:
        final_open = final_open.tz_localize(timezone.utc)
    else:
        final_open = final_open.tz_convert(timezone.utc)
    if final_open.to_pydatetime() + duration > current:
        return frame.iloc[:-1]
    return frame


def fetch_candles(market: Market, interval: str | None = None, period: str | None = None) -> pd.DataFrame:
    settings = get_settings()
    selected_interval = interval or settings.default_interval
    selected_period = period or settings.default_period
    key = _cache_key(market, selected_interval, selected_period)
    now = datetime.now(timezone.utc)

    cached = _cache.get(key)
    if cached and now - cached[0] < timedelta(seconds=settings.cache_ttl_seconds):
        return cached[1].copy()

    download_interval = "1h" if selected_interval == "4h" else selected_interval
    frame = yf.download(
        market.symbol,
        interval=download_interval,
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

    if selected_interval == "4h":
        frame = frame.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "high", "low", "close"])

    frame = completed_candles(frame, selected_interval, now=now)
    if frame.empty:
        raise ValueError(f"No completed candle data returned for {market.code}")

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
