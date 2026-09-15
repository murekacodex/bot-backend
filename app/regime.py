from __future__ import annotations

import numpy as np
import pandas as pd

from app.models import MarketRegime


def classify_regime(frame: pd.DataFrame) -> MarketRegime:
    """Classify trend, volatility, and volume using only completed candles."""
    if len(frame) < 30:
        return MarketRegime(trend="unknown", volatility="unknown", volume="unknown")
    data = frame.copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    returns = close.pct_change()
    true_range = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    atr_ratio = atr / close.replace(0, np.nan)
    current_atr = float(atr_ratio.iloc[-1]) if pd.notna(atr_ratio.iloc[-1]) else 0.0
    history = atr_ratio.dropna().iloc[-100:]
    percentile = float((history <= current_atr).mean()) if len(history) else 0.5

    ema_fast = close.ewm(span=21, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    slope = (float(ema_slow.iloc[-1]) - float(ema_slow.iloc[-6])) / max(abs(float(close.iloc[-1])), 1e-12)
    separation = (float(ema_fast.iloc[-1]) - float(ema_slow.iloc[-1])) / max(abs(float(close.iloc[-1])), 1e-12)
    trend_strength = min(100.0, (abs(slope) + abs(separation)) * 20_000)
    if trend_strength < 18:
        trend = "sideways"
    elif slope > 0 and separation > 0:
        trend = "bullish"
    elif slope < 0 and separation < 0:
        trend = "bearish"
    else:
        trend = "transition"

    volatility = "low" if percentile < 0.3 else "high" if percentile > 0.75 else "normal"
    if "volume" not in data or data["volume"].dropna().empty or float(data["volume"].iloc[-20:].sum()) == 0:
        volume = "unavailable"
    else:
        recent_volume = float(data["volume"].iloc[-5:].mean())
        baseline_volume = float(data["volume"].iloc[-20:].mean())
        ratio = recent_volume / max(baseline_volume, 1e-12)
        volume = "rising" if ratio > 1.15 else "falling" if ratio < 0.85 else "stable"
    realized = returns.iloc[-20:].std()
    return MarketRegime(
        trend=trend,
        volatility=volatility,
        volume=volume,
        trend_strength=round(trend_strength, 2),
        atr_percentile=round(percentile, 4),
        realized_volatility=round(float(realized) if pd.notna(realized) else 0.0, 6),
    )
