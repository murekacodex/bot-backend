from __future__ import annotations

import pandas as pd

from app.models import FactorBreakdown, Market, MarketRegime, StrategyEvaluation


def _status(score: float) -> str:
    return "entry_ready" if score >= 70 else "watchlist" if score >= 52 else "inactive"


def evaluate_strategies(
    data: pd.DataFrame,
    market: Market,
    regime: MarketRegime,
    factors: FactorBreakdown,
) -> list[StrategyEvaluation]:
    if len(data) < 30:
        return []
    current = data.iloc[-1]
    previous = data.iloc[-2]
    atr = float(current.atr) if pd.notna(current.atr) else 0.0
    close = float(current.close)
    results = [
        _trend_pullback(current, previous, atr, close, regime, factors),
        _volatility_breakout(data, current, atr, close, regime, factors),
        _range_reversion(data, current, atr, close, regime, factors),
    ]
    return sorted(results, key=lambda item: item.score, reverse=True)


def _trend_pullback(current, previous, atr: float, close: float, regime: MarketRegime, factors: FactorBreakdown) -> StrategyEvaluation:
    direction = regime.trend if regime.trend in {"bullish", "bearish"} else "neutral"
    reasons: list[str] = []
    score = 20.0
    if direction != "neutral":
        score += 25
        reasons.append(f"{direction.title()} EMA trend")
    pullback_distance = abs(close - float(current.ema_21))
    if atr > 0 and pullback_distance <= atr * 0.75:
        score += 20
        reasons.append("Price pulled back near EMA 21")
    rsi = float(current.rsi) if pd.notna(current.rsi) else 50.0
    rsi_reset = 42 <= rsi <= 62
    if rsi_reset:
        score += 10
        reasons.append("RSI reset without an extreme")
    confirming = (direction == "bullish" and current.close > current.open) or (
        direction == "bearish" and current.close < current.open
    )
    if confirming:
        score += 15
        reasons.append("Latest candle confirms trend direction")
    if regime.volatility == "high":
        score -= 15
    score += min(10.0, abs(factors.composite) / 8.0)
    return StrategyEvaluation(
        name="trend_pullback", direction=direction, score=round(max(0, min(100, score)), 2),
        status=_status(score), reasons=reasons, suitable_regimes=["bullish", "bearish"],
    )


def _volatility_breakout(data, current, atr: float, close: float, regime: MarketRegime, factors: FactorBreakdown) -> StrategyEvaluation:
    prior = data.iloc[-21:-1]
    range_high = float(prior.high.max())
    range_low = float(prior.low.min())
    bullish = close > range_high
    bearish = close < range_low
    direction = "bullish" if bullish else "bearish" if bearish else "neutral"
    score = 20.0
    reasons: list[str] = []
    if direction != "neutral":
        score += 35
        reasons.append(f"Completed candle closed beyond the 20-bar {direction} boundary")
    prior_atr = float(data.atr.iloc[-6:-1].mean()) if "atr" in data else atr
    if atr > prior_atr:
        score += 15
        reasons.append("ATR is expanding")
    if regime.volume == "rising":
        score += 10
        reasons.append("Volume confirms expansion")
    if regime.volatility == "low":
        score += 10
        reasons.append("Breakout follows volatility compression")
    candle_range = max(float(current.high - current.low), 1e-12)
    if atr and candle_range > atr * 2.5:
        score -= 20
        reasons.append("Breakout candle is extended")
    score += min(10.0, abs(factors.composite) / 8.0)
    return StrategyEvaluation(
        name="volatility_breakout", direction=direction, score=round(max(0, min(100, score)), 2),
        status=_status(score), reasons=reasons, suitable_regimes=["sideways", "transition", "bullish", "bearish"],
    )


def _range_reversion(data, current, atr: float, close: float, regime: MarketRegime, factors: FactorBreakdown) -> StrategyEvaluation:
    prior = data.iloc[-21:-1]
    support = float(prior.low.min())
    resistance = float(prior.high.max())
    near_support = atr > 0 and close <= support + atr * 0.45
    near_resistance = atr > 0 and close >= resistance - atr * 0.45
    rsi = float(current.rsi) if pd.notna(current.rsi) else 50.0
    bullish = near_support and current.close > current.open and rsi <= 42
    bearish = near_resistance and current.close < current.open and rsi >= 58
    direction = "bullish" if bullish else "bearish" if bearish else "neutral"
    score = 20.0
    reasons: list[str] = []
    if regime.trend == "sideways":
        score += 25
        reasons.append("Market regime is sideways")
    else:
        score -= 20
    if direction != "neutral":
        score += 35
        reasons.append("Price rejected a range boundary with RSI displacement")
    if regime.volatility in {"low", "normal"}:
        score += 10
    return StrategyEvaluation(
        name="range_reversion", direction=direction, score=round(max(0, min(100, score)), 2),
        status=_status(score), reasons=reasons, suitable_regimes=["sideways"],
    )
