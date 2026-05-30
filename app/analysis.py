import math
from datetime import timezone

import numpy as np
import pandas as pd

from app.models import Candle, Market, NewsSentiment, RiskPlan, Signal


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift()).abs()
    low_close = (frame["low"] - frame["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


def _detect_patterns(frame: pd.DataFrame) -> tuple[list[str], float]:
    if len(frame) < 3:
        return [], 0

    previous = frame.iloc[-2]
    current = frame.iloc[-1]
    before = frame.iloc[-3]

    body = abs(current.close - current.open)
    candle_range = max(current.high - current.low, 1e-12)
    upper_wick = current.high - max(current.open, current.close)
    lower_wick = min(current.open, current.close) - current.low
    patterns: list[str] = []
    score = 0.0

    current_bullish = current.close > current.open
    previous_bearish = previous.close < previous.open
    current_bearish = current.close < current.open
    previous_bullish = previous.close > previous.open

    if current_bullish and previous_bearish and current.close > previous.open and current.open < previous.close:
        patterns.append("Bullish engulfing candle")
        score += 1.5

    if current_bearish and previous_bullish and current.open > previous.close and current.close < previous.open:
        patterns.append("Bearish engulfing candle")
        score -= 1.5

    if lower_wick > body * 2 and upper_wick < body and body / candle_range < 0.45:
        patterns.append("Hammer-style rejection from lows")
        score += 1.0

    if upper_wick > body * 2 and lower_wick < body and body / candle_range < 0.45:
        patterns.append("Shooting-star rejection from highs")
        score -= 1.0

    if body / candle_range < 0.1:
        patterns.append("Doji indecision candle")

    if before.close < before.open and previous.close < previous.open and current_bullish and current.close > previous.open:
        patterns.append("Three-candle bullish reversal attempt")
        score += 1.0

    if before.close > before.open and previous.close > previous.open and current_bearish and current.close < previous.open:
        patterns.append("Three-candle bearish reversal attempt")
        score -= 1.0

    return patterns, score


def _finite(value: float | int | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _news_score(news: NewsSentiment | None) -> float:
    if not news or news.articles_analyzed == 0:
        return 0.0
    return max(-1.5, min(1.5, news.score * 0.75))


def analyze_market(market: Market, frame: pd.DataFrame, interval: str, period: str, news: NewsSentiment | None = None) -> Signal:
    if len(frame) < 30:
        raise ValueError("At least 30 candles are needed for signal analysis")

    data = frame.copy()
    data["ema_9"] = data["close"].ewm(span=9, adjust=False).mean()
    data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()
    data["sma_50"] = data["close"].rolling(50).mean()
    data["rsi"] = _rsi(data["close"])
    data["atr"] = _atr(data)
    data["macd"] = data["close"].ewm(span=12, adjust=False).mean() - data["close"].ewm(span=26, adjust=False).mean()
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    latest = data.iloc[-1]
    previous = data.iloc[-2]
    patterns, pattern_score = _detect_patterns(data)

    reasons: list[str] = []
    warnings: list[str] = []
    score = pattern_score

    if latest.ema_9 > latest.ema_21:
        score += 1.0
        reasons.append("EMA 9 is above EMA 21, showing short-term bullish momentum")
    else:
        score -= 1.0
        reasons.append("EMA 9 is below EMA 21, showing short-term bearish momentum")

    if pd.notna(latest.sma_50):
        if latest.close > latest.sma_50:
            score += 0.75
            reasons.append("Price is trading above SMA 50")
        else:
            score -= 0.75
            reasons.append("Price is trading below SMA 50")

    if latest.macd > latest.macd_signal and previous.macd <= previous.macd_signal:
        score += 1.25
        reasons.append("MACD crossed bullish")
    elif latest.macd < latest.macd_signal and previous.macd >= previous.macd_signal:
        score -= 1.25
        reasons.append("MACD crossed bearish")
    elif latest.macd > latest.macd_signal:
        score += 0.5
        reasons.append("MACD remains above signal")
    else:
        score -= 0.5
        reasons.append("MACD remains below signal")

    if pd.notna(latest.rsi):
        if latest.rsi < 30:
            score += 0.75
            reasons.append("RSI is oversold")
        elif latest.rsi > 70:
            score -= 0.75
            reasons.append("RSI is overbought")
        elif 45 <= latest.rsi <= 60:
            reasons.append("RSI is balanced")

    if patterns:
        reasons.extend(patterns)

    news_adjustment = _news_score(news)
    if news_adjustment:
        score += news_adjustment
        reasons.append(f"Market news sentiment is {news.sentiment} with a {news_adjustment:+.2f} score adjustment")
    elif news and news.error:
        warnings.append(news.error)
    elif news and news.articles_analyzed == 0:
        warnings.append("No recent Yahoo Finance news was available for this market")

    if pd.isna(latest.atr) or latest.atr == 0:
        warnings.append("ATR is unavailable, so risk levels are omitted")

    if abs(score) < 1.25:
        direction = "neutral"
        strategy = "Wait for confirmation"
    elif score > 0:
        direction = "bullish"
        strategy = "Trend-following long setup with candlestick confirmation"
    else:
        direction = "bearish"
        strategy = "Trend-following short setup with candlestick confirmation"

    confidence = min(95, max(5, int(50 + abs(score) * 10)))
    if direction == "neutral":
        confidence = min(confidence, 55)

    risk = None
    atr = _finite(latest.atr)
    close = float(latest.close)
    if atr and direction != "neutral":
        stop_distance = atr * 1.5
        target_distance = atr * 2.25
        if direction == "bullish":
            stop_loss = close - stop_distance
            take_profit_1 = close + target_distance
            take_profit_2 = close + target_distance * 1.6
        else:
            stop_loss = close + stop_distance
            take_profit_1 = close - target_distance
            take_profit_2 = close - target_distance * 1.6
        risk = RiskPlan(
            entry=round(close, 5),
            stop_loss=round(stop_loss, 5),
            take_profit_1=round(take_profit_1, 5),
            take_profit_2=round(take_profit_2, 5),
            risk_reward=1.5,
        )

    timestamp = data.index[-1]
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)

    return Signal(
        market=market,
        interval=interval,
        period=period,
        timestamp=timestamp.isoformat(),
        direction=direction,
        confidence=confidence,
        score=round(float(score), 2),
        strategy=strategy,
        reasons=reasons,
        warnings=warnings,
        indicators={
            "close": round(close, 5),
            "ema_9": round(float(latest.ema_9), 5),
            "ema_21": round(float(latest.ema_21), 5),
            "sma_50": round(float(latest.sma_50), 5) if pd.notna(latest.sma_50) else None,
            "rsi": round(float(latest.rsi), 2) if pd.notna(latest.rsi) else None,
            "atr": round(float(latest.atr), 5) if pd.notna(latest.atr) else None,
            "macd": round(float(latest.macd), 5),
            "macd_signal": round(float(latest.macd_signal), 5),
            "news_score": news.score if news else None,
        },
        news=news,
        risk=risk,
        last_candle=Candle(
            time=timestamp.isoformat(),
            open=float(latest.open),
            high=float(latest.high),
            low=float(latest.low),
            close=close,
            volume=float(latest.volume) if "volume" in latest and pd.notna(latest.volume) else None,
        ),
    )
