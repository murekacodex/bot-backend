import math
from datetime import timezone

import numpy as np
import pandas as pd

from app.learning import AdaptiveSignalModel, aggregate_features
from app.config import get_settings
from app.models import Candle, Market, NewsSentiment, PatternHint, RiskPlan, Signal, TimeframeContext
from app.market_data import interval_duration
from app.factors import build_factor_breakdown
from app.portfolio_risk import adjusted_risk_fraction
from app.regime import classify_regime
from app.session import recommend_session_entry
from app.signal_journal import calibrated_confidence, rolling_performance_edge
from app.strategy_engine import evaluate_strategies


FOCUS_MARKETS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "EURGBP", "XAUUSD", "XAGUSD", "XPTUSD",
}


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


def _detect_patterns(frame: pd.DataFrame) -> tuple[list[PatternHint], float]:
    if len(frame) < 3:
        return [], 0

    previous = frame.iloc[-2]
    current = frame.iloc[-1]
    before = frame.iloc[-3]

    body = abs(current.close - current.open)
    candle_range = max(current.high - current.low, 1e-12)
    upper_wick = current.high - max(current.open, current.close)
    lower_wick = min(current.open, current.close) - current.low
    patterns: list[PatternHint] = []
    score = 0.0
    recent = frame.iloc[-20:-1]
    atr = float(current.atr) if "atr" in current and pd.notna(current.atr) else candle_range
    near_support = current.low <= float(recent.low.min()) + (atr * 0.25)
    near_resistance = current.high >= float(recent.high.max()) - (atr * 0.25)
    prior_down = previous.close < before.close
    prior_up = previous.close > before.close

    current_bullish = current.close > current.open
    previous_bearish = previous.close < previous.open
    current_bearish = current.close < current.open
    previous_bullish = previous.close > previous.open

    if current_bullish and previous_bearish and current.close > previous.open and current.open < previous.close:
        patterns.append(PatternHint(name="Bullish engulfing", bias="bullish", strength=82, meaning="Buyers fully covered the previous bearish body.", confirmation="Prefer a close above this candle's high.", candle_offset=0))
        if prior_down and near_support:
            score += 1.5

    if current_bearish and previous_bullish and current.open > previous.close and current.close < previous.open:
        patterns.append(PatternHint(name="Bearish engulfing", bias="bearish", strength=82, meaning="Sellers fully covered the previous bullish body.", confirmation="Prefer a close below this candle's low.", candle_offset=0))
        if prior_up and near_resistance:
            score -= 1.5

    if lower_wick > body * 2 and upper_wick < body and body / candle_range < 0.45:
        patterns.append(PatternHint(name="Hammer", bias="bullish", strength=68, meaning="Lower prices were rejected before the candle closed.", confirmation="Wait for the next candle to break the hammer high.", candle_offset=0))
        if prior_down and near_support:
            score += 1.0

    if upper_wick > body * 2 and lower_wick < body and body / candle_range < 0.45:
        patterns.append(PatternHint(name="Shooting star", bias="bearish", strength=68, meaning="Higher prices were rejected before the candle closed.", confirmation="Wait for the next candle to break the star low.", candle_offset=0))
        if prior_up and near_resistance:
            score -= 1.0

    if body / candle_range < 0.1:
        patterns.append(PatternHint(name="Doji", bias="neutral", strength=45, meaning="Buyers and sellers finished near balance.", confirmation="Do not predict direction until price breaks the doji range.", candle_offset=0))

    if before.close < before.open and previous.close < previous.open and current_bullish and current.close > previous.open:
        patterns.append(PatternHint(name="Three-candle bullish reversal", bias="bullish", strength=72, meaning="Selling pressure weakened and buyers reclaimed the prior body.", confirmation="Confirm with follow-through above the formation high.", candle_offset=0))
        if near_support:
            score += 1.0

    if before.close > before.open and previous.close > previous.open and current_bearish and current.close < previous.open:
        patterns.append(PatternHint(name="Three-candle bearish reversal", bias="bearish", strength=72, meaning="Buying pressure weakened and sellers reclaimed the prior body.", confirmation="Confirm with follow-through below the formation low.", candle_offset=0))
        if near_resistance:
            score -= 1.0

    return patterns, score


def _finite(value: float | int | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _capped_stop_loss(direction: str, entry: float, raw_stop: float, max_distance: float) -> float:
    if direction == "bullish":
        return max(raw_stop, entry - max_distance)
    return min(raw_stop, entry + max_distance)


def _take_profit_levels(
    direction: str,
    entry: float,
    stop_distance: float,
    tp1_r: float,
    tp2_r: float,
    strategy: str | None,
    recent_low: float,
    recent_high: float,
) -> tuple[float, float]:
    sign = 1 if direction == "bullish" else -1
    default_tp1 = entry + sign * stop_distance * tp1_r
    default_tp2 = entry + sign * stop_distance * tp2_r
    if strategy != "range_reversion":
        return default_tp1, default_tp2

    midpoint = (recent_low + recent_high) / 2
    if direction == "bullish":
        tp1 = min(default_tp1, midpoint) if midpoint > entry else default_tp1
        tp2 = min(default_tp2, recent_high) if recent_high > tp1 else default_tp2
    else:
        tp1 = max(default_tp1, midpoint) if midpoint < entry else default_tp1
        tp2 = max(default_tp2, recent_low) if recent_low < tp1 else default_tp2
    return tp1, tp2


def _apply_session_quality(score: float, alignment: str, adjustment: float) -> float:
    """Change conviction without ever introducing a bullish/bearish bias."""
    if score == 0:
        return score
    magnitude = abs(adjustment)
    if alignment == "aligned":
        return score + math.copysign(magnitude, score)
    return score - math.copysign(min(magnitude, abs(score)), score)


def _news_score(news: NewsSentiment | None, market: Market) -> float:
    if not news or news.articles_analyzed == 0:
        return 0.0
    adjustment = max(-1.5, min(1.5, news.score * 0.75))
    if market.category == "forex" and market.code.startswith("USD"):
        adjustment *= -1
    return adjustment


def _contract_units(market: Market) -> float:
    if market.code == "XAUUSD":
        return 100.0
    if market.code == "XAGUSD":
        return 5000.0
    if market.code == "XPTUSD":
        return 50.0
    return 100_000.0


def _suggested_lot_size(market: Market, entry: float, stop_loss: float, risk_amount: float) -> float:
    price_risk = abs(entry - stop_loss)
    if price_risk <= 0 or risk_amount <= 0:
        return 0.0

    units = _contract_units(market)
    quote_currency = market.code[-3:]
    if market.category == "forex" and quote_currency != "USD":
        if market.code.startswith("USD"):
            quote_conversion = entry
        else:
            return 0.0
    else:
        quote_conversion = 1.0
    per_lot_risk = (price_risk * units) / max(quote_conversion, 1e-12)
    if per_lot_risk <= 0:
        return 0.0
    lot_size = risk_amount / per_lot_risk
    return round(lot_size, 2) if lot_size >= 0.01 else 0.0


def _prepared_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ema_9"] = data["close"].ewm(span=9, adjust=False).mean()
    data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()
    data["sma_50"] = data["close"].rolling(50).mean()
    data["rsi"] = _rsi(data["close"])
    data["atr"] = _atr(data)
    data["macd"] = data["close"].ewm(span=12, adjust=False).mean() - data["close"].ewm(span=26, adjust=False).mean()
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    return data


def summarize_timeframe(frame: pd.DataFrame, interval: str, period: str) -> TimeframeContext:
    if len(frame) < 30:
        raise ValueError("At least 30 candles are needed for timeframe context")

    data = _prepared_indicators(frame)
    latest = data.iloc[-1]
    close = float(latest.close)
    ema_9 = float(latest.ema_9)
    ema_21 = float(latest.ema_21)
    macd_delta = float(latest.macd - latest.macd_signal)
    score = 0.0

    if ema_9 > ema_21:
        score += 1.0
    else:
        score -= 1.0

    if pd.notna(latest.sma_50):
        score += 0.5 if latest.close > latest.sma_50 else -0.5

    if macd_delta > 0:
        score += 0.5
    else:
        score -= 0.5

    if pd.notna(latest.rsi):
        if latest.rsi < 30:
            score += 0.35
        elif latest.rsi > 70:
            score -= 0.35

    if abs(score) < 0.75:
        direction = "neutral"
    elif score > 0:
        direction = "bullish"
    else:
        direction = "bearish"

    summary = f"{interval} context is {direction}: EMA 9 {'above' if ema_9 > ema_21 else 'below'} EMA 21"
    return TimeframeContext(
        interval=interval,
        period=period,
        direction=direction,
        score=round(score, 2),
        close=round(close, 5),
        ema_9=round(ema_9, 5),
        ema_21=round(ema_21, 5),
        rsi=round(float(latest.rsi), 2) if pd.notna(latest.rsi) else None,
        macd_delta=round(macd_delta, 5),
        summary=summary,
    )


def _build_feature_map(latest: pd.Series, news: NewsSentiment | None, pattern_score: float, score: float, market: Market, interval: str) -> dict[str, float]:
    close = float(latest.close)
    ema_9 = float(latest.ema_9)
    ema_21 = float(latest.ema_21)
    sma_50 = float(latest.sma_50) if pd.notna(latest.sma_50) else close
    rsi = float(latest.rsi) if pd.notna(latest.rsi) else 50.0
    atr = float(latest.atr) if pd.notna(latest.atr) else 0.0
    macd = float(latest.macd)
    macd_signal = float(latest.macd_signal)
    news_score = float(news.score) if news else 0.0
    news_confidence = float(news.confidence) / 100.0 if news else 0.0

    return aggregate_features(
        [
            ("ema_gap", (ema_9 - ema_21) / max(close, 1e-12)),
            ("price_vs_sma50", (close - sma_50) / max(close, 1e-12)),
            ("rsi_centered", (50.0 - rsi) / 50.0),
            ("atr_ratio", atr / max(close, 1e-12)),
            ("macd_delta", (macd - macd_signal) / max(abs(close), 1e-12)),
            ("news_score", news_score / 3.0),
            ("news_confidence", news_confidence),
            ("pattern_score", pattern_score / 4.0),
            ("technical_score", score / 6.0),
            (f"market:{market.code}", 1.0),
            (f"interval:{interval}", 1.0),
        ]
    )


def _directional_pattern_score(signal: Signal) -> bool:
    pattern_score = float((signal.features or {}).get("pattern_score", 0.0))
    return pattern_score > 0 if signal.direction == "bullish" else pattern_score < 0


def _directional_rsi(signal: Signal) -> bool:
    rsi = signal.indicators.get("rsi")
    if rsi is None:
        return False
    value = float(rsi)
    return 40 <= value <= 70 if signal.direction == "bullish" else 30 <= value <= 60


def trade_candidate_tier(signal: Signal) -> str | None:
    """Classify a safe, direction-aware setup for alerts and the dashboard."""
    if signal.market.code not in FOCUS_MARKETS or signal.direction == "neutral" or not signal.risk:
        return None
    features = signal.features or {}
    max_atr_ratio = 0.015 if signal.market.category == "metal" else 0.002
    if float(features.get("atr_ratio", float("inf"))) > max_atr_ratio:
        return None
    if signal.session is None or signal.session.alignment != "aligned":
        return None
    if not set(signal.session.active_sessions).intersection(signal.session.preferred_sessions):
        return None
    selected = next(
        (
            evaluation for evaluation in (getattr(signal, "strategy_evaluations", None) or [])
            if evaluation.name == getattr(signal, "selected_strategy", None) and evaluation.direction == signal.direction
        ),
        None,
    )
    if selected:
        if selected.status == "entry_ready":
            if (
                getattr(signal, "historical_edge", None)
                and signal.historical_edge.sufficient_evidence
                and signal.historical_edge.expectancy_r <= 0
            ):
                return "watchlist"
            return "entry_ready"
        if selected.status == "watchlist":
            return "watchlist"
        return None

    # Backward-compatible gate for persisted/legacy signals without strategy metadata.
    if abs(signal.score) < 2.0:
        return None
    if not _directional_pattern_score(signal):
        return None
    directional_patterns = [pattern for pattern in signal.patterns if pattern.bias == signal.direction]
    if not directional_patterns or max(pattern.strength for pattern in directional_patterns) < 60:
        return None
    if not _directional_rsi(signal):
        return None
    close = signal.indicators.get("close")
    ema_9 = signal.indicators.get("ema_9")
    ema_21 = signal.indicators.get("ema_21")
    sma_50 = signal.indicators.get("sma_50")
    if None in (close, ema_9, ema_21, sma_50):
        return None
    trend_aligned = (
        signal.direction == "bullish" and float(close) > float(sma_50) and float(ema_9) > float(ema_21)
    ) or (
        signal.direction == "bearish" and float(close) < float(sma_50) and float(ema_9) < float(ema_21)
    )
    if not trend_aligned:
        return None
    strongest_pattern = max(pattern.strength for pattern in directional_patterns)
    if abs(signal.score) >= 3.0 and strongest_pattern >= 65:
        return "entry_ready"
    return "watchlist"


def is_focused_trade_candidate(signal: Signal) -> bool:
    return trade_candidate_tier(signal) is not None


def analyze_market(
    market: Market,
    frame: pd.DataFrame,
    interval: str,
    period: str,
    news: NewsSentiment | None = None,
    learner: AdaptiveSignalModel | None = None,
    timeframes: dict[str, TimeframeContext] | None = None,
) -> Signal:
    settings = get_settings()
    if len(frame) < 30:
        raise ValueError("At least 30 candles are needed for signal analysis")

    data = _prepared_indicators(frame)

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
        reasons.extend(pattern.name for pattern in patterns)

    timeframe_context = timeframes or {}
    higher_context = timeframe_context.get("higher")
    lower_context = timeframe_context.get("lower")
    if higher_context:
        if higher_context.direction == "bullish":
            score += 0.85
        elif higher_context.direction == "bearish":
            score -= 0.85
        reasons.append(f"Higher timeframe {higher_context.interval}: {higher_context.direction} bias")
    if lower_context:
        if lower_context.direction == "bullish":
            score += 0.45
        elif lower_context.direction == "bearish":
            score -= 0.45
        reasons.append(f"Lower timeframe {lower_context.interval}: {lower_context.direction} timing")

    news_adjustment = _news_score(news, market)
    if news_adjustment:
        score += news_adjustment
        reasons.append(f"Market news sentiment is {news.sentiment} with a {news_adjustment:+.2f} score adjustment")
    elif news and news.error:
        warnings.append(news.error)
    elif news and news.articles_analyzed == 0:
        warnings.append("No recent Yahoo Finance news was available for this market")

    if pd.isna(latest.atr) or latest.atr == 0:
        warnings.append("ATR is unavailable, so risk levels are omitted")

    regime = classify_regime(data)
    session_signal = recommend_session_entry(market)
    factors = build_factor_breakdown(
        technical_score=score,
        pattern_score=pattern_score,
        rsi=float(latest.rsi) if pd.notna(latest.rsi) else None,
        regime=regime,
        session_aligned=session_signal.alignment == "aligned",
        news=news,
    )
    strategy_evaluations = evaluate_strategies(data, market, regime, factors)
    base_direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    selected_evaluation = next(
        (
            evaluation for evaluation in strategy_evaluations
            if evaluation.status != "inactive"
            and evaluation.direction != "neutral"
            and (base_direction == "neutral" or evaluation.direction == base_direction)
        ),
        None,
    )
    if selected_evaluation:
        strategy_adjustment = min(1.5, selected_evaluation.score / 50.0)
        score += strategy_adjustment if selected_evaluation.direction == "bullish" else -strategy_adjustment
        reasons.append(
            f"{selected_evaluation.name.replace('_', ' ').title()} qualified at {selected_evaluation.score:.0f}/100"
        )
    edge_direction = selected_evaluation.direction if selected_evaluation else base_direction
    historical_edge = rolling_performance_edge(
        market.code,
        edge_direction,
        strategy=selected_evaluation.name if selected_evaluation else None,
        regime=regime.trend,
    )
    if historical_edge.sufficient_evidence and edge_direction != "neutral":
        direction_sign = 1.0 if edge_direction == "bullish" else -1.0
        score += direction_sign * historical_edge.score_adjustment
        reasons.append(
            f"Recent expectancy is {historical_edge.expectancy_r:+.2f}R across "
            f"{historical_edge.samples} comparable resolved signals"
        )

    feature_map = _build_feature_map(latest, news, pattern_score, score, market, interval)
    model_signal = learner.summary(feature_map) if learner else None
    if model_signal:
        model_adjustment = model_signal.adjustment
        score += model_adjustment
        reasons.append(f"Adaptive model adjusted the score by {model_adjustment:+.2f}")
    else:
        model_adjustment = 0.0

    if settings.enable_session_suggestions:
        score = _apply_session_quality(score, session_signal.alignment, session_signal.score_adjustment)
        if session_signal.alignment == "aligned":
            reasons.append(session_signal.suggestion)
        else:
            warnings.append(session_signal.suggestion)

    if abs(score) < 1.25:
        direction = "neutral"
        strategy = "Wait for confirmation"
    elif selected_evaluation and score > 0:
        direction = "bullish"
        strategy = selected_evaluation.name.replace("_", " ").title()
    elif selected_evaluation and score < 0:
        direction = "bearish"
        strategy = selected_evaluation.name.replace("_", " ").title()
    elif score > 0:
        direction = "bullish"
        strategy = "Trend-following long setup with candlestick confirmation"
    else:
        direction = "bearish"
        strategy = "Trend-following short setup with candlestick confirmation"

    raw_confidence = min(85, max(5, int(50 + abs(score) * 8)))
    confidence = calibrated_confidence(market.code, interval, direction, raw_confidence)
    if direction == "neutral":
        confidence = min(confidence, 55)

    risk = None
    atr = _finite(latest.atr)
    close = float(latest.close)
    if atr and direction != "neutral":
        execution_bps = settings.execution_cost_bps_metal if market.category == "metal" else settings.execution_cost_bps_forex
        execution_cost = close * max(execution_bps, 0.0) / 10_000.0
        recent = data.iloc[-20:]
        selected_name = selected_evaluation.name if selected_evaluation else None
        max_stop_distance = atr * max(settings.max_stop_atr, 1.0)
        tp1_r = max(settings.take_profit_1_r, 0.1)
        tp2_r = max(settings.take_profit_2_r, tp1_r)
        recent_low = float(recent.low.min())
        recent_high = float(recent.high.max())
        if direction == "bullish":
            entry = close + execution_cost
            if selected_name == "volatility_breakout":
                raw_stop = min(entry - atr, float(recent.iloc[:-1].high.max()) - atr * 0.25)
            else:
                raw_stop = min(entry - (atr * 1.5), recent_low - (atr * 0.1))
            stop_loss = _capped_stop_loss(direction, entry, raw_stop, max_stop_distance)
            stop_distance = entry - stop_loss
        else:
            entry = close - execution_cost
            if selected_name == "volatility_breakout":
                raw_stop = max(entry + atr, float(recent.iloc[:-1].low.min()) + atr * 0.25)
            else:
                raw_stop = max(entry + (atr * 1.5), recent_high + (atr * 0.1))
            stop_loss = _capped_stop_loss(direction, entry, raw_stop, max_stop_distance)
            stop_distance = stop_loss - entry
        take_profit_1, take_profit_2 = _take_profit_levels(
            direction, entry, stop_distance, tp1_r, tp2_r,
            selected_name, recent_low, recent_high,
        )
        risk_percent = adjusted_risk_fraction(
            max(settings.risk_percent, 0.0),
            volatility=regime.volatility,
            setup_status=selected_evaluation.status if selected_evaluation else None,
        )
        risk_amount = max(settings.risk_account_balance, 0.0) * risk_percent / 100.0
        risk = RiskPlan(
            entry=round(entry, 5),
            stop_loss=round(stop_loss, 5),
            take_profit_1=round(take_profit_1, 5),
            take_profit_2=round(take_profit_2, 5),
            risk_reward=round(abs(take_profit_1 - entry) / max(stop_distance, 1e-12), 2),
            risk_percent=round(risk_percent, 2),
            risk_amount=round(risk_amount, 2),
            suggested_lot_size=_suggested_lot_size(market, entry, stop_loss, risk_amount),
        )
        warnings.append(f"Entry includes an estimated {execution_bps:.1f} bps execution cost; verify live broker pricing")
        if risk.suggested_lot_size == 0 and market.category == "forex" and market.code[-3:] != settings.account_currency:
            warnings.append(
                f"Lot size omitted because {market.code[-3:]} to {settings.account_currency} conversion is unavailable"
            )

    candle_timestamp = data.index[-1]
    if candle_timestamp.tzinfo is None:
        candle_timestamp = candle_timestamp.tz_localize(timezone.utc)
    timestamp = candle_timestamp + interval_duration(interval)

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
            "model_adjustment": round(model_adjustment, 4),
            "higher_timeframe": higher_context.direction if higher_context else None,
            "lower_timeframe": lower_context.direction if lower_context else None,
        },
        timeframes=timeframe_context,
        features=feature_map,
        news=news,
        model=model_signal,
        patterns=patterns,
        regime=regime,
        factors=factors,
        historical_edge=historical_edge,
        strategy_evaluations=strategy_evaluations,
        selected_strategy=selected_evaluation.name if selected_evaluation else None,
        session=session_signal,
        risk=risk,
        last_candle=Candle(
            time=candle_timestamp.isoformat(),
            open=float(latest.open),
            high=float(latest.high),
            low=float(latest.low),
            close=close,
            volume=float(latest.volume) if "volume" in latest and pd.notna(latest.volume) else None,
        ),
    )
