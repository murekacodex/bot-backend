from __future__ import annotations

import math

from app.models import FactorBreakdown, MarketRegime, NewsSentiment


FACTOR_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.20,
    "structure": 0.20,
    "volatility": 0.15,
    "session": 0.10,
    "macro": 0.05,
}


def build_factor_breakdown(
    *,
    technical_score: float,
    pattern_score: float,
    rsi: float | None,
    regime: MarketRegime,
    session_aligned: bool,
    news: NewsSentiment | None,
) -> FactorBreakdown:
    direction = 1.0 if technical_score > 0 else -1.0 if technical_score < 0 else 0.0
    trend = direction * min(100.0, regime.trend_strength)
    if rsi is None:
        momentum = 0.0
    else:
        momentum = max(-100.0, min(100.0, (float(rsi) - 50.0) * 4.0))
    structure = max(-100.0, min(100.0, pattern_score * 50.0))
    volatility = {"low": 25.0, "normal": 75.0, "high": -40.0}.get(regime.volatility, 0.0) * direction
    session = (70.0 if session_aligned else -30.0) * direction
    macro = 0.0
    if news and news.articles_analyzed:
        macro = max(-100.0, min(100.0, float(news.score) * 33.33))
    values = {
        "trend": trend,
        "momentum": momentum,
        "structure": structure,
        "volatility": volatility,
        "session": session,
        "macro": macro,
    }
    composite = sum(values[name] * FACTOR_WEIGHTS[name] for name in FACTOR_WEIGHTS)
    if not math.isfinite(composite):
        composite = 0.0
    return FactorBreakdown(
        values={name: round(value, 2) for name, value in values.items()},
        weights=FACTOR_WEIGHTS,
        composite=round(composite, 2),
    )
