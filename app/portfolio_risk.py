from __future__ import annotations

from collections import defaultdict


def adjusted_risk_fraction(
    base_risk_percent: float,
    *,
    volatility: str,
    setup_status: str | None,
    current_drawdown: float = 0.0,
) -> float:
    multiplier = 1.0
    if volatility == "high":
        multiplier *= 0.5
    elif volatility == "low":
        multiplier *= 0.8
    if setup_status == "watchlist":
        multiplier *= 0.5
    if current_drawdown <= -0.10:
        multiplier *= 0.25
    elif current_drawdown <= -0.05:
        multiplier *= 0.5
    return round(max(0.0, min(2.0, base_risk_percent * multiplier)), 3)


def currency_exposure(positions: list[dict]) -> dict[str, float]:
    """Aggregate directional notional so correlated FX trades are visible."""
    totals: dict[str, float] = defaultdict(float)
    for position in positions:
        market = str(position.get("market", "")).upper().replace("/", "")
        if len(market) != 6:
            continue
        notional = float(position.get("notional", 0.0))
        direction = 1.0 if position.get("direction") == "bullish" else -1.0
        totals[market[:3]] += notional * direction
        totals[market[3:]] -= notional * direction
    return {currency: round(value, 2) for currency, value in sorted(totals.items())}


def exposure_warnings(positions: list[dict], account_balance: float) -> list[str]:
    if account_balance <= 0:
        return ["Account balance must be positive"]
    warnings: list[str] = []
    for currency, notional in currency_exposure(positions).items():
        multiple = abs(notional) / account_balance
        if multiple > 2:
            warnings.append(f"{currency} exposure is {multiple:.1f}x account equity")
    return warnings
