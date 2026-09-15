from __future__ import annotations

from app.models import MacroAssessment, MacroInputs


def assess_macro(inputs: MacroInputs) -> MacroAssessment:
    """Score relative policy, inflation, and growth without inventing missing data."""
    rate_component = max(-2.0, min(2.0, inputs.policy_rate - inputs.neutral_rate))
    inflation_gap = inputs.inflation - inputs.inflation_target
    inflation_component = max(-1.5, min(1.5, inflation_gap)) * (0.5 if rate_component > 0 else -0.5)
    growth_component = max(-1.5, min(1.5, inputs.growth)) * 0.5
    score = rate_component + inflation_component + growth_component
    bias = "bullish" if score >= 0.75 else "bearish" if score <= -0.75 else "neutral"
    return MacroAssessment(
        bias=bias,
        score=round(score, 3),
        reasons=[
            f"Policy rate is {inputs.policy_rate - inputs.neutral_rate:+.2f} points versus neutral",
            f"Inflation is {inflation_gap:+.2f} points versus target",
            f"Growth input is {inputs.growth:+.2f}%",
        ],
    )
