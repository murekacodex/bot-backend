from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from app.factors import build_factor_breakdown
from app.models import Market
from app.regime import classify_regime
from app.strategy_engine import evaluate_strategies


def _prepared(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    data["ema_9"] = close.ewm(span=9, adjust=False).mean()
    data["ema_21"] = close.ewm(span=21, adjust=False).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - close.shift()).abs(),
            (data["low"] - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = true_range.rolling(14).mean()
    return data


def backtest_frame(
    frame: pd.DataFrame,
    market: Market,
    *,
    minimum_strategy_score: float = 52.0,
    reward_risk: float = 1.5,
    horizon_bars: int = 12,
    risk_fraction: float = 0.01,
) -> dict:
    """Walk completed bars, enter on the next open, and resolve pessimistically."""
    data = _prepared(frame)
    trades: list[dict] = []
    next_eligible_index = 60
    for index in range(60, len(data) - horizon_bars - 1):
        if index < next_eligible_index or pd.isna(data.iloc[index].atr):
            continue
        window = data.iloc[: index + 1]
        current = window.iloc[-1]
        technical = (
            (1 if current.ema_9 > current.ema_21 else -1)
            + (0.75 if current.close > current.sma_50 else -0.75)
            + (0.5 if current.macd > current.macd_signal else -0.5)
        )
        regime = classify_regime(window)
        factors = build_factor_breakdown(
            technical_score=technical,
            pattern_score=0.0,
            rsi=float(current.rsi) if pd.notna(current.rsi) else None,
            regime=regime,
            session_aligned=True,
            news=None,
        )
        evaluations = evaluate_strategies(window, market, regime, factors)
        selected = next(
            (
                item for item in evaluations
                if item.direction != "neutral" and item.score >= minimum_strategy_score
            ),
            None,
        )
        if not selected:
            continue
        entry_bar = data.iloc[index + 1]
        entry = float(entry_bar.open)
        atr = float(current.atr)
        if atr <= 0:
            continue
        stop = entry - atr * 1.5 if selected.direction == "bullish" else entry + atr * 1.5
        target = entry + atr * 1.5 * reward_risk if selected.direction == "bullish" else entry - atr * 1.5 * reward_risk
        outcome_r = None
        exit_price = float(data.iloc[index + horizon_bars].close)
        exit_index = index + horizon_bars
        for cursor in range(index + 1, index + horizon_bars + 1):
            candle = data.iloc[cursor]
            stop_hit = float(candle.low) <= stop if selected.direction == "bullish" else float(candle.high) >= stop
            target_hit = float(candle.high) >= target if selected.direction == "bullish" else float(candle.low) <= target
            if stop_hit:
                outcome_r, exit_price, exit_index = -1.0, stop, cursor
                break
            if target_hit:
                outcome_r, exit_price, exit_index = reward_risk, target, cursor
                break
        if outcome_r is None:
            move = (exit_price - entry) if selected.direction == "bullish" else (entry - exit_price)
            outcome_r = max(-1.0, min(reward_risk, move / max(abs(entry - stop), 1e-12)))
        trades.append(
            {
                "strategy": selected.name,
                "direction": selected.direction,
                "signal_time": data.index[index].isoformat(),
                "exit_time": data.index[exit_index].isoformat(),
                "entry": round(entry, 6),
                "exit": round(exit_price, 6),
                "r_multiple": round(float(outcome_r), 4),
                "regime": regime.trend,
                "strategy_score": selected.score,
            }
        )
        next_eligible_index = exit_index + 1
    return {"metrics": performance_metrics(trades, risk_fraction=risk_fraction), "trades": trades}


def performance_metrics(trades: list[dict], *, risk_fraction: float = 0.01) -> dict:
    if not trades:
        return {
            "trades": 0, "win_rate": None, "average_r": None, "profit_factor": None,
            "total_return": 0.0, "cagr": None, "sharpe": None, "max_drawdown": 0.0,
            "average_recovery_trades": None,
        }
    returns = np.array([float(item["r_multiple"]) * risk_fraction for item in trades], dtype=float)
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    recovery_lengths: list[int] = []
    underwater_start = None
    for index, drawdown in enumerate(drawdowns):
        if drawdown < 0 and underwater_start is None:
            underwater_start = index
        elif drawdown >= 0 and underwater_start is not None:
            recovery_lengths.append(index - underwater_start)
            underwater_start = None
    start = pd.Timestamp(trades[0]["signal_time"])
    end = pd.Timestamp(trades[-1]["exit_time"])
    years = max((end - start).total_seconds() / (365.25 * 86400), 1 / 365.25)
    total_return = float(equity[-1] - 1)
    cagr = float(equity[-1] ** (1 / years) - 1)
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=1) > 0 else None
    return {
        "trades": len(trades),
        "win_rate": round(float((returns > 0).mean()), 4),
        "average_r": round(float(np.mean([item["r_multiple"] for item in trades])), 4),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 4) if len(losses) and losses.sum() else None,
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "max_drawdown": round(float(drawdowns.min()), 4),
        "average_recovery_trades": round(float(np.mean(recovery_lengths)), 2) if recovery_lengths else None,
    }


def monte_carlo(trades: list[dict], *, simulations: int = 1000, risk_fraction: float = 0.01, seed: int = 7) -> dict:
    if not trades:
        return {"simulations": simulations, "probability_of_loss": None, "return_percentiles": {}, "worst_drawdown": None}
    rng = np.random.default_rng(seed)
    samples = np.array([float(item["r_multiple"]) * risk_fraction for item in trades])
    ending_returns: list[float] = []
    worst_drawdowns: list[float] = []
    for _ in range(simulations):
        path = rng.choice(samples, size=len(samples), replace=True)
        equity = np.cumprod(1 + path)
        drawdown = equity / np.maximum.accumulate(equity) - 1
        ending_returns.append(float(equity[-1] - 1))
        worst_drawdowns.append(float(drawdown.min()))
    return {
        "simulations": simulations,
        "probability_of_loss": round(float(np.mean(np.array(ending_returns) < 0)), 4),
        "return_percentiles": {
            "p05": round(float(np.percentile(ending_returns, 5)), 4),
            "p50": round(float(np.percentile(ending_returns, 50)), 4),
            "p95": round(float(np.percentile(ending_returns, 95)), 4),
        },
        "worst_drawdown": round(float(min(worst_drawdowns)), 4),
        "drawdown_p95": round(float(np.percentile(worst_drawdowns, 5)), 4),
    }


def optimize_strategy(frame: pd.DataFrame, market: Market) -> dict:
    """Tune on the first 70% and report the winner on untouched later data."""
    split = max(80, int(len(frame) * 0.7))
    if len(frame) - split < 30:
        return {"best": None, "validation": None, "candidates": [], "warning": "Not enough history for chronological validation"}
    training = frame.iloc[:split]
    validation_start = max(0, split - 60)
    validation = frame.iloc[validation_start:]
    validation_boundary = pd.Timestamp(frame.index[split])
    candidates: list[dict] = []
    for minimum_score in (52.0, 60.0, 70.0):
        for reward_risk in (1.25, 1.5, 2.0):
            result = backtest_frame(
                training, market, minimum_strategy_score=minimum_score, reward_risk=reward_risk
            )
            metrics = result["metrics"]
            candidates.append(
                {"minimum_score": minimum_score, "reward_risk": reward_risk, "metrics": metrics}
            )
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["metrics"]["sharpe"] if item["metrics"]["sharpe"] is not None else -999,
            item["metrics"]["total_return"],
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    validation_result = None
    if best:
        validation_result = backtest_frame(
            validation,
            market,
            minimum_strategy_score=best["minimum_score"],
            reward_risk=best["reward_risk"],
        )
        validation_trades = [
            trade for trade in validation_result["trades"]
            if pd.Timestamp(trade["signal_time"]) >= validation_boundary
        ]
        validation_result = {
            "metrics": performance_metrics(validation_trades),
            "trades": validation_trades[-250:],
        }
    validation_metrics = validation_result["metrics"] if validation_result else {}
    promotion_eligible = bool(
        validation_metrics
        and validation_metrics.get("trades", 0) >= 30
        and (validation_metrics.get("sharpe") or 0) > 0
        and validation_metrics.get("total_return", 0) > 0
        and validation_metrics.get("max_drawdown", -1) > -0.20
    )
    return {
        "training_fraction": 0.7,
        "best": best,
        "validation": validation_result,
        "promotion_eligible": promotion_eligible,
        "promotion_requirements": {
            "minimum_validation_trades": 30,
            "positive_sharpe": True,
            "positive_total_return": True,
            "maximum_drawdown": 0.20,
        },
        "candidates": ranked,
    }


def attribute_outcomes(entries: list[dict]) -> dict:
    groups: dict[str, list[bool]] = defaultdict(list)
    for entry in entries:
        outcome = entry.get("outcome") or {}
        if outcome.get("status") != "resolved" or outcome.get("success") is None:
            continue
        strategy = entry.get("strategy") or "unknown"
        market = entry.get("market_code") or "unknown"
        direction = entry.get("direction") or "unknown"
        regime = entry.get("market_regime") or "unknown"
        groups[f"strategy:{strategy}"].append(bool(outcome["success"]))
        groups[f"market:{market}"].append(bool(outcome["success"]))
        groups[f"direction:{direction}"].append(bool(outcome["success"]))
        groups[f"regime:{regime}"].append(bool(outcome["success"]))
    return {
        key: {"trades": len(values), "win_rate": round(sum(values) / len(values), 4)}
        for key, values in sorted(groups.items())
    }
