import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import pandas as pd

from app.analysis import (
    FOCUS_MARKETS,
    _apply_session_quality,
    _capped_stop_loss,
    _detect_patterns,
    _directional_pattern_score,
    _directional_rsi,
    _suggested_lot_size,
    _take_profit_levels,
)
from app.config import ANALYSIS_TIMEFRAMES
from app.learning import AdaptiveSignalModel
from app.main import validate_timeframe
from app.market_data import completed_candles
from app.macro import assess_macro
from app.markets import get_market
from app.models import MacroInputs, Market, RiskPlan, SessionSignal, Signal
from app.portfolio_risk import adjusted_risk_fraction, currency_exposure
from app.regime import classify_regime
from app.research import monte_carlo, performance_metrics
from app.session import market_is_open
from app.signal_journal import _resolve_entry, signal_outcome_stats
from app.signal_journal import rolling_performance_edge
from app.telegram_alerts import _alert_key, _copy_keyboard, _message, _password_matches, remove_outdated_alerts
from app.telegram_assistant import _completed_text


class TimeframeValidationTests(unittest.TestCase):
    def test_background_scan_covers_every_dashboard_timeframe(self):
        self.assertEqual(set(ANALYSIS_TIMEFRAMES), {"1m", "15m", "30m", "1h", "4h", "1d"})

    def test_allows_one_minute_history_supported_by_provider(self):
        validate_timeframe("1m", "5d")

    def test_rejects_too_short_daily_period(self):
        with self.assertRaises(HTTPException) as raised:
            validate_timeframe("1d", "5d")
        self.assertEqual(raised.exception.status_code, 422)

    def test_allows_resampled_four_hour_period(self):
        validate_timeframe("4h", "1mo")


class RiskSizingTests(unittest.TestCase):
    def test_cross_currency_pair_does_not_return_unsafe_lot_size(self):
        market = Market(code="EURGBP", symbol="EURGBP=X", name="EUR/GBP", category="forex")
        self.assertEqual(_suggested_lot_size(market, 0.85, 0.84, 10), 0.0)

    def test_minimum_lot_does_not_exceed_risk_budget(self):
        market = Market(code="EURUSD", symbol="EURUSD=X", name="EUR/USD", category="forex")
        self.assertEqual(_suggested_lot_size(market, 1.1, 1.0, 1), 0.0)

    def test_stop_distance_is_capped_at_two_atr(self):
        self.assertEqual(_capped_stop_loss("bullish", 100, 90, 2), 98)
        self.assertEqual(_capped_stop_loss("bearish", 100, 110, 2), 102)

    def test_standard_targets_use_one_and_one_point_five_r(self):
        self.assertEqual(
            _take_profit_levels("bullish", 100, 2, 1, 1.5, None, 95, 110),
            (102, 103),
        )

    def test_range_reversion_prefers_closer_midpoint(self):
        self.assertEqual(
            _take_profit_levels("bullish", 100, 4, 1, 1.5, "range_reversion", 96, 106),
            (101, 106),
        )


class ClosedCandleTests(unittest.TestCase):
    def test_excludes_a_still_forming_final_bar(self):
        frame = pd.DataFrame(
            {"close": [1.0, 1.1]},
            index=pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-01T10:01:00Z"]),
        )
        result = completed_candles(frame, "1m", datetime(2026, 1, 1, 10, 1, 30, tzinfo=timezone.utc))
        self.assertEqual(len(result), 1)


class SessionScoringTests(unittest.TestCase):
    def test_session_quality_never_changes_direction(self):
        self.assertGreater(_apply_session_quality(0.1, "off_session", -0.2), -0.0001)
        self.assertLess(_apply_session_quality(-0.1, "aligned", 0.4), 0)


class FocusedCandidateTests(unittest.TestCase):
    def test_all_configured_markets_are_eligible(self):
        self.assertIn("EURUSD", FOCUS_MARKETS)
        self.assertIn("XAUUSD", FOCUS_MARKETS)

    def test_pattern_score_is_direction_aware(self):
        bullish = SimpleNamespace(direction="bullish", features={"pattern_score": 0.25})
        bearish = SimpleNamespace(direction="bearish", features={"pattern_score": -0.25})
        self.assertTrue(_directional_pattern_score(bullish))
        self.assertTrue(_directional_pattern_score(bearish))

    def test_rsi_range_is_direction_aware(self):
        bullish = SimpleNamespace(direction="bullish", indicators={"rsi": 65})
        bearish = SimpleNamespace(direction="bearish", indicators={"rsi": 35})
        self.assertTrue(_directional_rsi(bullish))
        self.assertTrue(_directional_rsi(bearish))


class TelegramAlertTests(unittest.TestCase):
    def test_password_hash_check(self):
        import hashlib
        salt = "00112233445566778899aabbccddeeff"
        digest = hashlib.pbkdf2_hmac("sha256", b"nchabirichoi", bytes.fromhex(salt), 1000).hex()
        encoded = f"pbkdf2_sha256$1000${salt}${digest}"
        self.assertTrue(_password_matches("nchabirichoi", encoded))
        self.assertFalse(_password_matches("wrong", encoded))

    def test_alert_contains_trade_plan_and_has_stable_candle_key(self):
        signal = Signal.model_construct(
            market=Market(code="USDCAD", symbol="CAD=X", name="USD/CAD", category="forex"),
            interval="1h", direction="bullish", timestamp="2026-01-01T11:00:00+00:00",
            confidence=68, score=3.4,
            risk=RiskPlan(entry=1.35, stop_loss=1.34, take_profit_1=1.365, take_profit_2=1.374, risk_reward=1.5, risk_percent=1, risk_amount=10, suggested_lot_size=.01),
            session=SessionSignal(current_session="new_york", active_sessions=["new_york"], preferred_sessions=["new_york"], alignment="aligned", suggestion="Entry", score_adjustment=.4, confidence=80, reasons=[]),
        )
        self.assertIn("🟢 USDCAD · BULLISH · 1H", _message(signal))
        self.assertIn("🛑  STOP LOSS   1.34", _message(signal))
        self.assertIn("🎯  TAKE PROFIT 1   1.365", _message(signal))
        self.assertEqual(_alert_key(signal), "entry_ready:USDCAD:1h:bullish:2026-01-01T11:00:00+00:00")
        keyboard = json.loads(_copy_keyboard(signal))
        self.assertEqual(keyboard["inline_keyboard"][0][1]["copy_text"]["text"], "1.34")
        self.assertEqual(keyboard["inline_keyboard"][1][0]["copy_text"]["text"], "1.365")
        self.assertEqual(keyboard["inline_keyboard"][1][1]["copy_text"]["text"], "1.374")

    def test_extracts_assistant_output_text(self):
        response = {
            "output": [
                {"type": "reasoning"},
                {"type": "message", "content": [{"type": "output_text", "text": "Finished safely."}]},
            ]
        }
        self.assertEqual(_completed_text(response), "Finished safely.")

    def test_removes_alert_after_entry_window_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "telegram_alerts.json"
            state_path.write_text(
                '{"sent": ["old"], "active_alerts": {"EURUSD:1h": '
                '{"key": "old", "messages": [{"chat_id": "7", "message_id": 42}]}}}'
            )
            settings = SimpleNamespace(telegram_bot_token="token", telegram_alert_state_path=str(state_path))
            with patch("app.telegram_alerts.get_settings", return_value=settings), patch("app.telegram_alerts._api_call") as api_call:
                removed = remove_outdated_alerts(set())
            self.assertEqual(removed, 1)
            api_call.assert_called_once_with("deleteMessage", {"chat_id": "7", "message_id": 42})
            self.assertEqual(json.loads(state_path.read_text())["active_alerts"], {})


class OutcomePathTests(unittest.TestCase):
    def test_take_profit_hit_resolves_before_time_horizon(self):
        signal_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        frame = pd.DataFrame(
            [{"open": 1.0, "high": 1.03, "low": 0.995, "close": 1.02}],
            index=pd.to_datetime(["2026-01-01T10:15:00Z"]),
        )
        entry = {
            "market_code": "EURUSD", "interval": "15m", "period": "5d",
            "candle_time": signal_time.isoformat(), "generated_at": signal_time.isoformat(),
            "direction": "bullish", "entry_price": 1.0,
            "risk": {"stop_loss": 0.99, "take_profit_1": 1.02},
            "outcome": {"status": "pending"},
        }
        with patch("app.signal_journal.fetch_candles", return_value=frame):
            self.assertTrue(_resolve_entry(entry, signal_time + timedelta(minutes=30)))
        self.assertTrue(entry["outcome"]["success"])
        self.assertEqual(entry["outcome"]["label"], "take_profit_1")

    def test_stats_identify_best_market_resolved_in_last_24_hours(self):
        now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
        entries = [
            {"market_code": "EURUSD", "generated_at": now.isoformat(), "interval": "1h", "period": "5d", "candle_time": "a", "direction": "bullish", "outcome": {"status": "resolved", "resolved_at": (now - timedelta(hours=2)).isoformat(), "success": True}},
            {"market_code": "GBPUSD", "generated_at": now.isoformat(), "interval": "1h", "period": "5d", "candle_time": "b", "direction": "bullish", "outcome": {"status": "resolved", "resolved_at": (now - timedelta(hours=2)).isoformat(), "success": True}},
            {"market_code": "GBPUSD", "generated_at": now.isoformat(), "interval": "1h", "period": "5d", "candle_time": "c", "direction": "bullish", "outcome": {"status": "resolved", "resolved_at": (now - timedelta(hours=3)).isoformat(), "success": True}},
            {"market_code": "USDJPY", "generated_at": now.isoformat(), "interval": "1h", "period": "5d", "candle_time": "d", "direction": "bullish", "outcome": {"status": "resolved", "resolved_at": (now - timedelta(hours=25)).isoformat(), "success": True}},
        ]
        with patch("app.signal_journal._read_state", return_value={"signals": entries}):
            stats = signal_outcome_stats(now=now)
        self.assertEqual(stats.best_market_24h["code"], "GBPUSD")
        self.assertEqual(stats.best_market_24h["resolved"], 2)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model = AdaptiveSignalModel()
        self.model.path = Path(self.temp_dir.name) / "model.json"
        self.model.state = self.model._default_state()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_same_candle_is_registered_once(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        arguments = dict(
            market_code="EURUSD",
            direction="bullish",
            entry_price=1.1,
            features={"technical_score": 0.5},
            interval="1h",
            period="5d",
            timestamp=timestamp,
        )
        self.model.register_prediction(**arguments)
        self.model.register_prediction(**arguments)
        self.assertEqual(self.model.state["samples_seen"], 1)
        self.assertEqual(len(self.model.state["pending_predictions"]), 1)

    def test_successful_bearish_move_teaches_downward_direction(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.model.register_prediction(
            market_code="EURUSD",
            direction="bearish",
            entry_price=1.0,
            features={"technical_score": 1.0},
            interval="1h",
            period="5d",
            timestamp=timestamp,
        )
        self.model.update_from_price("EURUSD", 0.98, now=timestamp + timedelta(hours=25))
        self.assertLess(self.model.state["weights"]["technical_score"], 0)
        # An untrained model starts at 50% (classified upward), so its own
        # prediction was wrong even though the rule-based direction was right.
        self.assertEqual(self.model.state["correct_predictions"], 0)
        self.assertEqual(self.model.state["squared_error_sum"], 0.25)
        self.assertEqual(self.model.state["successful_signals_learned"], 1)

    def test_successful_bullish_signal_strengthens_upward_weight(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.model.register_prediction(
            market_code="EURUSD", direction="bullish", entry_price=1.0,
            features={"technical_score": 1.0}, interval="1h", period="5d", timestamp=timestamp,
        )
        self.model.update_from_price("EURUSD", 1.02, now=timestamp + timedelta(hours=25))
        self.assertGreater(self.model.state["weights"]["technical_score"], 0)
        self.assertEqual(self.model.state["successful_signals_learned"], 1)

    def test_failed_bullish_signal_teaches_downward_direction(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.model.register_prediction(
            market_code="EURUSD", direction="bullish", entry_price=1.0,
            features={"technical_score": 1.0}, interval="1h", period="5d", timestamp=timestamp,
        )
        self.model.update_from_price("EURUSD", 0.98, now=timestamp + timedelta(hours=25))
        self.assertLess(self.model.state["weights"]["technical_score"], 0)
        self.assertEqual(self.model.state["unsuccessful_signals_learned"], 1)

    def test_failed_bearish_signal_teaches_upward_direction(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.model.register_prediction(
            market_code="EURUSD", direction="bearish", entry_price=1.0,
            features={"technical_score": 1.0}, interval="1h", period="5d", timestamp=timestamp,
        )
        self.model.update_from_price("EURUSD", 1.02, now=timestamp + timedelta(hours=25))
        self.assertGreater(self.model.state["weights"]["technical_score"], 0)
        self.assertEqual(self.model.state["unsuccessful_signals_learned"], 1)

    def test_flat_unsuccessful_signal_is_counted_and_reduces_confidence(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.model.state["bias"] = 1.0
        self.model._save()
        self.model.register_prediction(
            market_code="EURUSD", direction="bullish", entry_price=1.0,
            features={"technical_score": 1.0}, interval="1h", period="5d", timestamp=timestamp,
        )
        self.model.update_from_price("EURUSD", 1.0001, now=timestamp + timedelta(hours=25))
        self.assertLess(self.model.state["bias"], 1.0)
        self.assertEqual(self.model.state["resolved_predictions"], 1)
        self.assertEqual(self.model.state["unsuccessful_signals_learned"], 1)

    def test_model_does_not_adjust_signals_before_warmup(self):
        summary = self.model.summary({"technical_score": 1.0})
        self.assertFalse(summary.learning_ready)
        self.assertEqual(summary.adjustment, 0.0)


class PatternTests(unittest.TestCase):
    def test_bullish_engulfing_returns_explainable_hint(self):
        frame = pd.DataFrame([
            {"open": 1.01, "high": 1.02, "low": 0.99, "close": 1.00},
            {"open": 1.00, "high": 1.01, "low": 0.97, "close": 0.98},
            {"open": 0.97, "high": 1.02, "low": 0.96, "close": 1.01},
        ])
        patterns, score = _detect_patterns(frame)
        engulfing = next(pattern for pattern in patterns if pattern.name == "Bullish engulfing")
        self.assertEqual(engulfing.bias, "bullish")
        self.assertIn("close above", engulfing.confirmation)
        self.assertGreater(score, 0)


class MarketHoursTests(unittest.TestCase):
    def test_usdjpy_prefers_asia_and_new_york_for_entry_alerts(self):
        self.assertEqual(get_market("USDJPY").preferred_sessions, ["asia", "new_york"])

    def test_globex_daily_maintenance_break_is_closed(self):
        market = Market(code="XAUUSD", symbol="GC=F", name="Gold", category="metal", session="cme_globex")
        # 17:30 New York during standard time.
        instant = datetime(2026, 1, 6, 22, 30, tzinfo=timezone.utc)
        self.assertFalse(market_is_open(market, instant)[0])


class StrategyFrameworkTests(unittest.TestCase):
    def test_regime_detects_rising_market(self):
        index = pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC")
        close = pd.Series([1 + item * 0.001 for item in range(80)], index=index)
        frame = pd.DataFrame(
            {"open": close - 0.0002, "high": close + 0.0005, "low": close - 0.0005, "close": close}
        )
        self.assertEqual(classify_regime(frame).trend, "bullish")

    def test_high_volatility_and_watchlist_reduce_risk(self):
        self.assertEqual(
            adjusted_risk_fraction(1.0, volatility="high", setup_status="watchlist"),
            0.25,
        )

    def test_currency_exposure_nets_base_and_quote(self):
        result = currency_exposure(
            [{"market": "EURUSD", "direction": "bullish", "notional": 1000}]
        )
        self.assertEqual(result, {"EUR": 1000.0, "USD": -1000.0})

    def test_macro_assessment_uses_rates_inflation_and_growth(self):
        result = assess_macro(
            MacroInputs(policy_rate=5, neutral_rate=3, inflation=3, inflation_target=2, growth=2)
        )
        self.assertEqual(result.bias, "bullish")

    def test_metrics_and_monte_carlo_are_deterministic(self):
        trades = [
            {"r_multiple": 1.5, "signal_time": "2025-01-01T00:00:00Z", "exit_time": "2025-01-02T00:00:00Z"},
            {"r_multiple": -1.0, "signal_time": "2025-01-03T00:00:00Z", "exit_time": "2025-01-04T00:00:00Z"},
            {"r_multiple": 1.5, "signal_time": "2025-01-05T00:00:00Z", "exit_time": "2025-01-06T00:00:00Z"},
        ]
        self.assertEqual(performance_metrics(trades)["trades"], 3)
        first = monte_carlo(trades, simulations=100, seed=11)
        second = monte_carlo(trades, simulations=100, seed=11)
        self.assertEqual(first, second)

    def test_recent_edge_requires_enough_samples_and_is_shrunk(self):
        entries = []
        for index in range(35):
            entries.append(
                {
                    "market_code": "EURUSD",
                    "interval": "1h",
                    "period": "5d",
                    "candle_time": f"2026-01-01T{index:02d}:00:00Z",
                    "generated_at": f"2026-02-01T{index:02d}:00:00Z",
                    "direction": "bullish",
                    "entry_price": 1.0,
                    "risk": {"stop_loss": 0.99, "risk_reward": 1.5},
                    "outcome": {"status": "resolved", "success": True, "label": "take_profit_1"},
                }
            )
        with patch("app.signal_journal._read_state", return_value={"signals": entries}):
            edge = rolling_performance_edge("EURUSD", "bullish")
        self.assertTrue(edge.sufficient_evidence)
        self.assertGreater(edge.expectancy_r, 0)
        self.assertLess(edge.expectancy_r, 1.5)
        self.assertGreater(edge.score_adjustment, 0)


if __name__ == "__main__":
    unittest.main()
