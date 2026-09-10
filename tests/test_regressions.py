import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException

import pandas as pd

from app.analysis import _detect_patterns, _suggested_lot_size
from app.learning import AdaptiveSignalModel
from app.main import validate_timeframe
from app.models import Market
from app.session import market_is_open


class TimeframeValidationTests(unittest.TestCase):
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
    def test_globex_daily_maintenance_break_is_closed(self):
        market = Market(code="XAUUSD", symbol="GC=F", name="Gold", category="metal", session="cme_globex")
        # 17:30 New York during standard time.
        instant = datetime(2026, 1, 6, 22, 30, tzinfo=timezone.utc)
        self.assertFalse(market_is_open(market, instant)[0])


if __name__ == "__main__":
    unittest.main()
