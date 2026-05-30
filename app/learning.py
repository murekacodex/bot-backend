from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from app.config import get_settings
from app.models import ModelSignal


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class AdaptiveSignalModel:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.path = Path(self.settings.model_state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _default_state(self) -> dict:
        return {
            "bias": 0.0,
            "weights": {},
            "samples_seen": 0,
            "resolved_predictions": 0,
            "correct_predictions": 0,
            "pending_predictions": [],
            "updated_at": None,
        }

    def _load_state(self) -> dict:
        if not self.path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.path.read_text())
        except Exception:
            return self._default_state()
        state = self._default_state()
        state.update(payload if isinstance(payload, dict) else {})
        state["weights"] = dict(state.get("weights") or {})
        state["pending_predictions"] = list(state.get("pending_predictions") or [])
        return state

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.state, indent=2, sort_keys=True))

    def _vector_score(self, features: dict[str, float]) -> float:
        score = float(self.state.get("bias") or 0.0)
        weights = self.state.get("weights") or {}
        for name, value in features.items():
            score += float(weights.get(name, 0.0)) * float(value)
        return score

    def probability(self, features: dict[str, float]) -> float:
        return _sigmoid(self._vector_score(features))

    def summary(self, features: dict[str, float]) -> ModelSignal:
        probability = self.probability(features)
        adjustment = (probability - 0.5) * 2.0
        resolved = int(self.state.get("resolved_predictions") or 0)
        correct = int(self.state.get("correct_predictions") or 0)
        accuracy = (correct / resolved) if resolved else None
        return ModelSignal(
            probability=round(probability, 4),
            adjustment=round(adjustment, 4),
            samples_seen=int(self.state.get("samples_seen") or 0),
            resolved_predictions=resolved,
            accuracy=round(accuracy, 4) if accuracy is not None else None,
            bias=round(float(self.state.get("bias") or 0.0), 4),
        )

    def register_prediction(
        self,
        *,
        market_code: str,
        direction: str,
        entry_price: float,
        features: dict[str, float],
        interval: str,
        period: str,
        timestamp: datetime | None = None,
    ) -> None:
        if not self.settings.enable_learning:
            return

        self.state["pending_predictions"].append(
            {
                "id": str(uuid4()),
                "market_code": market_code,
                "direction": direction,
                "entry_price": float(entry_price),
                "features": {name: float(value) for name, value in features.items()},
                "interval": interval,
                "period": period,
                "predicted_at": (timestamp or datetime.now(timezone.utc)).isoformat(),
                "resolved": False,
            }
        )
        self.state["samples_seen"] = int(self.state.get("samples_seen") or 0) + 1
        self._save()

    def update_from_price(self, market_code: str, current_price: float, now: datetime | None = None) -> int:
        if not self.settings.enable_learning:
            return 0

        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        else:
            current = current.astimezone(timezone.utc)
        horizon = timedelta(hours=self.settings.learning_horizon_hours)
        pending = self.state.get("pending_predictions") or []
        updated = 0
        retained: list[dict] = []

        for item in pending:
            if item.get("resolved"):
                retained.append(item)
                continue

            if item.get("market_code") != market_code:
                retained.append(item)
                continue

            try:
                predicted_at = datetime.fromisoformat(item["predicted_at"])
                if predicted_at.tzinfo is None:
                    predicted_at = predicted_at.replace(tzinfo=timezone.utc)
            except Exception:
                predicted_at = current

            if current - predicted_at < horizon:
                retained.append(item)
                continue

            entry_price = float(item["entry_price"])
            direction = item["direction"]
            move = (float(current_price) - entry_price) / entry_price
            threshold = float(self.settings.learning_min_move_pct)

            if direction == "bullish":
                label = 1 if move >= threshold else 0
            elif direction == "bearish":
                label = 1 if move <= -threshold else 0
            else:
                label = 1 if abs(move) < threshold else 0

            features = {name: float(value) for name, value in item.get("features", {}).items()}
            probability = self.probability(features)
            error = label - probability

            weights = self.state.setdefault("weights", {})
            for name, value in features.items():
                weights[name] = float(weights.get(name, 0.0)) + (self.settings.learning_rate * error * value)
                weights[name] *= 0.999

            self.state["bias"] = float(self.state.get("bias") or 0.0) + (self.settings.learning_rate * error)
            self.state["resolved_predictions"] = int(self.state.get("resolved_predictions") or 0) + 1
            if label == 1:
                self.state["correct_predictions"] = int(self.state.get("correct_predictions") or 0) + 1

            item["resolved"] = True
            item["resolved_at"] = current.isoformat()
            item["outcome_label"] = label
            item["realized_move_pct"] = round(move * 100, 4)
            item["predicted_probability"] = round(probability, 4)
            item["actual_price"] = float(current_price)
            retained.append(item)
            updated += 1

        self.state["pending_predictions"] = retained
        if updated:
            self._save()
        return updated


def aggregate_features(values: Iterable[tuple[str, float]]) -> dict[str, float]:
    return {name: float(value) for name, value in values}
