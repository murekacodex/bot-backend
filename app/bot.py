import json
import time
from datetime import timezone

from app.analysis import analyze_market
from app.config import get_settings
from app.learning import AdaptiveSignalModel
from app.market_data import fetch_candles
from app.markets import MARKETS
from app.news import fetch_news_sentiment
from app.session import attach_market_status


learner = AdaptiveSignalModel()


def run_once() -> list[dict]:
    settings = get_settings()
    signals = []
    for market in [attach_market_status(market) for market in MARKETS.values()]:
        if settings.filter_closed_markets and not market.is_open:
            continue
        frame = fetch_candles(market, interval=settings.default_interval, period=settings.default_period)
        current_close = float(frame.iloc[-1]["close"])
        current_time = frame.index[-1].to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)
        market_news = fetch_news_sentiment(market) if settings.enable_news_analysis else None
        learner.update_from_price(market.code, current_close, now=current_time)
        signal = analyze_market(
            market,
            frame,
            interval=settings.default_interval,
            period=settings.default_period,
            news=market_news,
            learner=learner,
        )
        learner.register_prediction(
            market_code=market.code,
            direction=signal.direction,
            entry_price=signal.last_candle.close,
            features=signal.features or {},
            interval=settings.default_interval,
            period=settings.default_period,
            timestamp=current_time,
        )
        signals.append(signal.model_dump())
    return signals


def main() -> None:
    settings = get_settings()
    while True:
        try:
            print(json.dumps(run_once(), default=str), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
        time.sleep(settings.bot_poll_seconds)


if __name__ == "__main__":
    main()
