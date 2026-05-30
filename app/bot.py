import json
import time

from app.analysis import analyze_market
from app.config import get_settings
from app.market_data import fetch_candles
from app.markets import MARKETS
from app.news import fetch_news_sentiment


def run_once() -> list[dict]:
    settings = get_settings()
    signals = []
    for market in MARKETS.values():
        frame = fetch_candles(market, interval=settings.default_interval, period=settings.default_period)
        market_news = fetch_news_sentiment(market) if settings.enable_news_analysis else None
        signal = analyze_market(
            market,
            frame,
            interval=settings.default_interval,
            period=settings.default_period,
            news=market_news,
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
