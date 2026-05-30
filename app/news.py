from datetime import datetime, timedelta, timezone
from typing import Any

import yfinance as yf

from app.config import get_settings
from app.models import Market, NewsItem, NewsSentiment

_cache: dict[str, tuple[datetime, NewsSentiment]] = {}

BULLISH_TERMS = {
    "beat",
    "beats",
    "bullish",
    "climb",
    "climbs",
    "cut",
    "cuts",
    "demand",
    "eases",
    "gain",
    "gains",
    "growth",
    "higher",
    "optimism",
    "rally",
    "rebound",
    "recovery",
    "rise",
    "rises",
    "safe haven",
    "soft landing",
    "surge",
    "upside",
    "weaker dollar",
}

BEARISH_TERMS = {
    "bearish",
    "concern",
    "concerns",
    "decline",
    "declines",
    "drop",
    "drops",
    "fear",
    "hawkish",
    "higher dollar",
    "inflation",
    "loss",
    "losses",
    "pressure",
    "recession",
    "risk-off",
    "slump",
    "strong dollar",
    "tumble",
    "uncertainty",
    "weak",
}


def _article_content(article: dict[str, Any]) -> tuple[str, str, str | None, str | None, datetime | None]:
    content = article.get("content") if isinstance(article.get("content"), dict) else {}
    title = str(article.get("title") or content.get("title") or "").strip()
    publisher = article.get("publisher") or content.get("provider", {}).get("displayName")
    link = article.get("link") or content.get("canonicalUrl", {}).get("url")
    published_raw = article.get("providerPublishTime") or content.get("pubDate")

    published_at = None
    if isinstance(published_raw, int | float):
        published_at = datetime.fromtimestamp(published_raw, tz=timezone.utc)
    elif isinstance(published_raw, str):
        try:
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except ValueError:
            published_at = None

    summary = article.get("summary") or content.get("summary") or content.get("description") or ""
    text = f"{title} {summary}".lower()
    return text, title, publisher, link, published_at


def _empty_sentiment(error: str | None = None) -> NewsSentiment:
    return NewsSentiment(
        sentiment="neutral",
        score=0,
        confidence=0,
        articles_analyzed=0,
        bullish_terms=[],
        bearish_terms=[],
        headlines=[],
        error=error,
    )


def _score_text(text: str) -> tuple[float, list[str], list[str]]:
    found_bullish = sorted(term for term in BULLISH_TERMS if term in text)
    found_bearish = sorted(term for term in BEARISH_TERMS if term in text)
    score = sum(1 for _ in found_bullish) - sum(1 for _ in found_bearish)
    return float(score), found_bullish, found_bearish


def fetch_news_sentiment(market: Market) -> NewsSentiment:
    settings = get_settings()
    key = market.code
    now = datetime.now(timezone.utc)
    cached = _cache.get(key)
    if cached and now - cached[0] < timedelta(seconds=settings.news_cache_ttl_seconds):
        return cached[1]

    cutoff = now - timedelta(hours=settings.news_lookback_hours)
    try:
        raw_articles = yf.Ticker(market.symbol).news or []
    except Exception as exc:
        result = _empty_sentiment(f"Yahoo Finance news unavailable: {exc}")
        _cache[key] = (now, result)
        return result
    headlines: list[NewsItem] = []
    all_bullish_terms: set[str] = set()
    all_bearish_terms: set[str] = set()
    total_score = 0.0

    for article in raw_articles[:12]:
        text, title, publisher, link, published_at = _article_content(article)
        if not title:
            continue
        if published_at and published_at < cutoff:
            continue

        article_score, bullish_terms, bearish_terms = _score_text(text)
        all_bullish_terms.update(bullish_terms)
        all_bearish_terms.update(bearish_terms)
        total_score += article_score

        if article_score > 0:
            sentiment = "bullish"
        elif article_score < 0:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        headlines.append(
            NewsItem(
                title=title,
                publisher=publisher,
                link=link,
                published_at=published_at.isoformat() if published_at else None,
                sentiment=sentiment,
                score=article_score,
            )
        )

    if not headlines:
        result = _empty_sentiment()
    else:
        normalized = max(-3.0, min(3.0, total_score / max(1, len(headlines))))
        if normalized > 0.25:
            sentiment = "bullish"
        elif normalized < -0.25:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        result = NewsSentiment(
            sentiment=sentiment,
            score=round(normalized, 2),
            confidence=min(90, int(abs(normalized) * 30) + min(30, len(headlines) * 4)),
            articles_analyzed=len(headlines),
            bullish_terms=sorted(all_bullish_terms)[:8],
            bearish_terms=sorted(all_bearish_terms)[:8],
            headlines=headlines[:5],
        )

    _cache[key] = (now, result)
    return result
