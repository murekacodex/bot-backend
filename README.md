# Forex Signal Bot Backend

FastAPI service that fetches forex and gold candles, analyzes indicators and candlestick patterns, and exposes signal endpoints for a React UI.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`.

## Heroku

This folder contains the Heroku files:

- `Procfile`
- `requirements.txt`
- `.python-version`

Deploy from inside `backend/` or use `git subtree push --prefix backend heroku main`.

Set CORS for your deployed React app:

```bash
heroku config:set CORS_ORIGINS=https://your-react-app-domain.com
```

## Configuration

Environment variables:

- `CORS_ORIGINS`: comma-separated list of allowed frontend origins.
- `DEFAULT_INTERVAL`: default candle interval, for example `15m`, `1h`, `1d`.
- `DEFAULT_PERIOD`: default yfinance period, for example `5d`, `1mo`.
- `CACHE_TTL_SECONDS`: how long fetched candles stay cached.
- `BOT_POLL_SECONDS`: worker polling interval.
- `ENABLE_NEWS_ANALYSIS`: include Yahoo Finance news sentiment in signal decisions.
- `NEWS_CACHE_TTL_SECONDS`: how long news sentiment stays cached.
- `NEWS_LOOKBACK_HOURS`: maximum article age used for sentiment.
