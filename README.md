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

Run both process types in Heroku:

- `web` serves the API.
- `worker` updates the learner from past predictions.

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
- `FILTER_CLOSED_MARKETS`: skip closed markets from `/markets` and `/signals`.
- `ENABLE_LEARNING`: turn on the adaptive model that updates from prior predictions.
- `ENABLE_SESSION_SUGGESTIONS`: add Asia/London/New York entry timing hints to signals.
- `SESSION_TIMEZONE`: timezone used for session suggestions, defaults to `Africa/Nairobi`.
- `ASIA_SESSION_OPEN` / `ASIA_SESSION_CLOSE`: Asia session window in `SESSION_TIMEZONE`, defaults to `03:00`-`11:00`.
- `LONDON_SESSION_OPEN` / `LONDON_SESSION_CLOSE`: London session window in `SESSION_TIMEZONE`, defaults to `10:00`-`19:00`.
- `NEW_YORK_SESSION_OPEN` / `NEW_YORK_SESSION_CLOSE`: New York session window in `SESSION_TIMEZONE`, defaults to `16:00`-`00:00`.
- `LEARNING_HORIZON_HOURS`: minimum age before a prediction can be scored.
- `LEARNING_MIN_MOVE_PCT`: minimum realized move used to label a past prediction.
- `LEARNING_RATE`: online update rate for the learner.
- `SESSION_ALIGNMENT_BOOST`: score boost when the current session matches the market.
- `SESSION_OFFSESSION_PENALTY`: score penalty when the current session is a poor fit.
- `MODEL_STATE_PATH`: file path used to persist learner state.
- `AUTH_STATE_PATH`: file path used to persist login users.
- `AUTH_SECRET_KEY`: secret used to sign API tokens. Set a strong unique value in production.
- `AUTH_TOKEN_TTL_HOURS`: login token lifetime.

## Access Control

The first successful `/auth/login` request creates the initial admin user when no users exist.
After that, users must log in and send `Authorization: Bearer <token>` to access market data and signals.
Admins can manage users with `/users`, `/users/{user_id}`, and can grant or revoke admin/access status.
