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

## AWS deployment

The repository root contains a Docker Compose deployment for a dedicated AWS
Lightsail instance. It runs the API, learner worker, and frontend together. The
`bot-data` volume keeps authentication, model, and journal state across container
restarts, and state writes use inter-process locks.

Copy `.env.production.example` to `.env.production`, generate a unique
`AUTH_SECRET_KEY`, set the public origin, then run:

```console
docker compose up -d --build
```

Production startup fails if the default authentication secret is still present.

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
- `ENVIRONMENT`: set to `production` in deployed environments.
- `ACCOUNT_CURRENCY`: account currency used for safe lot-size conversion, currently `USD`.
- `AUTH_TOKEN_TTL_HOURS`: login token lifetime.
- `RISK_ACCOUNT_BALANCE`: account balance used for suggested lot-size calculations.
- `RISK_PERCENT`: account percentage to risk per signal.
- `SIGNAL_LOG_PATH`: file path used to persist generated signal history and outcomes.
- `SIGNAL_OUTCOME_HORIZON_HOURS`: minimum age before a logged signal can be scored.
- `SIGNAL_OUTCOME_MIN_MOVE_PCT`: minimum realized move used to decide if a signal succeeded.

## Access Control

The first successful `/auth/login` request creates the initial admin user when no users exist.
After that, users must log in and send `Authorization: Bearer <token>` to access market data and signals.
Only the initial admin account can be an admin. Admins can manage normal users with `/users` and `/users/{user_id}`, including enabling or disabling access.

## Signal Journal

Generated API and worker signals are persisted to `SIGNAL_LOG_PATH`.
Repeated refreshes are de-duplicated by source, market, interval, period, and candle timestamp.

Useful endpoints:

- `GET /signal-log`: latest logged signals, including pending and resolved outcomes.
- `GET /signal-log/stats`: aggregate outcome accuracy by market.
- `POST /signal-log/resolve`: admin-only endpoint to resolve eligible pending outcomes immediately.
