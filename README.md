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
- `EDGE_MIN_SAMPLES`: resolved comparable signals required before historical edge affects scoring.
- `EDGE_MAX_SAMPLES`: maximum recent outcomes included in an edge estimate.
- `EDGE_DECAY`: per-observation weight decay applied from newest to oldest.
- `EDGE_PRIOR_SAMPLES`: zero-expectancy Bayesian prior strength used to resist overfitting.
- `EDGE_MAX_SCORE_ADJUSTMENT`: maximum positive or negative live-score adjustment.
- `TELEGRAM_BOT_TOKEN`: BotFather token used for alerts and the private assistant.
- `TELEGRAM_CHAT_ID`: owner chat ID; only this chat can submit assistant requests.
- `TELEGRAM_ASSISTANT_STATE_PATH`: persisted OpenAI conversation IDs.
- `OPENAI_API_KEY`: OpenAI project API key required to enable Telegram assistant replies.
- `OPENAI_MODEL`: Responses API model, defaults to `gpt-5.4`.
- `OPENAI_BACKGROUND_POLL_SECONDS`: interval for checking background response completion.
- `OPENAI_BACKGROUND_TIMEOUT_SECONDS`: maximum time to wait before reporting a timeout.
- `TELEGRAM_MARKET_UPDATE_HOURS`: minimum interval between no-setup market updates; set to `0` to disable.

## Telegram Trade Alerts

All configured forex and metals markets are eligible. Direction-aware trend,
RSI, candlestick, volatility, and preferred-session checks classify qualifying
setups as `ENTRY READY` or `WATCHLIST`. If neither tier qualifies while markets
are open, a rate-limited market update confirms that the scanner is still running.

## Strategy and Research Framework

Live signals evaluate trend pullback, volatility breakout, and range-reversion
strategies. Each response includes the detected market regime, a weighted
trend/momentum/structure/volatility/session/macro factor breakdown, all strategy
evaluations, and the selected strategy. Risk is reduced for watchlists, high
volatility, and low-volatility conditions.

Research endpoints:

- `GET /research/regime/{code}`: current trend, volatility, and volume regime.
- `GET /research/backtest/{code}`: completed-candle, next-open historical simulation.
- `GET /research/monte-carlo/{code}`: bootstrapped loss and drawdown distribution.
- `GET /research/optimize/{code}`: admin-only grid search with chronological out-of-sample validation.
- `GET /research/attribution`: win rates by strategy, market, direction, and regime.
- `POST /research/macro`: deterministic rates, inflation, and growth assessment.
- `POST /research/portfolio-risk`: aggregate currency exposure and concentration warnings.

Backtests include pessimistic same-candle stop/target ordering and do not use a
forming candle. Results are research estimates, not profitability guarantees.

Historical winner similarity is never a hard market filter. The live engine uses
recent R-multiple expectancy only after the minimum sample count is reached,
shrinks it toward zero, and decays old observations. Negative evidence demotes
an otherwise ready setup to the watchlist. Optimized parameters are marked
promotion-eligible only after at least 30 profitable, positive-Sharpe validation
trades with less than 20% maximum drawdown.

## Telegram Assistant

After the Telegram owner is authorized, send any ordinary text message to start
an assistant request. The bot acknowledges the request immediately and sends the
answer when the background response completes. Only one request per owner chat
runs at a time. Send `/reset` to start a fresh conversation. Trade alerts and
the existing `/start` and `/status` commands continue to work independently.

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
