# Signal Bot Operational Diagnosis

## Intended flow

1. The worker fetches completed 4-hour and daily candles for each open market.
2. The API returns every current market analysis to the dashboard.
3. A setup meeting the watchlist rules sends a Telegram **WATCHLIST** message.
4. The watchlist remains active while the bot monitors confirmation.
5. After live-price validation, the bot sends an **ENTRY READY** message.
6. Alerts remain visible until they expire or are invalidated.

## Verified production status

- Heroku has separate web and worker dynos.
- The worker successfully analyzes 22 market/timeframe combinations per scan.
- EUR/USD live market data is available through Yahoo Finance.
- Telegram token and owner-chat delivery are valid.
- The worker uses completed candles, so a new 4-hour or daily strategy result is only possible when that candle closes.

## Remaining operational issues

### Health-message deletion

`remove_outdated_alerts()` deletes an existing no-setup market update on the next scan. `send_market_update()` then observes the six-hour rate limit and does not replace it. The result is a health message that disappears after roughly one scan.

### Direction is missing from active-alert scope

Active alerts are keyed by market and timeframe only. A bearish setup can therefore inherit a bullish watchlist for the same market/timeframe. Direction must be included when deciding whether a watchlist may be promoted to entry ready.

### Misleading worker-health endpoint

`/health/worker` checks in-memory state in the web dyno. The real scanner is a separate worker dyno, so the endpoint reports disabled even while the worker is healthy. A durable worker heartbeat should be stored in Postgres and exposed by the endpoint.

### Live-price dependency

`ENTRY READY` requires a current live quote. MetaAPI is not configured, so the bot uses Yahoo's one-minute fallback. A stale quote, excessive spread, price deviation, or opposite one-minute momentum correctly blocks entry-ready delivery.

### Strict alert gates

Watchlist and entry-ready messages require high confidence, confirmed structure displacement, session alignment, matching higher/lower timeframes, acceptable volatility, and a qualifying strategy. A successful scan with no Telegram trade alert can therefore be normal.

### Limited runtime observability

Worker logs show generated/error counts but not the count of neutral, watchlist, entry-ready, live-price-blocked, or notification-delivery outcomes. These metrics should be logged and persisted for straightforward production diagnosis.

### Test isolation

Some older tests assume file-backed state. Production uses Postgres, so those tests can read shared state instead of their temporary fixtures. Test configuration should explicitly disable `DATABASE_URL` or use an isolated database.

## Current polling policy

The worker defaults to starting scans every 60 seconds and caches market data for 45 seconds. The production environment should set `BOT_POLL_SECONDS=60` and `CACHE_TTL_SECONDS=45` explicitly as well. Because each scan itself takes time and only completed 4-hour/daily candles drive strategy changes, this improves responsiveness without treating an in-progress swing candle as a valid signal. Most scans will therefore see unchanged 4-hour/daily analysis; their value is live-price confirmation, alert lifecycle maintenance, and operational visibility.
