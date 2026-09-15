import threading
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.analysis import FOCUS_MARKETS, analyze_market, trade_candidate_tier, summarize_timeframe
from app.auth import admin_user, create_user, current_user, delete_user, list_users, login_or_create_admin, update_user, users_exist
from app.config import get_settings
from app.learning import AdaptiveSignalModel
from app.market_data import dataframe_to_candles, fetch_candles
from app.macro import assess_macro
from app.markets import MARKETS, get_market
from app.models import AuthResponse, Candle, CreateUserRequest, LoginRequest, MacroAssessment, MacroInputs, Market, MarketRegime, NewsSentiment, Signal, SignalLogEntry, SignalOutcomeStats, UpdateUserRequest, UserPublic
from app.news import fetch_news_sentiment
from app.portfolio_risk import currency_exposure, exposure_warnings
from app.regime import classify_regime
from app.research import attribute_outcomes, backtest_frame, monte_carlo, optimize_strategy
from app.session import attach_market_status
from app.signal_journal import list_signal_log, record_signal, record_signals, resolve_signal_outcomes, signal_outcome_stats
from app.telegram_alerts import send_viable_entry_alert, start_telegram_poller

settings = get_settings()
settings.validate_security()
learner = AdaptiveSignalModel()

app = FastAPI(
    title="Forex Signal Bot",
    description="Forex and gold candlestick signal API for bullish, bearish, and neutral trade ideas.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def start_integrations() -> None:
    start_telegram_poller()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/setup")
def auth_setup() -> dict[str, bool]:
    return {"needs_admin": not users_exist()}


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    token, user, setup_admin = login_or_create_admin(payload)
    return AuthResponse(token=token, user=user, setup_admin=setup_admin)


@app.get("/auth/me", response_model=UserPublic)
def me(user: UserPublic = Depends(current_user)) -> UserPublic:
    return user


@app.get("/users", response_model=list[UserPublic])
def users(_: UserPublic = Depends(admin_user)) -> list[UserPublic]:
    return list_users()


@app.post("/users", response_model=UserPublic)
def add_user(payload: CreateUserRequest, _: UserPublic = Depends(admin_user)) -> UserPublic:
    return create_user(payload)


@app.patch("/users/{user_id}", response_model=UserPublic)
def edit_user(user_id: str, payload: UpdateUserRequest, user: UserPublic = Depends(admin_user)) -> UserPublic:
    return update_user(user_id, payload, acting_user=user)


@app.delete("/users/{user_id}", status_code=204)
def remove_user(user_id: str, user: UserPublic = Depends(admin_user)) -> Response:
    delete_user(user_id, acting_user=user)
    return Response(status_code=204)


@app.get("/markets", response_model=list[Market])
def markets(include_closed: bool = Query(default=False), _: UserPublic = Depends(current_user)) -> list[Market]:
    markets = [attach_market_status(market) for market in MARKETS.values()]
    if settings.filter_closed_markets and not include_closed:
        return [market for market in markets if market.is_open]
    return markets


@app.get("/candles/{code}", response_model=list[Candle])
def candles(
    code: str,
    interval: str = Query(default=settings.default_interval),
    period: str = Query(default=settings.default_period),
    _: UserPublic = Depends(current_user),
) -> list[Candle]:
    validate_timeframe(interval, period)
    try:
        market = get_market(code)
        frame = fetch_candles(market, interval=interval, period=period)
        return dataframe_to_candles(frame.tail(200))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/news/{code}", response_model=NewsSentiment)
def news(code: str, _: UserPublic = Depends(current_user)) -> NewsSentiment:
    try:
        market = get_market(code)
        return fetch_news_sentiment(market)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


TIMEFRAME_MAP = {
    "1m": {"lower": None, "higher": ("5m", "1d")},
    "5m": {"lower": None, "higher": ("15m", "1d")},
    "15m": {"lower": ("5m", "1d"), "higher": ("1h", "5d")},
    "30m": {"lower": ("15m", "1d"), "higher": ("1h", "5d")},
    "1h": {"lower": ("15m", "1d"), "higher": ("1d", "3mo")},
    "4h": {"lower": ("1h", "5d"), "higher": ("1d", "3mo")},
    "1d": {"lower": ("1h", "5d"), "higher": ("1wk", "1y")},
}

SUPPORTED_TIMEFRAMES = {
    "1m": {"1d", "5d"},
    "15m": {"1d", "5d", "1mo"},
    "30m": {"1d", "5d", "1mo"},
    "1h": {"5d", "1mo", "3mo"},
    "4h": {"1mo", "3mo"},
    "1d": {"3mo"},
}


def validate_timeframe(interval: str, period: str) -> None:
    if period not in SUPPORTED_TIMEFRAMES.get(interval, set()):
        supported = ", ".join(sorted(SUPPORTED_TIMEFRAMES.get(interval, set()))) or "none"
        raise HTTPException(status_code=422, detail=f"Unsupported timeframe. {interval} supports: {supported}")


def timeframe_contexts(market: Market, interval: str) -> tuple[dict, list[str]]:
    contexts = {}
    warnings = []
    selected = TIMEFRAME_MAP.get(interval, TIMEFRAME_MAP["1h"])
    for label in ("higher", "lower"):
        selection = selected.get(label)
        if not selection:
            continue
        selected_interval, selected_period = selection
        try:
            frame = fetch_candles(market, interval=selected_interval, period=selected_period)
            contexts[label] = summarize_timeframe(frame, interval=selected_interval, period=selected_period)
        except Exception as exc:
            warnings.append(f"{label.title()} timeframe {selected_interval} unavailable: {exc}")
    return contexts, warnings


@app.get("/signals", response_model=list[Signal])
def signals(
    interval: str = Query(default=settings.default_interval),
    period: str = Query(default=settings.default_period),
    category: str | None = Query(default=None, pattern="^(forex|metal)$"),
    include_news: bool = Query(default=settings.enable_news_analysis),
    include_closed: bool = Query(default=False),
    _: UserPublic = Depends(current_user),
) -> list[Signal]:
    validate_timeframe(interval, period)
    output: list[Signal] = []
    errors: list[str] = []
    analyzed_count = 0

    for market in markets(include_closed=include_closed):
        if market.code not in FOCUS_MARKETS:
            continue
        if category and market.category != category:
            continue
        try:
            frame = fetch_candles(market, interval=interval, period=period)
            market_news = fetch_news_sentiment(market) if include_news else None
            contexts, timeframe_warnings = timeframe_contexts(market, interval)
            signal = analyze_market(
                market,
                frame,
                interval=interval,
                period=period,
                news=market_news,
                learner=learner,
                timeframes=contexts,
            )
            signal.warnings.extend(timeframe_warnings)
            analyzed_count += 1
            learner.register_prediction(
                market_code=market.code,
                direction=signal.direction,
                entry_price=signal.risk.entry if signal.risk else signal.last_candle.close,
                features=signal.features or {},
                interval=interval,
                period=period,
                timestamp=datetime.fromisoformat(signal.timestamp),
            )
            tier = trade_candidate_tier(signal)
            if tier:
                output.append(signal)
                send_viable_entry_alert(signal, tier=tier)
        except Exception as exc:  # Keep one bad data source from hiding other signals.
            errors.append(f"{market.code}: {exc}")

    if analyzed_count == 0 and errors:
        raise HTTPException(status_code=502, detail={"message": "No signals could be generated", "errors": errors})

    sorted_output = sorted(output, key=lambda signal: signal.confidence, reverse=True)
    record_signals(sorted_output, source="api_bulk")
    return sorted_output


@app.get("/signal-log", response_model=list[SignalLogEntry])
def signal_log(
    market: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    include_pending: bool = Query(default=True),
    _: UserPublic = Depends(current_user),
) -> list[SignalLogEntry]:
    return list_signal_log(market_code=market, limit=limit, include_pending=include_pending)


@app.post("/signal-log/resolve", response_model=SignalOutcomeStats)
def resolve_signal_log(_: UserPublic = Depends(admin_user)) -> SignalOutcomeStats:
    resolve_signal_outcomes()
    return signal_outcome_stats()


@app.get("/signal-log/stats", response_model=SignalOutcomeStats)
def signal_log_stats(_: UserPublic = Depends(current_user)) -> SignalOutcomeStats:
    resolve_signal_outcomes()
    return signal_outcome_stats()


@app.get("/research/regime/{code}", response_model=MarketRegime)
def market_regime(
    code: str,
    interval: str = Query(default="1h"),
    period: str = Query(default="1mo"),
    _: UserPublic = Depends(current_user),
) -> MarketRegime:
    market = get_market(code)
    return classify_regime(fetch_candles(market, interval=interval, period=period))


@app.get("/research/backtest/{code}")
def strategy_backtest(
    code: str,
    interval: str = Query(default="1d", pattern="^(1h|4h|1d)$"),
    period: str = Query(default="5y", pattern="^(1y|2y|5y|10y)$"),
    minimum_score: float = Query(default=52, ge=40, le=90),
    reward_risk: float = Query(default=1.5, ge=1, le=4),
    _: UserPublic = Depends(current_user),
) -> dict:
    market = get_market(code)
    frame = fetch_candles(market, interval=interval, period=period)
    result = backtest_frame(
        frame, market, minimum_strategy_score=minimum_score, reward_risk=reward_risk
    )
    result["trades"] = result["trades"][-250:]
    return result


@app.get("/research/monte-carlo/{code}")
def strategy_monte_carlo(
    code: str,
    interval: str = Query(default="1d", pattern="^(1h|4h|1d)$"),
    period: str = Query(default="5y", pattern="^(1y|2y|5y|10y)$"),
    simulations: int = Query(default=1000, ge=100, le=10000),
    _: UserPublic = Depends(current_user),
) -> dict:
    market = get_market(code)
    backtest = backtest_frame(fetch_candles(market, interval=interval, period=period), market)
    return {"backtest_metrics": backtest["metrics"], "simulation": monte_carlo(backtest["trades"], simulations=simulations)}


@app.get("/research/optimize/{code}")
def strategy_optimization(
    code: str,
    interval: str = Query(default="1d", pattern="^(4h|1d)$"),
    period: str = Query(default="5y", pattern="^(2y|5y|10y)$"),
    _: UserPublic = Depends(admin_user),
) -> dict:
    market = get_market(code)
    return optimize_strategy(fetch_candles(market, interval=interval, period=period), market)


@app.get("/research/attribution")
def strategy_attribution(_: UserPublic = Depends(current_user)) -> dict:
    entries = [entry.model_dump() for entry in list_signal_log(limit=1000, include_pending=False)]
    return attribute_outcomes(entries)


@app.post("/research/macro", response_model=MacroAssessment)
def macro_assessment(payload: MacroInputs, _: UserPublic = Depends(current_user)) -> MacroAssessment:
    return assess_macro(payload)


@app.post("/research/portfolio-risk")
def portfolio_risk(
    positions: list[dict],
    account_balance: float = Query(gt=0),
    _: UserPublic = Depends(current_user),
) -> dict:
    return {
        "currency_exposure": currency_exposure(positions),
        "warnings": exposure_warnings(positions, account_balance),
    }


@app.get("/signals/{code}", response_model=Signal)
def signal(
    code: str,
    interval: str = Query(default=settings.default_interval),
    period: str = Query(default=settings.default_period),
    include_news: bool = Query(default=settings.enable_news_analysis),
    _: UserPublic = Depends(current_user),
) -> Signal:
    validate_timeframe(interval, period)
    try:
        market = get_market(code)
        market = attach_market_status(market)
        if settings.filter_closed_markets and not market.is_open:
            raise HTTPException(status_code=409, detail=market.closed_reason or "Market is closed")
        frame = fetch_candles(market, interval=interval, period=period)
        market_news = fetch_news_sentiment(market) if include_news else None
        contexts, timeframe_warnings = timeframe_contexts(market, interval)
        result = analyze_market(market, frame, interval=interval, period=period, news=market_news, learner=learner, timeframes=contexts)
        result.warnings.extend(timeframe_warnings)
        learner.register_prediction(
            market_code=market.code,
            direction=result.direction,
            entry_price=result.risk.entry if result.risk else result.last_candle.close,
            features=result.features or {},
            interval=interval,
            period=period,
            timestamp=datetime.fromisoformat(result.timestamp),
        )
        record_signal(result, source="api_single")
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if settings.enable_background_worker:
    from app.bot import main as worker_main

    @app.on_event("startup")
    def start_background_worker() -> None:
        threading.Thread(target=worker_main, name="signal-worker", daemon=True).start()


if settings.static_dir and Path(settings.static_dir).is_dir():
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")
