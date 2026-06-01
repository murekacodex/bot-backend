from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from app.analysis import analyze_market, summarize_timeframe
from app.auth import admin_user, create_user, current_user, delete_user, list_users, login_or_create_admin, update_user, users_exist
from app.config import get_settings
from app.learning import AdaptiveSignalModel
from app.market_data import dataframe_to_candles, fetch_candles
from app.markets import MARKETS, get_market
from app.models import AuthResponse, Candle, CreateUserRequest, LoginRequest, Market, NewsSentiment, Signal, UpdateUserRequest, UserPublic
from app.news import fetch_news_sentiment
from app.session import attach_market_status

settings = get_settings()
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
    "5m": {"lower": None, "higher": ("15m", "1d")},
    "15m": {"lower": ("5m", "1d"), "higher": ("1h", "5d")},
    "30m": {"lower": ("15m", "1d"), "higher": ("1h", "5d")},
    "1h": {"lower": ("15m", "1d"), "higher": ("1d", "3mo")},
    "4h": {"lower": ("1h", "5d"), "higher": ("1d", "3mo")},
    "1d": {"lower": ("1h", "5d"), "higher": ("1wk", "1y")},
}


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
    include_news: bool = Query(default=settings.enable_news_analysis),
    include_closed: bool = Query(default=False),
    _: UserPublic = Depends(current_user),
) -> list[Signal]:
    output: list[Signal] = []
    errors: list[str] = []

    for market in markets(include_closed=include_closed):
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
            output.append(signal)
        except Exception as exc:  # Keep one bad data source from hiding other signals.
            errors.append(f"{market.code}: {exc}")

    if not output:
        raise HTTPException(status_code=502, detail={"message": "No signals could be generated", "errors": errors})

    return sorted(output, key=lambda signal: signal.confidence, reverse=True)


@app.get("/signals/{code}", response_model=Signal)
def signal(
    code: str,
    interval: str = Query(default=settings.default_interval),
    period: str = Query(default=settings.default_period),
    include_news: bool = Query(default=settings.enable_news_analysis),
    _: UserPublic = Depends(current_user),
) -> Signal:
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
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
