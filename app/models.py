from pydantic import BaseModel, Field


class Market(BaseModel):
    code: str
    symbol: str
    name: str
    category: str
    session: str | None = None
    preferred_sessions: list[str] = Field(default_factory=list)
    is_open: bool | None = None
    closed_reason: str | None = None


class Candle(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class RiskPlan(BaseModel):
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    risk_percent: float
    risk_amount: float
    suggested_lot_size: float


class TimeframeContext(BaseModel):
    interval: str
    period: str
    direction: str
    score: float
    close: float
    ema_9: float
    ema_21: float
    rsi: float | None = None
    macd_delta: float
    summary: str


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    link: str | None = None
    published_at: str | None = None
    sentiment: str
    score: float


class NewsSentiment(BaseModel):
    sentiment: str
    score: float
    confidence: int = Field(ge=0, le=100)
    articles_analyzed: int
    bullish_terms: list[str]
    bearish_terms: list[str]
    headlines: list[NewsItem]
    error: str | None = None


class ModelSignal(BaseModel):
    probability: float
    adjustment: float
    samples_seen: int
    resolved_predictions: int
    accuracy: float | None = None
    bias: float


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=8, max_length=200)


class CreateUserRequest(LoginRequest):
    is_admin: bool = False
    is_active: bool = True


class UpdateUserRequest(BaseModel):
    password: str | None = Field(default=None, min_length=8, max_length=200)
    is_admin: bool | None = None
    is_active: bool | None = None


class UserPublic(BaseModel):
    id: str
    username: str
    is_admin: bool
    is_active: bool
    created_at: str
    updated_at: str
    last_login_at: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserPublic
    setup_admin: bool = False


class SessionSignal(BaseModel):
    current_session: str
    active_sessions: list[str]
    preferred_sessions: list[str]
    next_session: str | None = None
    next_session_start: str | None = None
    alignment: str
    suggestion: str
    score_adjustment: float
    confidence: int = Field(ge=0, le=100)
    reasons: list[str]


class Signal(BaseModel):
    market: Market
    interval: str
    period: str
    timestamp: str
    direction: str = Field(description="bullish, bearish, or neutral")
    confidence: int = Field(ge=0, le=100)
    score: float
    strategy: str
    reasons: list[str]
    warnings: list[str]
    indicators: dict[str, float | str | None]
    timeframes: dict[str, TimeframeContext] = Field(default_factory=dict)
    features: dict[str, float] | None = None
    news: NewsSentiment | None = None
    model: ModelSignal | None = None
    session: SessionSignal | None = None
    risk: RiskPlan | None
    last_candle: Candle
