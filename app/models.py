from pydantic import BaseModel, Field


class Market(BaseModel):
    code: str
    symbol: str
    name: str
    category: str
    session: str | None = None
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
    features: dict[str, float] | None = None
    news: NewsSentiment | None = None
    model: ModelSignal | None = None
    risk: RiskPlan | None
    last_candle: Candle
