from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cors_origins: str = Field(
        default="http://localhost:5173,https://bot-frontend-sooty.vercel.app",
        alias="CORS_ORIGINS",
    )
    default_interval: str = Field(default="1h", alias="DEFAULT_INTERVAL")
    default_period: str = Field(default="5d", alias="DEFAULT_PERIOD")
    cache_ttl_seconds: int = Field(default=120, alias="CACHE_TTL_SECONDS")
    bot_poll_seconds: int = Field(default=300, alias="BOT_POLL_SECONDS")
    enable_news_analysis: bool = Field(default=True, alias="ENABLE_NEWS_ANALYSIS")
    news_cache_ttl_seconds: int = Field(default=900, alias="NEWS_CACHE_TTL_SECONDS")
    news_lookback_hours: int = Field(default=24, alias="NEWS_LOOKBACK_HOURS")
    filter_closed_markets: bool = Field(default=True, alias="FILTER_CLOSED_MARKETS")
    enable_learning: bool = Field(default=True, alias="ENABLE_LEARNING")
    enable_session_suggestions: bool = Field(default=True, alias="ENABLE_SESSION_SUGGESTIONS")
    learning_horizon_hours: int = Field(default=24, alias="LEARNING_HORIZON_HOURS")
    learning_min_move_pct: float = Field(default=0.0015, alias="LEARNING_MIN_MOVE_PCT")
    learning_rate: float = Field(default=0.08, alias="LEARNING_RATE")
    session_alignment_boost: float = Field(default=0.4, alias="SESSION_ALIGNMENT_BOOST")
    session_offsession_penalty: float = Field(default=0.2, alias="SESSION_OFFSESSION_PENALTY")
    model_state_path: str = Field(default="data/model_state.json", alias="MODEL_STATE_PATH")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
