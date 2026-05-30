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

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
