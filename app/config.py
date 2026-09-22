from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ANALYSIS_TIMEFRAMES = {
    "4h": "1mo",
    "1d": "3mo",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cors_origins: str = Field(
        default="http://localhost:5173,https://bot-frontend-sooty.vercel.app",
        alias="CORS_ORIGINS",
    )
    default_interval: str = Field(default="4h", alias="DEFAULT_INTERVAL")
    default_period: str = Field(default="1mo", alias="DEFAULT_PERIOD")
    cache_ttl_seconds: int = Field(default=120, alias="CACHE_TTL_SECONDS")
    bot_poll_seconds: int = Field(default=900, alias="BOT_POLL_SECONDS")
    enable_news_analysis: bool = Field(default=True, alias="ENABLE_NEWS_ANALYSIS")
    news_cache_ttl_seconds: int = Field(default=900, alias="NEWS_CACHE_TTL_SECONDS")
    news_lookback_hours: int = Field(default=24, alias="NEWS_LOOKBACK_HOURS")
    filter_closed_markets: bool = Field(default=True, alias="FILTER_CLOSED_MARKETS")
    enable_learning: bool = Field(default=True, alias="ENABLE_LEARNING")
    enable_background_worker: bool = Field(default=False, alias="ENABLE_BACKGROUND_WORKER")
    enable_session_suggestions: bool = Field(default=True, alias="ENABLE_SESSION_SUGGESTIONS")
    auto_dst_sessions: bool = Field(default=True, alias="AUTO_DST_SESSIONS")
    session_timezone: str = Field(default="Africa/Nairobi", alias="SESSION_TIMEZONE")
    asia_session_open: str = Field(default="03:00", alias="ASIA_SESSION_OPEN")
    asia_session_close: str = Field(default="11:00", alias="ASIA_SESSION_CLOSE")
    london_session_open: str = Field(default="10:00", alias="LONDON_SESSION_OPEN")
    london_session_close: str = Field(default="19:00", alias="LONDON_SESSION_CLOSE")
    new_york_session_open: str = Field(default="16:00", alias="NEW_YORK_SESSION_OPEN")
    new_york_session_close: str = Field(default="00:00", alias="NEW_YORK_SESSION_CLOSE")
    learning_horizon_hours: int = Field(default=72, alias="LEARNING_HORIZON_HOURS")
    learning_min_move_pct: float = Field(default=0.0015, alias="LEARNING_MIN_MOVE_PCT")
    learning_rate: float = Field(default=0.08, alias="LEARNING_RATE")
    session_alignment_boost: float = Field(default=0.4, alias="SESSION_ALIGNMENT_BOOST")
    session_offsession_penalty: float = Field(default=0.2, alias="SESSION_OFFSESSION_PENALTY")
    model_state_path: str = Field(default="data/model_state.json", alias="MODEL_STATE_PATH")
    auth_state_path: str = Field(default="data/users.json", alias="AUTH_STATE_PATH")
    auth_secret_key: str = Field(default="change-this-auth-secret", alias="AUTH_SECRET_KEY")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    account_currency: str = Field(default="USD", alias="ACCOUNT_CURRENCY")
    auth_token_ttl_hours: int = Field(default=24, alias="AUTH_TOKEN_TTL_HOURS")
    risk_account_balance: float = Field(default=1000.0, alias="RISK_ACCOUNT_BALANCE")
    risk_percent: float = Field(default=1.0, alias="RISK_PERCENT")
    max_stop_atr: float = Field(default=2.0, alias="MAX_STOP_ATR")
    take_profit_1_r: float = Field(default=1.0, alias="TAKE_PROFIT_1_R")
    take_profit_2_r: float = Field(default=1.5, alias="TAKE_PROFIT_2_R")
    execution_cost_bps_forex: float = Field(default=1.0, alias="EXECUTION_COST_BPS_FOREX")
    execution_cost_bps_metal: float = Field(default=3.0, alias="EXECUTION_COST_BPS_METAL")
    signal_log_path: str = Field(default="data/signal_log.json", alias="SIGNAL_LOG_PATH")
    signal_outcome_horizon_hours: int = Field(default=72, alias="SIGNAL_OUTCOME_HORIZON_HOURS")
    signal_outcome_min_move_pct: float = Field(default=0.0015, alias="SIGNAL_OUTCOME_MIN_MOVE_PCT")
    edge_min_samples: int = Field(default=30, alias="EDGE_MIN_SAMPLES")
    edge_max_samples: int = Field(default=100, alias="EDGE_MAX_SAMPLES")
    edge_decay: float = Field(default=0.98, alias="EDGE_DECAY")
    edge_prior_samples: float = Field(default=20.0, alias="EDGE_PRIOR_SAMPLES")
    edge_max_score_adjustment: float = Field(default=0.75, alias="EDGE_MAX_SCORE_ADJUSTMENT")
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")
    telegram_alert_state_path: str = Field(default="data/telegram_alerts.json", alias="TELEGRAM_ALERT_STATE_PATH")
    telegram_access_username: str | None = Field(default=None, alias="TELEGRAM_ACCESS_USERNAME")
    telegram_access_password_hash: str | None = Field(default=None, alias="TELEGRAM_ACCESS_PASSWORD_HASH")
    telegram_access_keyword_hash: str | None = Field(default=None, alias="TELEGRAM_ACCESS_KEYWORD_HASH")
    telegram_auth_state_path: str = Field(default="data/telegram_auth.json", alias="TELEGRAM_AUTH_STATE_PATH")
    telegram_assistant_state_path: str = Field(default="data/telegram_assistant.json", alias="TELEGRAM_ASSISTANT_STATE_PATH")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.4", alias="OPENAI_MODEL")
    openai_background_poll_seconds: int = Field(default=3, alias="OPENAI_BACKGROUND_POLL_SECONDS")
    openai_background_timeout_seconds: int = Field(default=900, alias="OPENAI_BACKGROUND_TIMEOUT_SECONDS")
    telegram_market_update_hours: int = Field(default=0, alias="TELEGRAM_MARKET_UPDATE_HOURS")
    minimum_alert_confidence: int = Field(default=72, alias="MINIMUM_ALERT_CONFIDENCE")
    model_resolved_retention: int = Field(default=5000, alias="MODEL_RESOLVED_RETENTION")
    signal_log_retention: int = Field(default=10000, alias="SIGNAL_LOG_RETENTION")
    metaapi_token: str | None = Field(default=None, alias="METAAPI_TOKEN")
    metaapi_account_id: str | None = Field(default=None, alias="METAAPI_ACCOUNT_ID")
    metaapi_region: str = Field(default="new-york", alias="METAAPI_REGION")
    require_live_price_for_alerts: bool = Field(default=True, alias="REQUIRE_LIVE_PRICE_FOR_ALERTS")
    enable_yahoo_live_fallback: bool = Field(default=True, alias="ENABLE_YAHOO_LIVE_FALLBACK")
    live_price_max_age_seconds: int = Field(default=45, alias="LIVE_PRICE_MAX_AGE_SECONDS")
    live_price_max_deviation_atr: float = Field(default=0.35, alias="LIVE_PRICE_MAX_DEVIATION_ATR")
    live_price_max_spread_bps_forex: float = Field(default=5.0, alias="LIVE_PRICE_MAX_SPREAD_BPS_FOREX")
    live_price_max_spread_bps_metal: float = Field(default=20.0, alias="LIVE_PRICE_MAX_SPREAD_BPS_METAL")
    telegram_login_max_attempts: int = Field(default=5, alias="TELEGRAM_LOGIN_MAX_ATTEMPTS")
    telegram_login_lockout_minutes: int = Field(default=15, alias="TELEGRAM_LOGIN_LOCKOUT_MINUTES")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def validate_security(self) -> None:
        if self.environment.lower() == "production" and self.auth_secret_key == "change-this-auth-secret":
            raise RuntimeError("AUTH_SECRET_KEY must be set to a strong unique value in production")


@lru_cache
def get_settings() -> Settings:
    return Settings()
