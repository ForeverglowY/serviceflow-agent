from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    deepseek_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
    )
    deepseek_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    database_connect_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
    )
    database_pool_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=60,
    )
    database_statement_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
    )


settings = Settings()
