from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: Literal["lm_studio", "gemini"] = "lm_studio"

    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = "local-model"
    lm_studio_enable_thinking: bool = False

    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    request_timeout_seconds: float = 120.0
    pdf_dpi: int = 300
    max_pdf_pages: int = 10
    max_retries: int = 2
    db_path: str = "lnvo_lens.db"
    max_workers_gemini: int = 4
    max_workers_lm_studio: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
