from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = "local-model"
    request_timeout_seconds: float = 120.0
    pdf_dpi: int = 200
    max_pdf_pages: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
