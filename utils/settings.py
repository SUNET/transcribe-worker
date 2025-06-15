import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Settings for the application.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        validate_assignment=True,
    )

    API_DEBUG: bool = True
    API_BACKEND_URL: str = ""
    API_FILE_STORAGE_DIR: str = "/storage"
    API_VERSION: str = "v1"
    API_WORKERS: int = 2

    # OIDC configuration.
    OIDC_TOKEN: str = ""

    # HF Whisper configuration.
    HF_WHISPER: bool = False
    HF_TOKEN: str = ""


@lru_cache
def get_settings() -> Settings:
    """
    Get the settings for the application.
    """
    if not os.path.exists(Settings().API_FILE_STORAGE_DIR):
        os.makedirs(Settings().API_FILE_STORAGE_DIR)

    return Settings()
