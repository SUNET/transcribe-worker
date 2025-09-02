import os
from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from utils.args import parse_arguments

_, _, _, envfile, _, _ = parse_arguments()


class Settings(BaseSettings):
    """
    Settings for the application.
    """

    model_config = SettingsConfigDict(
        env_file=envfile,
        env_file_encoding="utf-8",
        case_sensitive=True,
        validate_assignment=True,
    )

    DEBUG: bool = True
    WORKERS: int = 2
    FILE_STORAGE_DIR: str = "/storage"
    API_BACKEND_URL: str = ""
    API_VERSION: str = "v1"
    FFMPEG_PATH: str = "ffmpeg"
    WHISPER_BACKEND: str = "cpp"

    # HF Whisper configuration.
    HF_WHISPER: bool = False
    HF_TOKEN: str = ""

    # SSL configuration
    SSL_CERTFILE: str = ""
    SSL_KEYFILE: str = ""

    # whisper.cpp path
    WHISPER_CPP_PATH: str = "whisper-cli"

    # Mapping between language and model
    WHISPER_MODELS_CPP: ClassVar[dict[str, dict[str, str]]] = {
        "Swedish": {
            "tiny": "sv_tiny.bin",
            "base": "sv_base.bin",
            "small": "sv_small.bin",
            "medium": "sv_medium.bin",
            "large": "sv_large.bin",
        },
        "English": {
            "tiny": "whisper_tiny.bin",
            "base": "whisper_base.bin",
            "small": "whisper_small.bin",
            "medium": "whisper_medium.bin",
            "large": "whisper_large.bin",
        },
        "Finnish": {
            "tiny": "whisper_tiny.bin",
            "base": "whisper_base.bin",
            "small": "whisper_small.bin",
            "medium": "whisper_medium.bin",
            "large": "whisper_large.bin",
        },
        "Danish": {
            "tiny": "whisper_tiny.bin",
            "base": "whisper_base.bin",
            "small": "whisper_small.bin",
            "medium": "whisper_medium.bin",
            "large": "whisper_large.bin",
        },
        "Norwegian": {
            "tiny": "whisper_tiny.bin",
            "base": "whisper_base.bin",
            "small": "whisper_small.bin",
            "medium": "whisper_medium.bin",
            "large": "whisper_large.bin",
        },
    }

    WHISPER_MODELS_HF: ClassVar[dict[str, dict[str, str]]] = {
        "Swedish": {
            "tiny": "kblab/kb-whisper-tiny",
            "base": "kblab/kb-whisper-base",
            "small": "kblab/kb-whisper-small",
            "medium": "kblab/kb-whisper-medium",
            "large": "kblab/kb-whisper-large",
        },
        "English": {
            "tiny": "openai/whisper-tiny-en",
            "base": "openai/whisper-base-en",
            "small": "openai/whisper-small-en",
            "medium": "openai/whisper-medium-en",
            "large": "openai/whisper-large-en",
        },
        "Finnish": {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v2",
        },
        "Danish": {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v2",
        },
        "Norwegian": {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v2",
        },
    }

    WHISPER_MODELS_MLX: ClassVar[dict[str, dict[str, str]]] = {
        "Swedish": {
            "tiny": "mlx/sv_tiny",
            "base": "mlx/sv_base",
            "small": "mlx/sv_small",
            "medium": "mlx/sv_medium",
            "large": "mlx/sv_large",
        },
    }


@lru_cache
def get_settings() -> Settings:
    """
    Get the settings for the application.
    """
    if not os.path.exists(Settings().FILE_STORAGE_DIR):
        os.makedirs(Settings().FILE_STORAGE_DIR)

    return Settings()
