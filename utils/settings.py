import os

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar
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
            "fast transcription (normal accuracy)": "sv_base.bin",
            "slower transcription (higher accuracy)": "sv_large.bin",
        },
        "English": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Finnish": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Danish": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "dk_large.bin",
        },
        "Norwegian": {
            "fast transcription (normal accuracy)": "no_base.bin",
            "slower transcription (higher accuracy)": "no_large.bin",
        },
        "French": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "German": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Spanish": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Italian": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Russian": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Ukrainian": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
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


@lru_cache
def get_settings() -> Settings:
    """
    Get the settings for the application.
    """
    if not os.path.exists(Settings().FILE_STORAGE_DIR):
        os.makedirs(Settings().FILE_STORAGE_DIR)

    return Settings()
