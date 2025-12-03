import os

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar
from utils.args import parse_arguments

_, _, _, envfile, _, _, _, _ = parse_arguments()


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
    HF_TOKEN: str = ""

    # SSL configuration
    SSL_CERTFILE: str = ""
    SSL_KEYFILE: str = ""

    # Mapping between language and model
    WHISPER_MODELS_HF: ClassVar[dict[str, dict[str, str]]] = {
        "Swedish": {
            "fast transcription (normal accuracy)": "kblab/kb-whisper-base",
            "slower transcription (higher accuracy)": "kblab/kb-whisper-large",
        },
        "English": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Finnish": {
            "fast transcription (normal accuracy)": "Finnish-NLP/whisper-medium-finnish",
            "slower transcription (higher accuracy)": "Finnish-NLP/whisper-large-finnish-v3",
        },
        "Danish": {
            "fast transcription (normal accuracy)": "syvai/hviske-v2",
            "slower transcription (higher accuracy)": "syvai/hviske-v2",
        },
        "Norwegian": {
            "fast transcription (normal accuracy)": "NbAiLab/nb-whisper-medium",
            "slower transcription (higher accuracy)": "NbAiLab/nb-whisper-large",
        },
        "French": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "German": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Spanish": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Italian": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Russian": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Ukrainian": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
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
