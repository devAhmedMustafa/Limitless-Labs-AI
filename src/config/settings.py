import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():  # gracefully ignore missing .env during tests/deployments
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


@dataclass
class Settings:
    gemini_api_key: str | None
    gemini_model_name: str = DEFAULT_GEMINI_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL)
        return cls(gemini_api_key=api_key, gemini_model_name=model)
