from functools import lru_cache

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from exceptions import SettingsError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    google_api_key: str = Field(min_length=1)
    database_url: str = Field(min_length=1)
    llm_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "models/gemini-embedding-001"
    collection_name: str = "knowledge_base"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as err:
        raise SettingsError("invalid or missing application settings") from err
