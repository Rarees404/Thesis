# server/src/config.py
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_dir: Path = Path(".").resolve()

    config_path: Path | None = None
    index_path: Path | None = None
    logs_path: Path | None = None
    captioning_config_path: Path | None = None

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2-vision"
    ollama_enabled: bool = True

    # "auto" = try SAM3 then SAM2, "sam3" = SAM3 only, "sam2" = SAM2 only
    sam_backend: str = "auto"

    @field_validator(
        "config_path",
        "captioning_config_path",
        "index_path",
        "logs_path",
        mode="before",
    )
    @classmethod
    def _resolve_paths(cls, v, info):
        if v is None:
            return v
        p = Path(v)
        base: Path = info.data.get("base_dir", Path(".").resolve())
        return p if p.is_absolute() else (base / p)


settings = ServerSettings()


def resolve_repo(p: str | Path) -> str:
    p = Path(p)
    base = settings.base_dir
    return str(p if p.is_absolute() else (base / p))
