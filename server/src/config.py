from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_dir: Path = Path(".").resolve()

    config_path: Path | None = None
    index_path: Path | None = None
    logs_path: Path | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2-vision"
    ollama_enabled: bool = True

    sam_backend: str = "sam2"

    @field_validator(
        "config_path",
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


def repo_root() -> Path:
    """VisualRef repository root (parent of `server/`), independent of process cwd."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_repo(p: str | Path) -> str:
    """Resolve a path stored in configs or `image_paths.txt` relative to the repo root."""
    p = Path(p)
    if p.is_absolute():
        return str(p)
    return str(repo_root() / p)
