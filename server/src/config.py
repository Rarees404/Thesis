import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def server_dir() -> Path:
    """Directory containing server/.env in dev and /app/.env in Docker."""
    src_parent = Path(__file__).resolve().parent.parent
    return src_parent


def _default_repo_root() -> Path:
    srv = server_dir()
    return srv.parent if srv.name == "server" else srv


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=server_dir() / ".env", extra="ignore")

    base_dir: Path = server_dir()

    config_path: Path | None = None
    index_path: Path | None = None
    logs_path: Path | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2-vision"
    ollama_enabled: bool = True
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    sam_backend: str = "sam3"

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
    """VisualRef repository root, independent of process cwd and Docker layout."""
    configured = os.environ.get("VISUALREF_REPO_ROOT") or os.environ.get("REPO_ROOT")
    return Path(configured).resolve() if configured else _default_repo_root()


def resolve_repo(p: str | Path) -> str:
    """Resolve a path stored in configs or `image_paths.txt` relative to the repo root."""
    p = Path(p)
    if p.is_absolute():
        return str(p)
    return str(repo_root() / p)
