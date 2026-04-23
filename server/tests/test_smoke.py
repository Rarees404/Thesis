"""Lightweight checks that do not load SigLIP, FAISS, or SAM."""


def test_repo_resolve_exists():
    from src.config import repo_root, resolve_repo

    root = repo_root()
    assert (root / "server" / "src" / "config.py").is_file()
    assert resolve_repo("data/visual_genome").endswith("data/visual_genome")


def test_settings_load():
    from src.config import settings

    assert settings.ollama_url.startswith("http")
