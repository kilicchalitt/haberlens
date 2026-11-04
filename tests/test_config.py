from pathlib import Path

import pytest

from haberlens.config import get_settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HABERLENS_MODEL_PATH",
        "HABERLENS_API_KEY",
        "HABERLENS_MAX_BATCH_SIZE",
        "HABERLENS_CORS_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = get_settings()
    assert settings.model_path == Path("models/haberlens.joblib")
    assert settings.api_key is None
    assert settings.max_batch_size == 256
    assert settings.cors_origins == ()


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HABERLENS_MODEL_PATH", "/srv/model.joblib")
    monkeypatch.setenv("HABERLENS_API_KEY", "anahtar")
    monkeypatch.setenv("HABERLENS_MAX_BATCH_SIZE", "10")
    monkeypatch.setenv("HABERLENS_CORS_ORIGINS", "https://a.com, https://b.com ,")
    settings = get_settings()
    assert settings.model_path == Path("/srv/model.joblib")
    assert settings.api_key == "anahtar"
    assert settings.max_batch_size == 10
    assert settings.cors_origins == ("https://a.com", "https://b.com")


def test_empty_api_key_disables_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HABERLENS_API_KEY", "")
    assert get_settings().api_key is None
