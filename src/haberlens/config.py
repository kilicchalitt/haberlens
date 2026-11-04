"""Merkezi yapılandırma.

Tüm ayarlar ortam değişkenleriyle (``HABERLENS_*``) geçersiz kılınabilir; böylece
aynı imaj geliştirme, test ve üretim ortamlarında kod değişikliği olmadan çalışır.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Yollar çalışma dizinine göredir; böylece paket site-packages'a kurulduğunda da
# (Docker, pip install) proje kökünden çalıştırılan komutlar doğru dosyaları bulur.
DEFAULT_DATA_PATH = Path(os.getenv("HABERLENS_DATA_PATH", "data/raw/turkish_news.csv"))
DEFAULT_MODEL_PATH = Path(os.getenv("HABERLENS_MODEL_PATH", "models/haberlens.joblib"))
DEFAULT_TEXT_COLUMN = "HABERLER"
DEFAULT_LABEL_COLUMN = "ETIKET"
RANDOM_STATE = 42


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - savunmacı kontrol
        raise ValueError(f"{name} bir tam sayı olmalıdır, alınan değer: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Çalışma zamanı ayarları."""

    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("HABERLENS_MODEL_PATH", "models/haberlens.joblib"))
    )
    api_key: str | None = field(default_factory=lambda: os.getenv("HABERLENS_API_KEY") or None)
    max_batch_size: int = field(default_factory=lambda: _env_int("HABERLENS_MAX_BATCH_SIZE", 256))
    max_text_length: int = field(
        default_factory=lambda: _env_int("HABERLENS_MAX_TEXT_LENGTH", 20_000)
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip()
            for origin in os.getenv("HABERLENS_CORS_ORIGINS", "").split(",")
            if origin.strip()
        )
    )


def get_settings() -> Settings:
    """Ortam değişkenlerini o anki değerleriyle okuyarak yeni bir ``Settings`` döndürür."""
    return Settings()
