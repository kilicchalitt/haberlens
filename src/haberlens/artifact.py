"""Eğitilmiş modelin diske kaydedilmesi ve yüklenmesi.

Model; pipeline, kategori listesi ve izlenebilirlik için meta veriler (eğitim
zamanı, veri özeti, kütüphane sürümleri, metrikler) ile birlikte tek bir
``.joblib`` dosyasında saklanır. Metrikler ayrıca insan tarafından okunabilir bir
``.json`` dosyasına yazılır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

__all__ = ["ARTIFACT_FORMAT_VERSION", "ModelArtifact", "ModelLoadError"]

ARTIFACT_FORMAT_VERSION = 1


class ModelLoadError(RuntimeError):
    """Model dosyası bulunamadığında veya uyumsuz olduğunda fırlatılır."""


@dataclass
class ModelArtifact:
    """Kaydedilen modelin içeriği."""

    pipeline: Pipeline
    labels: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "pipeline": self.pipeline,
            "labels": self.labels,
            "metadata": self.metadata,
        }
        joblib.dump(payload, path, compress=3)
        path.with_suffix(".json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        path = Path(path)
        if not path.is_file():
            raise ModelLoadError(
                f"Model dosyası bulunamadı: {path}. Önce `haberlens train` komutunu çalıştırın."
            )
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise ModelLoadError(f"{path} geçerli bir HaberLens model dosyası değil.")
        version = payload.get("format_version")
        if version != ARTIFACT_FORMAT_VERSION:
            raise ModelLoadError(
                f"Model dosyası sürümü ({version}) bu HaberLens sürümüyle uyumsuz "
                f"(beklenen: {ARTIFACT_FORMAT_VERSION}). Modeli yeniden eğitin."
            )
        return cls(
            pipeline=payload["pipeline"],
            labels=list(payload["labels"]),
            metadata=dict(payload.get("metadata", {})),
        )
