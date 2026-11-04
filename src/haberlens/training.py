"""Uçtan uca eğitim akışı: veri → doğrulama → eğitim → değerlendirme → kayıt."""

from __future__ import annotations

import hashlib
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from haberlens import __version__
from haberlens.artifact import ModelArtifact
from haberlens.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEXT_COLUMN,
    RANDOM_STATE,
)
from haberlens.data import LABEL, TEXT, load_dataset, split_dataset, summarize
from haberlens.evaluation import evaluate_predictions
from haberlens.pipelines import ARCHITECTURES, DEFAULT_ARCHITECTURE, build_pipeline

__all__ = ["TrainingResult", "train_model"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Eğitim çıktısı."""

    model_path: Path
    metrics: dict[str, Any]
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_model(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    architecture: str = DEFAULT_ARCHITECTURE,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    labels: list[str] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    refit_full: bool = True,
    random_state: int = RANDOM_STATE,
) -> TrainingResult:
    """Modeli eğitir, ayrılmış test kümesinde değerlendirir ve diske kaydeder.

    Args:
        cv_folds: Eğitim kümesi üzerinde yapılacak çapraz doğrulama katman sayısı.
            ``0`` verilirse çapraz doğrulama atlanır.
        refit_full: ``True`` ise raporlanan metrikler ayrılmış test kümesinden
            hesaplandıktan sonra son model tüm veriyle yeniden eğitilir.
    """
    data_path = Path(data_path)
    frame = load_dataset(
        data_path, text_column=text_column, label_column=label_column, labels=labels
    )
    class_labels = sorted(frame[LABEL].unique())
    logger.info("%d kayıt, %d kategori yüklendi.", len(frame), len(class_labels))

    split = split_dataset(frame, test_size=test_size, random_state=random_state)
    pipeline = build_pipeline(architecture)

    cv_summary: dict[str, float] | None = None
    if cv_folds and cv_folds > 1:
        logger.info("%d katlı çapraz doğrulama yapılıyor...", cv_folds)
        folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            clone(pipeline), split.X_train, split.y_train, cv=folds, scoring="f1_macro", n_jobs=-1
        )
        cv_summary = {
            "folds": cv_folds,
            "f1_macro_mean": round(float(scores.mean()), 4),
            "f1_macro_std": round(float(scores.std()), 4),
        }

    logger.info("'%s' mimarisi eğitiliyor...", architecture)
    pipeline.fit(split.X_train, split.y_train)
    metrics = evaluate_predictions(split.y_test, pipeline.predict(split.X_test), class_labels)
    metrics["test_size"] = len(split.X_test)
    if cv_summary:
        metrics["cross_validation"] = cv_summary

    if refit_full:
        logger.info("Son model tüm veriyle yeniden eğitiliyor...")
        pipeline = build_pipeline(architecture).fit(frame[TEXT], frame[LABEL])

    metadata: dict[str, Any] = {
        "name": "HaberLens",
        "version": __version__,
        "architecture": architecture,
        "architecture_description": ARCHITECTURES[architecture].description,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": class_labels,
        "data": {
            "file": data_path.name,
            "sha256": _sha256(data_path),
            **summarize(frame),
        },
        "refit_on_full_data": refit_full,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    artifact = ModelArtifact(pipeline=pipeline, labels=class_labels, metadata=metadata)
    saved_path = artifact.save(output_path)
    logger.info("Model kaydedildi: %s", saved_path)
    return TrainingResult(model_path=saved_path, metrics=metrics, metadata=metadata)
