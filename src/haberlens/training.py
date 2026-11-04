"""Uçtan uca eğitim akışı: veri → doğrulama → eğitim → değerlendirme → kayıt."""

from __future__ import annotations

import hashlib
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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
from haberlens.evaluation import evaluate_model
from haberlens.pipelines import ARCHITECTURES, DEFAULT_ARCHITECTURE, build_pipeline

__all__ = ["DEFAULT_LEAD_WORDS", "TrainingResult", "augment_with_leads", "train_model"]

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


DEFAULT_LEAD_WORDS = 12


def augment_with_leads(
    texts: pd.Series, labels: pd.Series, n_words: int = DEFAULT_LEAD_WORDS
) -> tuple[pd.Series, pd.Series]:
    """Uzun haberlerin ilk ``n_words`` kelimesini ek eğitim örneği olarak ekler.

    Kullanıcılar çoğu zaman yalnızca başlık gönderir; eğitim verisi ise büyük
    ölçüde tam metinlerden oluşur. Haber metinleri başlıkla başladığı için ilk
    kelimeler başlığın iyi bir vekilidir. Bu artırım yalnızca eğitim bölümüne
    uygulanır; test verisine asla eklenmez.
    """
    if not n_words:
        return texts.reset_index(drop=True), labels.reset_index(drop=True)
    texts, labels = texts.reset_index(drop=True), labels.reset_index(drop=True)
    words = texts.str.split()
    long_mask = words.str.len() > 2 * n_words
    leads = words[long_mask].str[:n_words].str.join(" ")
    return (
        pd.concat([texts, leads], ignore_index=True),
        pd.concat([labels, labels[long_mask]], ignore_index=True),
    )


def _fit(architecture: str, texts: pd.Series, labels: pd.Series, lead_words: int) -> Any:
    augmented_texts, augmented_labels = augment_with_leads(texts, labels, lead_words)
    return build_pipeline(architecture).fit(augmented_texts, augmented_labels)


def train_model(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    architecture: str = DEFAULT_ARCHITECTURE,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    labels: list[str] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 0,
    lead_words: int = DEFAULT_LEAD_WORDS,
    refit_full: bool = True,
    random_state: int = RANDOM_STATE,
) -> TrainingResult:
    """Modeli eğitir, ayrılmış test kümesinde değerlendirir ve diske kaydeder.

    Args:
        cv_folds: Eğitim kümesi üzerinde yapılacak çapraz doğrulama katman sayısı.
            ``0`` verilirse çapraz doğrulama atlanır (büyük derlemde varsayılan).
        lead_words: Başlık artırımında kullanılacak kelime sayısı (``0``: kapalı).
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

    cv_summary: dict[str, float] | None = None
    if cv_folds and cv_folds > 1:
        logger.info("%d katlı çapraz doğrulama yapılıyor...", cv_folds)
        folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, valid_idx in folds.split(split.X_train, split.y_train):
            model = _fit(
                architecture,
                split.X_train.iloc[train_idx],
                split.y_train.iloc[train_idx],
                lead_words,
            )
            predicted = model.predict(split.X_train.iloc[valid_idx])
            scores.append(f1_score(split.y_train.iloc[valid_idx], predicted, average="macro"))
        cv_summary = {
            "folds": cv_folds,
            "f1_macro_mean": round(float(np.mean(scores)), 4),
            "f1_macro_std": round(float(np.std(scores)), 4),
        }

    logger.info("'%s' mimarisi eğitiliyor...", architecture)
    pipeline = _fit(architecture, split.X_train, split.y_train, lead_words)
    metrics = evaluate_model(
        pipeline,
        split.X_test,
        split.y_test,
        class_labels,
        sources=split.source_test,
        lead_words=lead_words or DEFAULT_LEAD_WORDS,
    )
    metrics["test_size"] = len(split.X_test)
    if cv_summary:
        metrics["cross_validation"] = cv_summary

    if refit_full:
        logger.info("Son model tüm veriyle yeniden eğitiliyor...")
        pipeline = _fit(architecture, frame[TEXT], frame[LABEL], lead_words)

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
        "lead_augmentation_words": lead_words,
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
