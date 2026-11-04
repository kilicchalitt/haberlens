"""Tahmin (inference) katmanı.

``NewsClassifier`` eğitilmiş modeli yükler ve hem tekil hem toplu tahmin yapar.
Her tahmin; olasılık dağılımını, düşük güven uyarısını ve kararı en çok
etkileyen anahtar kelimeleri içerir. Bu sayede sistem bir "kara kutu" olmaktan
çıkar ve editörler kararın gerekçesini görebilir.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline

from haberlens.artifact import ModelArtifact
from haberlens.config import get_settings

__all__ = ["DEFAULT_CONFIDENCE_THRESHOLD", "Keyword", "NewsClassifier", "Prediction"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Keyword:
    """Tahmine katkı sağlayan bir terim ve katkı ağırlığı."""

    term: str
    weight: float


@dataclass(frozen=True)
class Prediction:
    """Tek bir metin için tahmin sonucu.

    Attributes:
        label: En olası kategori. Metin anlamlı içerik barındırmıyorsa ``None``.
        confidence: ``label`` için model olasılığı (0-1).
        scores: En olası ``top_k`` kategori ve olasılıkları (büyükten küçüğe).
        uncertain: Güven eşiğin altındaysa ``True``; insan onayına yönlendirme için.
        keywords: Kararı en çok destekleyen terimler (``explain=True`` ise).
    """

    label: str | None
    confidence: float
    scores: dict[str, float]
    uncertain: bool
    keywords: list[Keyword] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NewsClassifier:
    """Eğitilmiş HaberLens modelini kullanarak haber metinlerini sınıflandırır.

    Örnek:
        >>> clf = NewsClassifier.load()                       # doctest: +SKIP
        >>> clf.predict("Merkez Bankası faizi sabit tuttu").label  # doctest: +SKIP
        'Ekonomi'
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 0 ile 1 arasında olmalıdır.")
        self._artifact = artifact
        self._pipeline: Pipeline = artifact.pipeline
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------ yükleme
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> NewsClassifier:
        """Modeli diskten yükler. ``path`` verilmezse ``HABERLENS_MODEL_PATH`` kullanılır."""
        model_path = Path(path) if path is not None else get_settings().model_path
        return cls(ModelArtifact.load(model_path), confidence_threshold=confidence_threshold)

    # ---------------------------------------------------------------- özellikler
    @property
    def labels(self) -> list[str]:
        return list(self._pipeline.classes_)

    @property
    def pipeline(self) -> Pipeline:
        """Alttaki eğitilmiş scikit-learn pipeline'ı (salt okunur kullanım için)."""
        return self._pipeline

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._artifact.metadata)

    # ------------------------------------------------------------------- tahmin
    def predict(self, text: str, *, top_k: int = 3, explain: bool = True) -> Prediction:
        """Tek bir metni sınıflandırır."""
        return self.predict_batch([text], top_k=top_k, explain=explain)[0]

    def predict_batch(
        self,
        texts: Sequence[str],
        *,
        top_k: int = 3,
        explain: bool = False,
        n_keywords: int = 5,
    ) -> list[Prediction]:
        """Birden fazla metni tek seferde (vektörize ederek) sınıflandırır."""
        if isinstance(texts, str):
            raise TypeError("predict_batch bir metin listesi bekler; tek metin için predict().")
        if not texts:
            return []

        top_k = max(1, min(top_k, len(self.labels)))
        normalizer = self._pipeline.steps[0][1]
        normalized = normalizer.transform(texts)
        features = self._pipeline[1:-1].transform(normalized)
        probabilities = self._pipeline.steps[-1][1].predict_proba(features)
        classes = self._pipeline.classes_

        predictions: list[Prediction] = []
        for row, (clean, probs) in enumerate(zip(normalized, probabilities, strict=True)):
            if not clean:
                predictions.append(Prediction(None, 0.0, {}, uncertain=True))
                continue
            order = np.argsort(probs)[::-1]
            best = int(order[0])
            confidence = float(probs[best])
            keywords = self._explain(features, row, best, n_keywords) if explain else []
            predictions.append(
                Prediction(
                    label=str(classes[best]),
                    confidence=round(confidence, 4),
                    scores={str(classes[i]): round(float(probs[i]), 4) for i in order[:top_k]},
                    uncertain=confidence < self.confidence_threshold,
                    keywords=keywords,
                )
            )
        return predictions

    def predict_labels(self, texts: Sequence[str]) -> list[str]:
        """Yalnızca etiketleri döndüren hızlı yol (değerlendirme için)."""
        return [str(label) for label in self._pipeline.predict(list(texts))]

    # --------------------------------------------------------------- açıklama
    @cached_property
    def _explainable_features(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(özellik adları, okunabilir-kelime maskesi) çiftini döndürür."""
        features = self._pipeline[1:-1]
        try:
            names = np.asarray(features.get_feature_names_out(), dtype=object)
        except (AttributeError, ValueError):  # pragma: no cover - özel pipeline'lar
            return None

        vectorizer = features.steps[-1][1] if isinstance(features, Pipeline) else features
        if isinstance(vectorizer, FeatureUnion):
            mask = np.array([name.startswith("word__") for name in names])
            names = np.array([name.split("__", 1)[-1] for name in names], dtype=object)
        elif getattr(vectorizer, "analyzer", "word") == "word":
            mask = np.ones(len(names), dtype=bool)
        else:
            return None
        return names, mask

    @cached_property
    def _linear_coef(self) -> np.ndarray | None:
        """Açıklama için kullanılacak doğrusal model ağırlıklarını bulur.

        Tek bir doğrusal model, topluluk içindeki Lojistik Regresyon veya kalibre
        edilmiş SVM'nin iç modeli desteklenir. Naive Bayes gibi doğrusal ağırlığı
        olmayan modeller için ``None`` döner ve açıklama üretilmez.
        """
        classifier = self._pipeline.steps[-1][1]
        candidates = [classifier]
        named = getattr(classifier, "named_estimators_", None)
        if named is not None:
            candidates = [named[key] for key in ("lr", "svm") if key in named]
        for candidate in candidates:
            coef = getattr(candidate, "coef_", None)
            if coef is None and hasattr(candidate, "calibrated_classifiers_"):
                coef = getattr(candidate.calibrated_classifiers_[0].estimator, "coef_", None)
            if coef is not None:
                return np.asarray(coef)
        return None

    def _explain(self, features: Any, row: int, class_index: int, limit: int) -> list[Keyword]:
        explainable = self._explainable_features
        coef = self._linear_coef
        if explainable is None or coef is None:
            return []

        names, mask = explainable
        if coef.shape[0] == 1:  # ikili sınıflandırma: tek ağırlık vektörü
            class_coef = coef[0] if class_index == 1 else -coef[0]
        else:
            class_coef = coef[class_index]

        vector = features[row]
        indices = vector.indices
        contributions = vector.data * class_coef[indices]
        keep = (contributions > 0) & mask[indices]
        indices, contributions = indices[keep], contributions[keep]
        top = np.argsort(contributions)[::-1][:limit]
        return [Keyword(str(names[indices[i]]), round(float(contributions[i]), 4)) for i in top]
