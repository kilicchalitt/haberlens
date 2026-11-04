"""İsteğe bağlı yoğun (dense) metin temsili: Word2Vec.

Bu modül yalnızca karşılaştırmalı deneyler (benchmark) için kullanılır ve
``pip install 'haberlens[embeddings]'`` ile kurulan ``gensim`` bağımlılığına
ihtiyaç duyar. Üretim modeli bu bağımlılık olmadan çalışır.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from haberlens.config import RANDOM_STATE

__all__ = ["Word2VecVectorizer"]

logger = logging.getLogger(__name__)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Metinleri, eğitim verisinde öğrenilen Word2Vec vektörlerinin ortalamasıyla temsil eder.

    Pipeline içinde ``fit`` edildiği için çapraz doğrulamada test katmanının
    kelime dağarcığı eğitime sızmaz.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 20,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: Iterable[str], y: object = None) -> Word2VecVectorizer:
        from gensim.models import Word2Vec

        sentences = [text.split() for text in X]
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed,
            workers=1,  # deterministik sonuç için
        )
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        vectors = self.model_.wv
        rows = []
        for text in X:
            tokens = [token for token in text.split() if token in vectors]
            rows.append(
                np.mean([vectors[t] for t in tokens], axis=0)
                if tokens
                else np.zeros(self.vector_size, dtype=np.float32)
            )
        return np.vstack(rows)
