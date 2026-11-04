"""İsteğe bağlı yoğun (dense) metin temsilleri: Word2Vec ve BERT.

Bu modül yalnızca karşılaştırmalı deneyler (benchmark) için kullanılır ve
``pip install 'haberlens[embeddings]'`` ile kurulan ek bağımlılıklara ihtiyaç
duyar. Üretim modeli bu bağımlılıklar olmadan çalışır.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from haberlens.config import RANDOM_STATE

__all__ = ["DEFAULT_BERT_MODEL", "BertEmbedder", "Word2VecVectorizer"]

logger = logging.getLogger(__name__)

DEFAULT_BERT_MODEL = "dbmdz/bert-base-turkish-cased"


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


class BertEmbedder(BaseEstimator, TransformerMixin):
    """Önceden eğitilmiş Türkçe BERT ile metinleri sabit uzunluklu vektörlere dönüştürür.

    Orijinal çalışmadaki gibi metinleri tek tek işlemek yerine toplu (batch)
    işler ve ``[CLS]`` yerine maskeli ortalama (mean pooling) kullanır; bu,
    ince ayar (fine-tuning) yapılmayan senaryolarda genellikle daha iyi sonuç verir.
    BERT kendi alt-kelime (WordPiece) tokenizasyonunu kullandığından ham metin
    verilmelidir.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BERT_MODEL,
        max_length: int = 128,
        batch_size: int = 32,
        pooling: str = "mean",
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def fit(self, X: object = None, y: object = None) -> BertEmbedder:
        return self

    def _load(self) -> None:
        if getattr(self, "_model", None) is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info("BERT modeli yükleniyor: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def transform(self, X: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        texts = list(X)
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            outputs.append(pooled.cpu().numpy())
            done = min(start + self.batch_size, len(texts))
            if done % (self.batch_size * 20) == 0 or done == len(texts):
                logger.info("  BERT: %d/%d metin işlendi", done, len(texts))
        return np.vstack(outputs)
