"""Model mimarileri (metin temsili + sınıflandırıcı) için fabrika fonksiyonları.

Her mimari, ön işlemeden sınıflandırıcıya kadar tek bir scikit-learn
``Pipeline`` nesnesidir. Böylece model tek dosya olarak kaydedilir, yüklenir ve
sürümlenir.

Varsayılan mimari ``hybrid``: kelime (1-2 gram) ve karakter (2-5 gram) TF-IDF
özniteliklerini birleştirir. Türkçe sondan eklemeli bir dil olduğu için
karakter n-gramları ``ekonomi / ekonomik / ekonomide`` gibi çekimli biçimleri
ortak köklerinden yakalar; kelime n-gramları ise okunabilir açıklamalar üretir.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from haberlens.config import RANDOM_STATE
from haberlens.preprocessing import TurkishTextNormalizer

__all__ = ["ARCHITECTURES", "DEFAULT_ARCHITECTURE", "Architecture", "build_pipeline"]

DEFAULT_ARCHITECTURE = "hybrid"


def _classifier(c: float = 10.0) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE)


def _word_tfidf(ngram_range: tuple[int, int] = (1, 1)) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True, min_df=1)


def _char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


def _bow() -> Pipeline:
    return Pipeline(
        [
            ("normalize", TurkishTextNormalizer()),
            ("features", CountVectorizer()),
            ("classifier", _classifier(c=1.0)),
        ]
    )


def _tfidf() -> Pipeline:
    return Pipeline(
        [
            ("normalize", TurkishTextNormalizer()),
            ("features", _word_tfidf()),
            ("classifier", _classifier()),
        ]
    )


def _bigram() -> Pipeline:
    return Pipeline(
        [
            ("normalize", TurkishTextNormalizer()),
            ("features", TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=True)),
            ("classifier", _classifier()),
        ]
    )


def _char() -> Pipeline:
    return Pipeline(
        [
            ("normalize", TurkishTextNormalizer()),
            ("features", _char_tfidf()),
            ("classifier", _classifier()),
        ]
    )


def _hybrid() -> Pipeline:
    return Pipeline(
        [
            ("normalize", TurkishTextNormalizer()),
            (
                "features",
                FeatureUnion(
                    [
                        ("word", _word_tfidf(ngram_range=(1, 2))),
                        ("char", _char_tfidf()),
                    ]
                ),
            ),
            ("classifier", _classifier()),
        ]
    )


@dataclass(frozen=True)
class Architecture:
    """Kayıtlı bir model mimarisinin tanımı."""

    name: str
    description: str
    factory: Callable[[], Pipeline]


ARCHITECTURES: dict[str, Architecture] = {
    arch.name: arch
    for arch in (
        Architecture("bow", "Bag-of-Words (kelime frekansı) + Lojistik Regresyon", _bow),
        Architecture("tfidf", "Kelime TF-IDF + Lojistik Regresyon", _tfidf),
        Architecture("bigram", "Kelime ikilileri (bigram) TF-IDF + Lojistik Regresyon", _bigram),
        Architecture("char", "Karakter 2-5 gram TF-IDF + Lojistik Regresyon", _char),
        Architecture(
            "hybrid",
            "Kelime (1-2 gram) + karakter (2-5 gram) TF-IDF + Lojistik Regresyon",
            _hybrid,
        ),
    )
}


def build_pipeline(name: str = DEFAULT_ARCHITECTURE) -> Pipeline:
    """İsmi verilen mimari için eğitilmemiş yeni bir pipeline oluşturur."""
    try:
        return ARCHITECTURES[name].factory()
    except KeyError as exc:
        available = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"Bilinmeyen mimari: {name!r}. Seçenekler: {available}") from exc
