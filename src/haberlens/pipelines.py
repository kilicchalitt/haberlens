"""Model mimarileri (metin temsili + sınıflandırıcı) için fabrika fonksiyonları.

Her mimari, ön işlemeden sınıflandırıcıya kadar tek bir scikit-learn
``Pipeline`` nesnesidir. Böylece model tek dosya olarak kaydedilir, yüklenir ve
sürümlenir.

Varsayılan mimari ``ensemble``: kelime (1-2 gram) ve karakter (2-5 gram) TF-IDF
özniteliklerini ortak kullanan üç farklı sınıflandırıcının ağırlıklı olasılık
ortalamasıdır (yumuşak oylama):

* **Lojistik Regresyon** — iyi kalibre olasılıklar ve okunabilir açıklamalar
* **Doğrusal SVM** (sigmoid ile kalibre) — yüksek boyutlu seyrek metinde güçlü sınır
* **Complement Naive Bayes** — farklı tümevarımsal varsayım; dengesiz sınıflarda sağlam

Benchmark'ta topluluk, görülmemiş kaynaklarda Doğrusal SVM ile aynı düzeyde ve
Lojistik Regresyondan biraz iyidir; başlık benzeri kısa girdilerde ve ilk-2
doğrulukta en iyi sonucu verir (bkz. ``reports/benchmark.md``). Kaynağı kısıtlı
ortamlar için ``svm`` daha hafif bir seçenektir. Karakter n-gramları Türkçenin
sondan eklemeli yapısını (``ekonomi / ekonomik / ekonomide``) kök bulucuya
gerek kalmadan yakalar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from haberlens.config import RANDOM_STATE
from haberlens.preprocessing import TurkishTextNormalizer

__all__ = [
    "ARCHITECTURES",
    "DEFAULT_ARCHITECTURE",
    "ENSEMBLE_WEIGHTS",
    "MAX_CHARS",
    "Architecture",
    "build_pipeline",
]

DEFAULT_ARCHITECTURE = "ensemble"

#: Uzun haberlerde yalnızca başlık + giriş bölümü kullanılır (bkz. docs/EXPERIMENTS.md).
MAX_CHARS = 1500

#: Yumuşak oylama ağırlıkları: Lojistik Regresyon, SVM, Naive Bayes.
ENSEMBLE_WEIGHTS = (1.0, 1.0, 0.5)


# ------------------------------------------------------------------------- yapı taşları
def _normalizer() -> TurkishTextNormalizer:
    return TurkishTextNormalizer(max_chars=MAX_CHARS)


def _word_tfidf(ngram_range: tuple[int, int] = (1, 1), min_df: int = 1) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range, sublinear_tf=True, min_df=min_df, max_features=300_000
    )


def _char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3, max_features=300_000
    )


def _hybrid_features() -> FeatureUnion:
    return FeatureUnion([("word", _word_tfidf((1, 2), min_df=2)), ("char", _char_tfidf())])


def _logistic(c: float = 10.0) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE)


def _calibrated_svm() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        method="sigmoid",
        cv=3,
        ensemble=False,  # tek bir SVM + kalibratör: modeli küçük tutar
    )


def _naive_bayes() -> ComplementNB:
    return ComplementNB(alpha=0.1)


def _pipeline(features: object, classifier: object) -> Pipeline:
    return Pipeline(
        [("normalize", _normalizer()), ("features", features), ("classifier", classifier)]
    )


# --------------------------------------------------------------------------- mimariler
def _bow() -> Pipeline:
    return _pipeline(CountVectorizer(max_features=300_000), _logistic(c=1.0))


def _tfidf() -> Pipeline:
    return _pipeline(_word_tfidf(), _logistic())


def _bigram() -> Pipeline:
    return _pipeline(_word_tfidf((2, 2)), _logistic())


def _char() -> Pipeline:
    return _pipeline(_char_tfidf(), _logistic())


def _hybrid() -> Pipeline:
    return _pipeline(_hybrid_features(), _logistic())


def _svm() -> Pipeline:
    return _pipeline(_hybrid_features(), _calibrated_svm())


def _nb() -> Pipeline:
    return _pipeline(_hybrid_features(), _naive_bayes())


def _ensemble() -> Pipeline:
    voting = VotingClassifier(
        estimators=[("lr", _logistic()), ("svm", _calibrated_svm()), ("nb", _naive_bayes())],
        voting="soft",
        weights=list(ENSEMBLE_WEIGHTS),
    )
    return _pipeline(_hybrid_features(), voting)


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
        Architecture(
            "svm", "Kelime + karakter TF-IDF + Doğrusal SVM (sigmoid kalibrasyonlu)", _svm
        ),
        Architecture("nb", "Kelime + karakter TF-IDF + Complement Naive Bayes", _nb),
        Architecture(
            "ensemble",
            "Kelime + karakter TF-IDF + ağırlıklı topluluk "
            "(Lojistik Regresyon, Doğrusal SVM, Complement Naive Bayes)",
            _ensemble,
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
