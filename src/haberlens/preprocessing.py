"""Türkçeye özgü metin normalizasyonu.

Türkçe, büyük/küçük harf dönüşümünde (``I → ı``, ``İ → i``) ve özel isimlere
eklenen kesme işaretli çekim eklerinde (``Fed'in``, ``Türkiye'de``) standart
Python fonksiyonlarından farklı davranış gerektirir. Bu modül bu kuralları tek
bir yerde toplar; eğitim ve tahmin aşamaları aynı fonksiyonu kullandığı için
"training/serving skew" oluşmaz.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "TURKISH_STOPWORDS",
    "TurkishTextNormalizer",
    "normalize_text",
    "turkish_lower",
]

# Sık kullanılan ve sınıflandırmaya katkısı olmayan Türkçe işlev sözcükleri.
TURKISH_STOPWORDS: frozenset[str] = frozenset(
    """
    acaba ama ancak artık aslında az bana bazı belki ben beni benim beri bile bir birçok
    biri birkaç birşey biz bize bizi bizim bu buna bunda bundan bunu bunun burada böyle
    böylece da daha dahi de defa diye diğer eğer en gibi göre hangi hatta hem hep hepsi
    her herhangi herkes hiç hiçbir için ile ise işte kadar kendi kendine ki kim kimse
    mi mı mu mü nasıl ne neden nerede nereye niye niçin o olan olarak oldu olduğu olduğunu
    olmak olur on ona onda ondan onlar onları onların onu onun orada öyle sadece sanki
    siz size sizi sizin şey şeyi şimdi şu şuna şunda şundan şunu tüm tümü ve veya ya
    yani yine çok çünkü fakat lakin üzere zaten
    """.split()
)

_TURKISH_UPPER_MAP = str.maketrans({"I": "ı", "İ": "i"})
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Kesme işaretinden sonra gelen çekim ekini at: "Fed'in" -> "Fed", "Türkiye’de" -> "Türkiye"
_APOSTROPHE_SUFFIX_RE = re.compile(r"(\w)['’`´]\w*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_NON_WORD_RE = re.compile(r"[^\w\s]|_")
_WHITESPACE_RE = re.compile(r"\s+")


def turkish_lower(text: str) -> str:
    """Türkçe kurallarına uygun küçük harfe çevirme.

    >>> turkish_lower("İSTANBUL IŞIK")
    'istanbul ışık'
    """
    return text.translate(_TURKISH_UPPER_MAP).lower()


def normalize_text(
    text: object,
    *,
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
) -> str:
    """Ham haber metnini modele uygun, sade bir forma dönüştürür.

    Adımlar:
        1. Unicode NFC normalizasyonu ve HTML etiketlerinin temizlenmesi
        2. URL ve e-posta adreslerinin kaldırılması
        3. Kesme işaretli çekim eklerinin atılması (``Fed'in`` → ``fed``)
        4. Türkçe kurallarıyla küçük harfe çevirme
        5. Sayıların ve noktalama işaretlerinin kaldırılması
        6. (İsteğe bağlı) Türkçe etkisiz sözcüklerin ayıklanması

    >>> normalize_text("Fed'in faiz kararı 2024'te %5,25 oldu! https://ornek.com")
    'fed faiz kararı'
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    value = unicodedata.normalize("NFC", str(text))
    value = _HTML_TAG_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = _EMAIL_RE.sub(" ", value)
    value = _APOSTROPHE_SUFFIX_RE.sub(r"\1", value)
    value = turkish_lower(value)
    if remove_numbers:
        value = _NUMBER_RE.sub(" ", value)
    value = _NON_WORD_RE.sub(" ", value)

    tokens = _WHITESPACE_RE.split(value.strip())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in TURKISH_STOPWORDS]
    return " ".join(token for token in tokens if token)


class TurkishTextNormalizer(BaseEstimator, TransformerMixin):
    """``normalize_text`` fonksiyonunu scikit-learn pipeline adımı olarak sarmalar.

    Pipeline'ın içinde yer aldığı için kaydedilen model, kendi ön işleme
    mantığını da beraberinde taşır.
    """

    def __init__(self, remove_stopwords: bool = True, remove_numbers: bool = True) -> None:
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers

    def fit(self, X: Iterable[str], y: object = None) -> TurkishTextNormalizer:
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [
            normalize_text(
                text,
                remove_stopwords=self.remove_stopwords,
                remove_numbers=self.remove_numbers,
            )
            for text in X
        ]
