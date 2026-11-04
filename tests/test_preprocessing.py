import math

import pytest

from haberlens.preprocessing import (
    TURKISH_STOPWORDS,
    TurkishTextNormalizer,
    normalize_text,
    turkish_lower,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("İSTANBUL", "istanbul"),
        ("IŞIK", "ışık"),
        ("Iğdır İli", "ığdır ili"),
        ("çağdaş ÖĞRENCİ", "çağdaş öğrenci"),
    ],
)
def test_turkish_lower_handles_dotted_and_dotless_i(raw: str, expected: str) -> None:
    assert turkish_lower(raw) == expected


def test_apostrophe_suffixes_are_removed() -> None:
    assert normalize_text("Fed'in kararı Türkiye’de konuşuldu") == "fed kararı türkiye konuşuldu"


def test_urls_emails_and_html_are_removed() -> None:
    text = "<p>Detaylar için https://ornek.com.tr adresine ya da info@ornek.com yazın</p>"
    assert normalize_text(text) == "detaylar adresine yazın"


def test_numbers_and_punctuation_are_removed_by_default() -> None:
    assert normalize_text("Enflasyon %64,77 oldu; dolar 32.50 TL!") == "enflasyon dolar tl"


def test_numbers_can_be_kept() -> None:
    assert normalize_text("Skor 3-1", remove_numbers=False) == "skor 3 1"


def test_stopwords_can_be_kept() -> None:
    assert normalize_text("ve bu bir test", remove_stopwords=False) == "ve bu bir test"
    assert normalize_text("ve bu bir test") == "test"


@pytest.mark.parametrize("value", [None, math.nan, "", "   ", "!!! ... ???"])
def test_empty_like_inputs_return_empty_string(value: object) -> None:
    assert normalize_text(value) == ""


def test_stopwords_are_lowercase() -> None:
    assert all(word == turkish_lower(word) for word in TURKISH_STOPWORDS)


def test_transformer_is_pipeline_compatible() -> None:
    normalizer = TurkishTextNormalizer()
    output = normalizer.fit_transform(["GALATASARAY'ın maçı", "Apple yeni iPhone'u tanıttı"])
    assert output == ["galatasaray maçı", "apple yeni iphone tanıttı"]
    assert normalizer.get_params() == {
        "remove_stopwords": True,
        "remove_numbers": True,
        "max_chars": None,
    }


def test_badly_lowercased_dotted_i_is_repaired() -> None:
    # "İçecek".lower() Python'da "i̇çecek" (i + U+0307) üretir; kelime bölünmemeli.
    assert normalize_text("İçecek sanayi".lower()) == "içecek sanayi"


def test_max_chars_truncates_before_normalization() -> None:
    normalizer = TurkishTextNormalizer(max_chars=14)
    assert normalizer.transform(["Ekonomi haberi ve devamı çok uzun"]) == ["ekonomi haberi"]
    assert normalizer.transform([None]) == [""]


def test_models_pickled_before_max_chars_still_work() -> None:
    legacy = TurkishTextNormalizer()
    del legacy.max_chars  # v1.0 nesnelerinde bu öznitelik yoktu
    assert legacy.transform(["Borsa yükseldi"]) == ["borsa yükseldi"]
