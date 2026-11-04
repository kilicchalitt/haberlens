"""Veri seti yükleme, doğrulama ve bölme işlemleri."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from haberlens.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_SOURCE_COLUMN,
    DEFAULT_TEXT_COLUMN,
    RANDOM_STATE,
)

__all__ = ["DatasetError", "DatasetSplit", "load_dataset", "split_dataset", "summarize"]

logger = logging.getLogger(__name__)

TEXT = "text"
LABEL = "label"
SOURCE = "source"


class DatasetError(ValueError):
    """Veri seti beklenen şemaya uymadığında fırlatılır."""


@dataclass(frozen=True)
class DatasetSplit:
    """Eğitim ve test bölümlerini bir arada tutar."""

    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    source_train: pd.Series | None = None
    source_test: pd.Series | None = None


def load_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    labels: list[str] | None = None,
    min_samples_per_class: int = 10,
    source_column: str | None = DEFAULT_SOURCE_COLUMN,
) -> pd.DataFrame:
    """CSV dosyasını okuyup temizlenmiş ``text`` / ``label`` sütunlu bir tablo döndürür.

    Args:
        path: CSV dosyasının yolu.
        text_column: Haber metnini içeren sütunun adı.
        label_column: Kategori etiketini içeren sütunun adı.
        labels: Verilirse yalnızca bu kategoriler kullanılır.
        min_samples_per_class: Bir kategorinin kabul edilmesi için gereken en az örnek.
        source_column: Kaydın hangi kaynaktan geldiğini belirten isteğe bağlı sütun.
            Dosyada varsa ``source`` olarak korunur ve kaynak bazlı değerlendirmede
            kullanılır; yoksa yok sayılır.

    Raises:
        DatasetError: Dosya yoksa, sütunlar eksikse veya veri yetersizse.
    """
    path = Path(path)
    if not path.is_file():
        hint = ""
        if path == Path(DEFAULT_DATA_PATH):
            hint = " Önce `haberlens data build` ile eğitim derlemini oluşturun."
        raise DatasetError(f"Veri dosyası bulunamadı: {path}.{hint}")

    frame = pd.read_csv(path, encoding="utf-8")
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise DatasetError(
            f"Eksik sütun(lar): {sorted(missing)}. Mevcut sütunlar: {list(frame.columns)}"
        )

    columns = {text_column: TEXT, label_column: LABEL}
    if source_column and source_column in frame.columns:
        columns[source_column] = SOURCE
    frame = frame[list(columns)].rename(columns=columns)
    frame[TEXT] = frame[TEXT].astype("string").str.strip()
    frame[LABEL] = frame[LABEL].astype("string").str.strip()
    if SOURCE in frame.columns:
        frame[SOURCE] = frame[SOURCE].fillna("bilinmiyor").astype("string")

    initial = len(frame)
    frame = frame.dropna()
    frame = frame[(frame[TEXT] != "") & (frame[LABEL] != "")]
    frame = frame.drop_duplicates(subset=TEXT, keep="first")
    removed = initial - len(frame)
    if removed:
        logger.info("%d boş veya tekrar eden kayıt ayıklandı.", removed)

    if labels is not None:
        unknown = set(labels) - set(frame[LABEL].unique())
        if unknown:
            raise DatasetError(f"Veri setinde bulunmayan kategori(ler): {sorted(unknown)}")
        frame = frame[frame[LABEL].isin(labels)]

    counts = frame[LABEL].value_counts()
    too_small = counts[counts < min_samples_per_class]
    if not too_small.empty:
        raise DatasetError(
            "Şu kategorilerde yeterli örnek yok "
            f"(en az {min_samples_per_class} gerekli): {too_small.to_dict()}"
        )
    if counts.size < 2:
        raise DatasetError("Sınıflandırma için en az iki farklı kategori gereklidir.")

    dtypes = dict.fromkeys(frame.columns, str)
    return frame.reset_index(drop=True).astype(dtypes)


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Sınıf oranlarını koruyarak (stratified) eğitim/test ayrımı yapar.

    Kaynak sütunu varsa katmanlama kategori × kaynak kombinasyonuna göre yapılır;
    böylece her kaynak test kümesinde eğitimdeki oranıyla temsil edilir.
    """
    strata = frame[LABEL]
    if SOURCE in frame.columns:
        combined = frame[LABEL] + "|" + frame[SOURCE]
        if combined.value_counts().min() >= 2:
            strata = combined
    train_idx, test_idx = train_test_split(
        frame.index, test_size=test_size, random_state=random_state, stratify=strata
    )
    train, test = frame.loc[train_idx], frame.loc[test_idx]
    has_source = SOURCE in frame.columns
    return DatasetSplit(
        train[TEXT],
        test[TEXT],
        train[LABEL],
        test[LABEL],
        train[SOURCE] if has_source else None,
        test[SOURCE] if has_source else None,
    )


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    """Veri seti hakkında raporlamaya uygun özet istatistikler üretir."""
    lengths = frame[TEXT].str.len()
    summary: dict[str, object] = {
        "n_samples": len(frame),
        "n_classes": int(frame[LABEL].nunique()),
        "class_distribution": {k: int(v) for k, v in frame[LABEL].value_counts().items()},
        "mean_text_length": round(float(lengths.mean()), 1),
        "max_text_length": int(lengths.max()),
    }
    if SOURCE in frame.columns:
        summary["source_distribution"] = {
            k: int(v) for k, v in frame[SOURCE].value_counts().items()
        }
    return summary
