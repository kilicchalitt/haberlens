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
    DEFAULT_TEXT_COLUMN,
    RANDOM_STATE,
)

__all__ = ["DatasetError", "DatasetSplit", "load_dataset", "split_dataset", "summarize"]

logger = logging.getLogger(__name__)

TEXT = "text"
LABEL = "label"


class DatasetError(ValueError):
    """Veri seti beklenen şemaya uymadığında fırlatılır."""


@dataclass(frozen=True)
class DatasetSplit:
    """Eğitim ve test bölümlerini bir arada tutar."""

    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    labels: list[str] | None = None,
    min_samples_per_class: int = 10,
) -> pd.DataFrame:
    """CSV dosyasını okuyup temizlenmiş ``text`` / ``label`` sütunlu bir tablo döndürür.

    Args:
        path: CSV dosyasının yolu.
        text_column: Haber metnini içeren sütunun adı.
        label_column: Kategori etiketini içeren sütunun adı.
        labels: Verilirse yalnızca bu kategoriler kullanılır.
        min_samples_per_class: Bir kategorinin kabul edilmesi için gereken en az örnek.

    Raises:
        DatasetError: Dosya yoksa, sütunlar eksikse veya veri yetersizse.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Veri dosyası bulunamadı: {path}")

    frame = pd.read_csv(path, encoding="utf-8")
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise DatasetError(
            f"Eksik sütun(lar): {sorted(missing)}. Mevcut sütunlar: {list(frame.columns)}"
        )

    frame = frame[[text_column, label_column]].rename(
        columns={text_column: TEXT, label_column: LABEL}
    )
    frame[TEXT] = frame[TEXT].astype("string").str.strip()
    frame[LABEL] = frame[LABEL].astype("string").str.strip()

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

    return frame.reset_index(drop=True).astype({TEXT: str, LABEL: str})


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Sınıf oranlarını koruyarak (stratified) eğitim/test ayrımı yapar."""
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        frame[TEXT],
        frame[LABEL],
        test_size=test_size,
        random_state=random_state,
        stratify=frame[LABEL],
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    """Veri seti hakkında raporlamaya uygun özet istatistikler üretir."""
    lengths = frame[TEXT].str.len()
    return {
        "n_samples": len(frame),
        "n_classes": int(frame[LABEL].nunique()),
        "class_distribution": {k: int(v) for k, v in frame[LABEL].value_counts().items()},
        "mean_text_length": round(float(lengths.mean()), 1),
        "max_text_length": int(lengths.max()),
    }
