from pathlib import Path

import pandas as pd
import pytest

from haberlens.data import DatasetError, load_dataset, split_dataset, summarize


def _write(tmp_path: Path, rows: list[tuple[str, str]], columns=("HABERLER", "ETIKET")) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    return path


def test_load_dataset_renames_and_cleans(tmp_path: Path) -> None:
    rows = [(f"Spor haberi {i}", "Spor") for i in range(10)]
    rows += [(f"Ekonomi haberi {i}", "Ekonomi") for i in range(10)]
    rows += [("Spor haberi 0", "Spor"), ("   ", "Spor"), ("Etiketsiz", "")]
    frame = load_dataset(_write(tmp_path, rows), min_samples_per_class=5)

    assert list(frame.columns) == ["text", "label"]
    assert len(frame) == 20
    assert frame["text"].is_unique


def test_custom_column_names(tmp_path: Path) -> None:
    rows = [(f"metin {i}", "A" if i % 2 else "B") for i in range(20)]
    path = _write(tmp_path, rows, columns=("baslik", "kategori"))
    frame = load_dataset(path, text_column="baslik", label_column="kategori")
    assert set(frame["label"]) == {"A", "B"}


def test_label_filter(tmp_path: Path) -> None:
    rows = [(f"metin {i}", ["A", "B", "C"][i % 3]) for i in range(60)]
    frame = load_dataset(_write(tmp_path, rows), labels=["A", "C"])
    assert set(frame["label"]) == {"A", "C"}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="bulunamadı"):
        load_dataset(tmp_path / "yok.csv")


def test_missing_columns_raise(tmp_path: Path) -> None:
    path = _write(tmp_path, [("x", "y")], columns=("a", "b"))
    with pytest.raises(DatasetError, match="Eksik sütun"):
        load_dataset(path)


def test_unknown_label_filter_raises(tmp_path: Path) -> None:
    rows = [(f"metin {i}", "A" if i % 2 else "B") for i in range(40)]
    with pytest.raises(DatasetError, match="bulunmayan kategori"):
        load_dataset(_write(tmp_path, rows), labels=["Z"])


def test_too_few_samples_raise(tmp_path: Path) -> None:
    rows = [(f"metin {i}", "A") for i in range(20)] + [("tek", "B")]
    with pytest.raises(DatasetError, match="yeterli örnek yok"):
        load_dataset(_write(tmp_path, rows))


def test_single_class_raises(tmp_path: Path) -> None:
    rows = [(f"metin {i}", "A") for i in range(20)]
    with pytest.raises(DatasetError, match="en az iki"):
        load_dataset(_write(tmp_path, rows))


def test_split_is_stratified(sample_csv: Path) -> None:
    frame = load_dataset(sample_csv)
    split = split_dataset(frame, test_size=0.25)
    train_ratio = split.y_train.value_counts(normalize=True)
    test_ratio = split.y_test.value_counts(normalize=True)
    assert (train_ratio - test_ratio).abs().max() < 0.02
    assert len(split.X_test) == pytest.approx(len(frame) * 0.25, abs=1)


def test_summarize(sample_csv: Path) -> None:
    summary = summarize(load_dataset(sample_csv))
    assert summary["n_classes"] == 7
    assert sum(summary["class_distribution"].values()) == summary["n_samples"]
