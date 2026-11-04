"""Paylaşılan test fikstürleri.

Testlerin hızlı ve deterministik olması için gerçek veri setinden küçük ama
dengeli bir alt küme kullanılır.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "turkish_news.csv"
SAMPLES_PER_CLASS = 60


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    frame = pd.read_csv(DATA_PATH)
    sample = frame.groupby("ETIKET", group_keys=False).head(SAMPLES_PER_CLASS)
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    sample.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_model_path(sample_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from haberlens.training import train_model

    output = tmp_path_factory.mktemp("models") / "model.joblib"
    train_model(data_path=sample_csv, output_path=output, cv_folds=0)
    return output
