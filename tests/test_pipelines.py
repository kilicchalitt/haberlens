import pytest
from sklearn.pipeline import Pipeline

from haberlens.pipelines import ARCHITECTURES, DEFAULT_ARCHITECTURE, build_pipeline

TEXTS = [
    "Galatasaray derbide Fenerbahçe'yi 2-1 yendi",
    "Milli takım teknik direktörü kadroyu açıkladı",
    "Merkez Bankası politika faizini sabit tuttu",
    "Borsa İstanbul günü yükselişle tamamladı",
] * 3
LABELS = ["Spor", "Spor", "Ekonomi", "Ekonomi"] * 3


def test_default_architecture_is_registered() -> None:
    assert DEFAULT_ARCHITECTURE in ARCHITECTURES


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_every_architecture_trains_and_predicts(name: str) -> None:
    pipeline = build_pipeline(name)
    assert isinstance(pipeline, Pipeline)
    pipeline.fit(TEXTS, LABELS)
    probabilities = pipeline.predict_proba(["Faiz kararı piyasaları etkiledi"])
    assert probabilities.shape == (1, 2)
    assert probabilities.sum() == pytest.approx(1.0)


def test_build_pipeline_returns_fresh_instances() -> None:
    assert build_pipeline() is not build_pipeline()


def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen mimari"):
        build_pipeline("gpt-5")
