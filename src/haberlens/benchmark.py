"""Metin temsili yöntemlerinin karşılaştırmalı değerlendirmesi.

Tüm yöntemler aynı veri, aynı katmanlı (stratified) k-katlı çapraz doğrulama
bölmeleri ve aynı sınıflandırıcı ailesi (Lojistik Regresyon) ile ölçülür;
böylece fark yalnızca metin temsilinden kaynaklanır.

İsteğe bağlı Word2Vec ve BERT yöntemleri ``haberlens[embeddings]`` kuruluysa
çalıştırılır.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from haberlens.config import DEFAULT_DATA_PATH, RANDOM_STATE
from haberlens.data import LABEL, TEXT, load_dataset
from haberlens.pipelines import ARCHITECTURES, DEFAULT_ARCHITECTURE, build_pipeline
from haberlens.preprocessing import TurkishTextNormalizer

__all__ = ["BenchmarkResult", "run_benchmark", "write_report"]

logger = logging.getLogger(__name__)

DISPLAY_NAMES = {
    "bow": "Bag-of-Words",
    "tfidf": "TF-IDF",
    "bigram": "N-Gram (bigram)",
    "char": "Karakter n-gram",
    "hybrid": "**HaberLens Hybrid** (kelime + karakter)",
    "word2vec": "Word2Vec (ortalama)",
    "bert": "BERT (dbmdz/bert-base-turkish-cased, dondurulmuş)",
}


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    display_name: str
    accuracy_mean: float
    f1_macro_mean: float
    f1_macro_std: float
    fit_seconds: float


def _linear(scale: bool = False) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = []
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)))
    return steps


def _score(
    name: str, estimator: object, X: object, y: object, cv: StratifiedKFold
) -> BenchmarkResult:
    logger.info("Değerlendiriliyor: %s", name)
    started = time.perf_counter()
    scores = cross_validate(estimator, X, y, cv=cv, scoring=("accuracy", "f1_macro"), n_jobs=1)
    elapsed = time.perf_counter() - started
    return BenchmarkResult(
        method=name,
        display_name=DISPLAY_NAMES.get(name, name),
        accuracy_mean=round(float(np.mean(scores["test_accuracy"])), 4),
        f1_macro_mean=round(float(np.mean(scores["test_f1_macro"])), 4),
        f1_macro_std=round(float(np.std(scores["test_f1_macro"])), 4),
        fit_seconds=round(elapsed, 1),
    )


def run_benchmark(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    folds: int = 5,
    methods: list[str] | None = None,
    include_embeddings: bool = False,
) -> list[BenchmarkResult]:
    """Seçilen yöntemleri çapraz doğrulama ile karşılaştırır."""
    frame = load_dataset(data_path)
    X, y = frame[TEXT].tolist(), frame[LABEL].to_numpy()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    selected = methods or list(ARCHITECTURES)
    results = [_score(name, build_pipeline(name), X, y, cv) for name in selected]

    if include_embeddings:
        try:
            from haberlens.embeddings import BertEmbedder, Word2VecVectorizer
        except ImportError:  # pragma: no cover
            logger.warning("Gömme yöntemleri için: pip install 'haberlens[embeddings]'")
        else:
            w2v = Pipeline(
                [
                    ("normalize", TurkishTextNormalizer()),
                    ("features", Word2VecVectorizer()),
                    *_linear(scale=True),
                ]
            )
            results.append(_score("word2vec", w2v, X, y, cv))

            # BERT ağırlıkları dondurulmuş olduğundan (fit edilmez) gömmeler bir kez
            # hesaplanabilir; bu, test verisinin eğitime sızması anlamına gelmez.
            started = time.perf_counter()
            embeddings = BertEmbedder().transform(X)
            encode_seconds = time.perf_counter() - started
            bert = _score("bert", Pipeline(_linear(scale=True)), embeddings, y, cv)
            results.append(
                BenchmarkResult(
                    **{**asdict(bert), "fit_seconds": round(bert.fit_seconds + encode_seconds, 1)}
                )
            )

    return sorted(results, key=lambda r: r.f1_macro_mean, reverse=True)


def _tr(value: float, digits: int = 2) -> str:
    """Sayıyı Türkçe ondalık ayırıcıyla biçimlendirir: 98.42 -> '98,42'."""
    return f"{value:.{digits}f}".replace(".", ",")


def write_report(
    results: list[BenchmarkResult],
    output_dir: str | Path,
    *,
    folds: int,
    n_samples: int,
) -> Path:
    """Sonuçları ``benchmark.md`` ve ``benchmark.json`` olarak kaydeder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "# Karşılaştırmalı Değerlendirme (Benchmark)",
        "",
        f"- **Veri:** {n_samples} benzersiz haber metni, 7 dengeli kategori",
        f"- **Yöntem:** {folds} katlı katmanlı çapraz doğrulama, sabit tohum ({RANDOM_STATE})",
        "- **Sınıflandırıcı:** Lojistik Regresyon (tüm yöntemlerde aynı)",
        f"- **Üretim:** `haberlens benchmark` · {generated}",
        "",
        "| Sıra | Yöntem | Doğruluk | Makro F1 | Süre (sn) |",
        "| :-: | --- | :-: | :-: | :-: |",
    ]
    for rank, result in enumerate(results, start=1):
        lines.append(
            f"| {rank} | {result.display_name} | %{_tr(result.accuracy_mean * 100)} | "
            f"%{_tr(result.f1_macro_mean * 100)} ± {_tr(result.f1_macro_std * 100)} | "
            f"{_tr(result.fit_seconds, 1)} |"
        )
    lines += [
        "",
        f"> Varsayılan üretim mimarisi: `{DEFAULT_ARCHITECTURE}`. Süre sütunu tüm katmanlar için "
        "toplam eğitim + tahmin süresidir (BERT için gömme hesaplama süresi dahildir).",
        "",
    ]
    markdown = output_dir / "benchmark.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "benchmark.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown
