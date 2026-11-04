"""Model mimarilerinin karşılaştırmalı ve tekrarlanabilir değerlendirmesi.

İki soruya cevap verir:

1. **Ayrılmış test kümesi:** Tüm mimariler aynı katmanlı %80 / %20 bölmede
   eğitilir ve ölçülür (doğruluk, makro F1, ilk-2 doğruluk, başlık performansı).
2. **Görülmemiş kaynak (leave-one-source-out):** Her veri kaynağı sırayla
   tamamen dışarıda bırakılır; model diğer kaynaklarla eğitilip hiç görmediği
   kaynağın haberlerinde test edilir. Bu, modelin *başka bir yayın organının*
   haberlerine ne kadar genellendiğinin dürüst ölçüsüdür.

Eğitim verisine başlık artırımı (bkz. ``training.augment_with_leads``) yalnızca
eğitim bölümüne uygulanır.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from haberlens.config import DEFAULT_DATA_PATH, RANDOM_STATE
from haberlens.data import LABEL, SOURCE, TEXT, load_dataset, split_dataset
from haberlens.evaluation import evaluate_model
from haberlens.evaluation import format_percent as _pct
from haberlens.pipelines import ARCHITECTURES, DEFAULT_ARCHITECTURE, MAX_CHARS, build_pipeline
from haberlens.preprocessing import TurkishTextNormalizer
from haberlens.training import DEFAULT_LEAD_WORDS, augment_with_leads

__all__ = [
    "BenchmarkReport",
    "BenchmarkResult",
    "CrossSourceResult",
    "run_benchmark",
    "write_report",
]

logger = logging.getLogger(__name__)

DISPLAY_NAMES = {
    "bow": "Bag-of-Words",
    "tfidf": "TF-IDF",
    "bigram": "N-Gram (bigram)",
    "char": "Karakter n-gram",
    "hybrid": "Hibrit (kelime + karakter) + Lojistik Regresyon",
    "svm": "Hibrit + Doğrusal SVM (kalibre)",
    "nb": "Hibrit + Complement Naive Bayes",
    "ensemble": "**HaberLens Ensemble** (LR + SVM + NB)",
    "word2vec": "Word2Vec (ortalama vektör)",
}
DEFAULT_CROSS_SOURCE_METHODS = ("hybrid", "svm", "nb", "ensemble")


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    display_name: str
    accuracy: float
    f1_macro: float
    top2_accuracy: float
    headline_f1: float | None
    fit_seconds: float


@dataclass(frozen=True)
class CrossSourceResult:
    method: str
    held_out_source: str
    n_test: int
    accuracy: float
    f1_macro: float


@dataclass
class BenchmarkReport:
    n_samples: int
    n_classes: int
    test_size: float
    lead_words: int
    sources: dict[str, int] = field(default_factory=dict)
    holdout: list[BenchmarkResult] = field(default_factory=list)
    cross_source: list[CrossSourceResult] = field(default_factory=list)


def _word2vec_pipeline() -> Pipeline:
    from haberlens.embeddings import Word2VecVectorizer

    return Pipeline(
        [
            ("normalize", TurkishTextNormalizer(max_chars=MAX_CHARS)),
            ("features", Word2VecVectorizer()),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )


def _factory(name: str) -> Pipeline:
    return _word2vec_pipeline() if name == "word2vec" else build_pipeline(name)


def run_benchmark(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    methods: list[str] | None = None,
    include_embeddings: bool = False,
    cross_source_methods: list[str] | tuple[str, ...] | None = DEFAULT_CROSS_SOURCE_METHODS,
    test_size: float = 0.2,
    lead_words: int = DEFAULT_LEAD_WORDS,
) -> BenchmarkReport:
    """Mimarileri karşılaştırır ve bir ``BenchmarkReport`` döndürür.

    ``cross_source_methods`` boş verilirse veya veride kaynak sütunu yoksa
    görülmemiş kaynak testi atlanır.
    """
    frame = load_dataset(data_path)
    labels = sorted(frame[LABEL].unique())
    split = split_dataset(frame, test_size=test_size, random_state=RANDOM_STATE)
    train_texts, train_labels = augment_with_leads(split.X_train, split.y_train, lead_words)

    selected = list(methods or ARCHITECTURES)
    if include_embeddings:
        try:
            import gensim  # noqa: F401
        except ImportError:  # pragma: no cover - isteğe bağlı bağımlılık
            logger.warning("Word2Vec için: pip install 'haberlens[embeddings]'")
        else:
            selected.append("word2vec")

    report = BenchmarkReport(
        n_samples=len(frame),
        n_classes=len(labels),
        test_size=test_size,
        lead_words=lead_words,
        sources=(
            {k: int(v) for k, v in frame[SOURCE].value_counts().items()}
            if SOURCE in frame.columns
            else {}
        ),
    )

    for name in selected:
        logger.info("Ayrılmış test kümesi: %s", name)
        started = time.perf_counter()
        model = _factory(name).fit(train_texts, train_labels)
        elapsed = time.perf_counter() - started
        metrics = evaluate_model(
            model, split.X_test, split.y_test, labels, lead_words=lead_words or DEFAULT_LEAD_WORDS
        )
        report.holdout.append(
            BenchmarkResult(
                method=name,
                display_name=DISPLAY_NAMES.get(name, name),
                accuracy=metrics["accuracy"],
                f1_macro=metrics["f1_macro"],
                top2_accuracy=metrics["top2_accuracy"],
                headline_f1=metrics.get("headline", {}).get("f1_macro"),
                fit_seconds=round(elapsed, 1),
            )
        )
    report.holdout.sort(key=lambda r: r.f1_macro, reverse=True)

    if cross_source_methods and SOURCE in frame.columns and frame[SOURCE].nunique() > 1:
        for held_out in sorted(frame[SOURCE].unique()):
            train = frame[frame[SOURCE] != held_out]
            test = frame[(frame[SOURCE] == held_out) & frame[LABEL].isin(set(train[LABEL]))]
            present = sorted(set(test[LABEL]))
            if len(present) < 2:
                continue
            texts, targets = augment_with_leads(train[TEXT], train[LABEL], lead_words)
            for name in cross_source_methods:
                logger.info("Görülmemiş kaynak: %s → %s", name, held_out)
                predicted = _factory(name).fit(texts, targets).predict(test[TEXT])
                report.cross_source.append(
                    CrossSourceResult(
                        method=name,
                        held_out_source=held_out,
                        n_test=len(test),
                        accuracy=round(float(accuracy_score(test[LABEL], predicted)), 4),
                        f1_macro=round(
                            float(
                                f1_score(test[LABEL], predicted, labels=present, average="macro")
                            ),
                            4,
                        ),
                    )
                )
    return report


def write_report(report: BenchmarkReport, output_dir: str | Path) -> Path:
    """Sonuçları ``benchmark.md`` ve ``benchmark.json`` olarak kaydeder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sources = ", ".join(f"{k} ({v})" for k, v in report.sources.items()) or "tek kaynak"

    lines = [
        "# Karşılaştırmalı Değerlendirme (Benchmark)",
        "",
        f"- **Veri:** {report.n_samples} benzersiz haber, {report.n_classes} kategori",
        f"- **Kaynaklar:** {sources}",
        f"- **Bölme:** katmanlı %{int((1 - report.test_size) * 100)} eğitim / "
        f"%{int(report.test_size * 100)} test, sabit tohum ({RANDOM_STATE})",
        f"- **Başlık artırımı:** eğitimde uzun haberlerin ilk {report.lead_words} kelimesi",
        f"- **Üretim:** `haberlens benchmark` · {generated}",
        "",
        "## 1. Ayrılmış test kümesi",
        "",
        "| Sıra | Yöntem | Doğruluk | Makro F1 | İlk-2 doğruluk | Başlık F1¹ | Eğitim (sn) |",
        "| :-: | --- | :-: | :-: | :-: | :-: | :-: |",
    ]
    for rank, r in enumerate(report.holdout, start=1):
        lines.append(
            f"| {rank} | {r.display_name} | {_pct(r.accuracy)} | {_pct(r.f1_macro)} | "
            f"{_pct(r.top2_accuracy)} | {_pct(r.headline_f1)} | "
            f"{str(r.fit_seconds).replace('.', ',')} |"
        )
    lines += [
        "",
        f"¹ Uzun test haberlerinin yalnızca ilk {report.lead_words} kelimesi verildiğinde "
        "(başlık benzeri kısa girdi) makro F1.",
        "",
    ]

    if report.cross_source:
        held_outs = sorted({r.held_out_source for r in report.cross_source})
        methods = list(dict.fromkeys(r.method for r in report.cross_source))
        table = {(r.method, r.held_out_source): r for r in report.cross_source}
        lines += [
            "## 2. Görülmemiş kaynak testi (leave-one-source-out)",
            "",
            "Her sütunda o kaynak eğitimden tamamen çıkarılmış ve model yalnızca diğer "
            "kaynaklarla eğitilmiştir. Değerler makro F1'dir (yalnızca o kaynakta bulunan "
            "kategoriler üzerinden).",
            "",
            "| Yöntem | " + " | ".join(f"→ {h}" for h in held_outs) + " | Ortalama |",
            "| --- | " + " | ".join(":-:" for _ in held_outs) + " | :-: |",
        ]
        for method in methods:
            cells = [table[(method, h)].f1_macro for h in held_outs if (method, h) in table]
            lines.append(
                f"| {DISPLAY_NAMES.get(method, method)} | "
                + " | ".join(
                    _pct(table[(method, h)].f1_macro) if (method, h) in table else "—"
                    for h in held_outs
                )
                + f" | {_pct(float(np.mean(cells)) if cells else None)} |"
            )
        lines.append("")

    lines += [
        f"> Varsayılan üretim mimarisi: `{DEFAULT_ARCHITECTURE}`.",
        "",
    ]
    markdown = output_dir / "benchmark.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "benchmark.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown
