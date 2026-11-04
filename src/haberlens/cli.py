"""``haberlens`` komut satırı arayüzü.

Örnekler::

    haberlens train
    haberlens predict "Merkez Bankası faizi sabit tuttu"
    haberlens predict --file haberler.csv --text-column baslik --output sonuc.csv
    haberlens evaluate --data kurum_test.csv
    haberlens info
    haberlens serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from haberlens import __version__
from haberlens.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEXT_COLUMN,
)

logger = logging.getLogger("haberlens")

BATCH_CHUNK_SIZE = 1000


# --------------------------------------------------------------------------- yardımcılar
def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _chunks(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read_input_texts(path: Path, text_column: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, encoding="utf-8")
        if text_column not in frame.columns:
            raise SystemExit(
                f"Hata: '{text_column}' sütunu bulunamadı. Mevcut sütunlar: {list(frame.columns)}. "
                "--text-column ile doğru sütunu belirtin."
            )
        frame[text_column] = frame[text_column].fillna("").astype(str)
        return frame
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame({text_column: [line for line in lines if line]})


# ------------------------------------------------------------------------------ komutlar
def cmd_train(args: argparse.Namespace) -> int:
    from haberlens.evaluation import format_report
    from haberlens.training import train_model

    result = train_model(
        data_path=args.data,
        output_path=args.output,
        architecture=args.architecture,
        text_column=args.text_column,
        label_column=args.label_column,
        labels=args.labels,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        refit_full=not args.no_refit,
    )
    print("\n" + format_report(result.metrics))
    cv = result.metrics.get("cross_validation")
    if cv:
        print(
            f"\n{cv['folds']} katlı çapraz doğrulama (makro F1): "
            f"%{cv['f1_macro_mean'] * 100:.2f} ± {cv['f1_macro_std'] * 100:.2f}"
        )
    print(f"\nModel kaydedildi → {result.model_path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from haberlens.predictor import NewsClassifier

    classifier = NewsClassifier.load(args.model, confidence_threshold=args.threshold)

    if args.file:
        frame = _read_input_texts(Path(args.file), args.text_column)
        texts = frame[args.text_column].tolist()
        predictions = []
        for chunk in _chunks(texts, BATCH_CHUNK_SIZE):
            predictions.extend(classifier.predict_batch(chunk, top_k=args.top_k))
        frame["predicted_label"] = [p.label for p in predictions]
        frame["confidence"] = [p.confidence for p in predictions]
        frame["uncertain"] = [p.uncertain for p in predictions]
        if args.output:
            output = Path(args.output)
            if output.suffix.lower() == ".jsonl":
                frame.to_json(output, orient="records", lines=True, force_ascii=False)
            else:
                frame.to_csv(output, index=False, encoding="utf-8")
            n_uncertain = int(frame["uncertain"].sum())
            print(f"{len(frame)} metin sınıflandırıldı → {output}")
            print(f"İnsan onayı önerilen (düşük güvenli) kayıt: {n_uncertain}")
            print("\nKategori dağılımı:")
            print(frame["predicted_label"].value_counts(dropna=False).to_string())
        else:
            columns = [args.text_column, "predicted_label", "confidence", "uncertain"]
            print(frame[columns].to_csv(index=False), end="")
        return 0

    texts = args.texts or [line for line in sys.stdin.read().splitlines() if line.strip()]
    if not texts:
        print("Sınıflandırılacak metin verilmedi. `haberlens predict --help`", file=sys.stderr)
        return 2

    results = [classifier.predict(text, top_k=args.top_k) for text in texts]
    if args.json:
        payload = [{"text": t, **r.to_dict()} for t, r in zip(texts, results, strict=True)]
        print(json.dumps(payload if len(payload) > 1 else payload[0], ensure_ascii=False, indent=2))
        return 0

    for text, result in zip(texts, results, strict=True):
        preview = text if len(text) <= 80 else text[:77] + "..."
        print(f"\n📰 {preview}")
        if result.label is None:
            print("   ⚠️  Metin anlamlı içerik barındırmıyor.")
            continue
        flag = "  ⚠️  düşük güven — insan onayı önerilir" if result.uncertain else ""
        print(f"   → {result.label}  (%{result.confidence * 100:.1f}){flag}")
        for label, score in list(result.scores.items())[1:]:
            print(f"     {label:<10} %{score * 100:.1f}")
        if result.keywords:
            print("   Anahtar kelimeler: " + ", ".join(k.term for k in result.keywords))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from haberlens.data import LABEL, TEXT, load_dataset
    from haberlens.evaluation import evaluate_predictions, format_report
    from haberlens.predictor import NewsClassifier

    classifier = NewsClassifier.load(args.model)
    frame = load_dataset(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column,
        min_samples_per_class=1,
    )
    unknown = sorted(set(frame[LABEL]) - set(classifier.labels))
    if unknown:
        raise SystemExit(f"Hata: Model bu kategorileri tanımıyor: {unknown}")
    predictions = classifier.predict_labels(frame[TEXT].tolist())
    metrics = evaluate_predictions(frame[LABEL], predictions, classifier.labels)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(format_report(metrics))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from haberlens.predictor import NewsClassifier

    metadata = NewsClassifier.load(args.model).metadata
    if args.json:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    metrics = metadata.get("metrics", {})
    data = metadata.get("data", {})
    print(f"HaberLens model v{metadata.get('version')}")
    print(f"  Mimari        : {metadata.get('architecture_description')}")
    print(f"  Eğitim tarihi : {metadata.get('trained_at')}")
    print(f"  Kategoriler   : {', '.join(metadata.get('labels', []))}")
    print(f"  Veri          : {data.get('file')} ({data.get('n_samples')} kayıt)")
    if metrics:
        print(f"  Doğruluk      : %{metrics['accuracy'] * 100:.2f}")
        print(f"  Makro F1      : %{metrics['f1_macro'] * 100:.2f}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - isteğe bağlı bağımlılık
        print("API için ek bağımlılıklar gerekli: pip install 'haberlens[api]'", file=sys.stderr)
        return 1
    import os

    if args.model:
        os.environ["HABERLENS_MODEL_PATH"] = str(args.model)
    uvicorn.run(
        "haberlens.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )
    return 0


# ---------------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    from haberlens.pipelines import ARCHITECTURES, DEFAULT_ARCHITECTURE

    parser = argparse.ArgumentParser(
        prog="haberlens",
        description="HaberLens — Türkçe haber metinleri için açıklanabilir sınıflandırma.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="ayrıntılı günlük çıktısı")
    sub = parser.add_subparsers(dest="command", required=True, metavar="KOMUT")

    def add_model_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--model",
            type=Path,
            default=None,
            help="model dosyası (varsayılan: $HABERLENS_MODEL_PATH veya models/haberlens.joblib)",
        )

    def add_column_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="metin sütunu adı")
        p.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN, help="etiket sütunu adı")

    # train
    p_train = sub.add_parser("train", help="modeli eğit ve kaydet")
    p_train.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="eğitim CSV dosyası")
    p_train.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="çıktı dosyası")
    p_train.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=DEFAULT_ARCHITECTURE,
        help="model mimarisi",
    )
    add_column_args(p_train)
    p_train.add_argument("--labels", nargs="+", help="yalnızca bu kategorilerle eğit")
    p_train.add_argument("--test-size", type=float, default=0.2, help="test kümesi oranı")
    p_train.add_argument("--cv-folds", type=int, default=5, help="çapraz doğrulama (0: kapalı)")
    p_train.add_argument(
        "--no-refit", action="store_true", help="son modeli tüm veriyle yeniden eğitme"
    )
    p_train.set_defaults(func=cmd_train)

    # predict
    p_predict = sub.add_parser("predict", help="metin(ler)i sınıflandır")
    p_predict.add_argument("texts", nargs="*", help="metinler (boşsa stdin okunur)")
    p_predict.add_argument("--file", help="toplu tahmin için .csv veya satır başına bir metin .txt")
    p_predict.add_argument("--output", help="toplu tahmin çıktısı (.csv veya .jsonl)")
    p_predict.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="CSV metin sütunu")
    p_predict.add_argument("--top-k", type=int, default=3, help="gösterilecek kategori sayısı")
    p_predict.add_argument("--threshold", type=float, default=0.5, help="düşük güven eşiği")
    p_predict.add_argument("--json", action="store_true", help="JSON çıktı üret")
    add_model_arg(p_predict)
    p_predict.set_defaults(func=cmd_predict)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="kayıtlı modeli etiketli bir veriyle değerlendir")
    p_eval.add_argument("--data", type=Path, required=True, help="etiketli CSV dosyası")
    add_column_args(p_eval)
    p_eval.add_argument("--json", action="store_true", help="JSON çıktı üret")
    add_model_arg(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    # info
    p_info = sub.add_parser("info", help="model kartını göster")
    p_info.add_argument("--json", action="store_true", help="JSON çıktı üret")
    add_model_arg(p_info)
    p_info.set_defaults(func=cmd_info)

    # serve
    p_serve = sub.add_parser("serve", help="REST API ve web arayüzünü başlat")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--workers", type=int, default=1)
    p_serve.add_argument("--reload", action="store_true", help="geliştirme modu")
    add_model_arg(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    from haberlens.artifact import ModelLoadError
    from haberlens.data import DatasetError

    try:
        return int(args.func(args) or 0)
    except (ModelLoadError, DatasetError, FileNotFoundError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
