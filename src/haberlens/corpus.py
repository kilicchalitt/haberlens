"""Çok kaynaklı eğitim derleminin (corpus) oluşturulması.

Tek bir kaynaktan eğitilen model, yalnızca o kaynağın yazım üslubunu öğrenir ve
başka sitelerin haberlerinde belirgin şekilde başarı kaybeder (bkz.
``reports/benchmark.md`` → çapraz kaynak testi). Bu modül; farklı yayın
organlarından, farklı yıllardan ve farklı uzunluklarda (başlık + tam metin)
haberleri tek bir tutarlı derlemde birleştirir:

1. Kaynak dosyaları sabitlenmiş commit adreslerinden indirir ve SHA-256 ile doğrular.
2. Her kaynağın etiketlerini ortak kategori kümesine eşler; konu belirtmeyen
   etiketleri (ör. *genel*, *güncel*) dışarıda bırakır.
3. Yayın organı ön eklerini ve ajans imzalarını temizler (kaynak sızıntısı).
4. Kaynaklar arası tekrarları ve çelişkili etiketli tekrarları ayıklar.
5. Derlemi ``data/processed/news_corpus.csv`` ve bir manifesto olarak yazar.

Üçüncü taraf veri setleri depoya kopyalanmaz; lisans ve atıf bilgileri
``data/README.md`` dosyasındadır.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import unicodedata
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from haberlens.config import DEFAULT_LABEL_COLUMN, DEFAULT_TEXT_COLUMN
from haberlens.preprocessing import normalize_text

__all__ = [
    "CATEGORIES",
    "SOURCES",
    "CorpusError",
    "Source",
    "build_corpus",
    "clean_text",
]

logger = logging.getLogger(__name__)

SOURCE_COLUMN = "KAYNAK"
DEFAULT_RAW_PATH = Path("data/raw/turkish_news.csv")
DEFAULT_CACHE_DIR = Path("data/external")
DEFAULT_OUTPUT_PATH = Path("data/processed/news_corpus.csv")

#: Ortak kategori kümesi. Tüm kaynak etiketleri bu adlara eşlenir.
CATEGORIES: tuple[str, ...] = (
    "Dünya",
    "Ekonomi",
    "Kültür-Sanat",
    "Magazin",
    "Sağlık",
    "Siyaset",
    "Spor",
    "Teknoloji",
    "Yaşam",
)

_OUTLETS = (
    r"(?:RADİKAL|HABERTÜRK|HÜRRİYET|POSTA|ZAMAN|SABAH|MİLLİYET"
    r"|İHA|IHA|DHA|AA|ANKA|CİHAN|NTV)"
)
_OUTLET_LEAD_RE = re.compile(rf"^\s*{_OUTLETS}\s*[-–—]\s*")
_AGENCY_TAG_RE = re.compile(r"\((?:DHA|İHA|IHA|AA|ANKA|Cihan|CİHAN|AP|Reuters|AFP)\)")
_WHITESPACE_RE = re.compile(r"\s+")
_DEDUP_KEY_CHARS = 300


class CorpusError(RuntimeError):
    """Derlem oluşturulamadığında fırlatılır."""


def clean_text(text: object) -> str:
    """Kaynağa özgü izleri temizler; anlamsal içeriğe dokunmaz.

    >>> clean_text("\\ufeffRADİKAL - Merkez Bankası  faizi (AA) sabit tuttu")
    'Merkez Bankası faizi sabit tuttu'
    """
    value = unicodedata.normalize("NFC", str(text)).replace("﻿", "")
    value = _WHITESPACE_RE.sub(" ", value).strip()
    value = _OUTLET_LEAD_RE.sub("", value)
    value = _AGENCY_TAG_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


# ----------------------------------------------------------------------------- kaynaklar
@dataclass(frozen=True)
class Source:
    """Derleme katılan bir veri kaynağının tanımı."""

    name: str
    title: str
    url: str | None
    sha256: str | None
    license: str
    citation: str
    label_map: dict[str, str]
    reader: Callable[[Path], pd.DataFrame]
    filename: str | None = None
    required: bool = True


def _read_haberlens(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    return pd.DataFrame({"text": frame["HABERLER"], "raw_label": frame["ETIKET"]})


def _read_kemik(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    return pd.DataFrame({"text": frame["text"], "raw_label": frame["label"]})


def _find_archive_tool() -> list[str] | None:
    """RAR açabilen bir libarchive (bsdtar) komutu bulur."""
    for candidate in ("bsdtar", "tar"):
        exe = shutil.which(candidate)
        if not exe:
            continue
        try:
            version = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, check=False, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - savunmacı
            continue
        if "bsdtar" in version or "libarchive" in version:
            return [exe, "-xf"]
    return None


def _read_ttc3600(path: Path) -> pd.DataFrame:
    tool = _find_archive_tool()
    if tool is None:
        raise CorpusError(
            "TTC-3600 bir RAR arşivi; açmak için libarchive gerekli "
            "(Debian/Ubuntu: `apt install libarchive-tools`, macOS ve Windows 10+: yerleşik `tar`)."
        )
    rows: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([*tool, str(path), "-C", tmp], check=True, capture_output=True)
        for file in sorted(Path(tmp).rglob("*.txt")):
            text = file.read_text(encoding="utf-8-sig", errors="replace")
            rows.append((text, file.parent.name))
    if not rows:
        raise CorpusError(f"{path.name} arşivinden metin çıkarılamadı.")
    return pd.DataFrame(rows, columns=["text", "raw_label"])


SOURCES: dict[str, Source] = {
    source.name: source
    for source in (
        Source(
            name="haberlens",
            title="HaberLens özgün veri seti (kısa haber metinleri ve başlıklar)",
            url=None,
            sha256=None,
            license="Araştırma ve eğitim amaçlı",
            citation="Halit Kılıç, HaberLens (2026).",
            label_map={label: label for label in CATEGORIES},
            reader=_read_haberlens,
        ),
        Source(
            name="kemik42k",
            title="YTÜ Kemik NLP Grubu — 42.000 Türkçe haber metni",
            url=(
                "https://raw.githubusercontent.com/furkankarakuz/TurkishNewsClassification/"
                "5dade345d3767daf3e27b73ecf96a44862ccf20a/data/turkish_news_13class_dataset.csv"
            ),
            sha256="2a610a144b09b735ea20b08eaec748c62a0fdc275b47e16853ddd18fc4672071",
            license="Akademik araştırma amaçlı (YTÜ Kemik NLP Grubu)",
            citation=(
                "Yıldız Teknik Üniversitesi Kemik Doğal Dil İşleme Grubu, 42 bin haber derlemi."
            ),
            label_map={
                "dunya": "Dünya",
                "ekonomi": "Ekonomi",
                "kultur-sanat": "Kültür-Sanat",
                "magazin": "Magazin",
                "saglik": "Sağlık",
                "siyaset": "Siyaset",
                "spor": "Spor",
                "teknoloji": "Teknoloji",
                "yasam": "Yaşam",
                # "genel", "guncel", "turkiye": konu değil, yayın bölümü → dışarıda bırakılır
            },
            reader=_read_kemik,
            filename="kemik42k.csv",
        ),
        Source(
            name="ttc3600",
            title="TTC-3600: Türkçe metin sınıflandırma kıyas veri seti",
            url=(
                "https://raw.githubusercontent.com/denopas/TTC-3600/"
                "e72a02dc79403a754b2964f81ca1ab43bfbd36e6/TTC-3600_Orj.rar"
            ),
            sha256="378dfe445aea9697fabcbeb0892f9eb49407fb27d6df38929d6bf1f00f458994",
            license="CC BY 4.0 (UCI Machine Learning Repository)",
            citation=(
                "Kılınç, D. vd. (2017). TTC-3600: A new benchmark dataset for Turkish text "
                "categorization. Journal of Information Science, 43(2), 174-185."
            ),
            label_map={
                "ekonomi": "Ekonomi",
                "kultursanat": "Kültür-Sanat",
                "saglik": "Sağlık",
                "siyaset": "Siyaset",
                "spor": "Spor",
                "teknoloji": "Teknoloji",
            },
            reader=_read_ttc3600,
            filename="ttc3600.rar",
            required=False,
        ),
    )
}


# ----------------------------------------------------------------------------- indirme
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(source: Source, cache_dir: Path) -> Path:
    """Kaynak dosyasını önbellekten döndürür; yoksa indirip bütünlüğünü doğrular."""
    assert source.url and source.filename and source.sha256
    target = cache_dir / source.filename
    if target.is_file() and _sha256(target) == source.sha256:
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    logger.info("İndiriliyor: %s", source.title)
    try:
        request = urllib.request.Request(source.url, headers={"User-Agent": "haberlens"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise CorpusError(f"{source.name} indirilemedi: {exc}") from exc

    actual = _sha256(partial)
    if actual != source.sha256:
        partial.unlink(missing_ok=True)
        raise CorpusError(
            f"{source.name} bütünlük doğrulaması başarısız (beklenen {source.sha256[:12]}…, "
            f"alınan {actual[:12]}…). Kaynak değişmiş olabilir."
        )
    partial.replace(target)
    return target


def _load_source(source: Source, raw_path: Path, cache_dir: Path) -> pd.DataFrame:
    path = raw_path if source.url is None else _fetch(source, cache_dir)
    frame = source.reader(path)
    frame["raw_label"] = frame["raw_label"].astype(str).str.strip()
    mapped = frame["raw_label"].map(source.label_map)
    dropped = frame.loc[mapped.isna(), "raw_label"].value_counts()
    if not dropped.empty:
        logger.info(
            "%s: eşlenmeyen etiketler dışarıda bırakıldı: %s", source.name, dropped.to_dict()
        )
    frame = frame.assign(label=mapped).dropna(subset=["label"])
    return pd.DataFrame({"text": frame["text"], "label": frame["label"], "source": source.name})


# ----------------------------------------------------------------------------- derleme
def build_corpus(
    *,
    sources: list[str] | None = None,
    raw_path: str | Path = DEFAULT_RAW_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_chars: int = 20,
) -> dict[str, object]:
    """Seçilen kaynaklardan derlemi oluşturur ve manifestoyu döndürür.

    İsteğe bağlı (``required=False``) bir kaynak yüklenemezse uyarı verilerek
    atlanır; zorunlu bir kaynak yüklenemezse ``CorpusError`` fırlatılır.
    """
    names = sources or list(SOURCES)
    unknown = sorted(set(names) - set(SOURCES))
    if unknown:
        raise CorpusError(f"Bilinmeyen kaynak(lar): {unknown}. Seçenekler: {sorted(SOURCES)}")

    raw_path, cache_dir, output_path = Path(raw_path), Path(cache_dir), Path(output_path)
    frames, used, skipped = [], [], {}
    for name in names:
        source = SOURCES[name]
        try:
            frames.append(_load_source(source, raw_path, cache_dir))
            used.append(name)
        except (CorpusError, OSError, subprocess.CalledProcessError) as exc:
            if source.required:
                raise CorpusError(f"Zorunlu kaynak yüklenemedi ({name}): {exc}") from exc
            logger.warning("İsteğe bağlı kaynak atlandı (%s): %s", name, exc)
            skipped[name] = str(exc)

    corpus = pd.concat(frames, ignore_index=True)
    initial = len(corpus)
    corpus["text"] = corpus["text"].map(clean_text)
    corpus = corpus[corpus["text"].str.len() >= min_chars]

    # Tekrar tespiti: normalize edilmiş metnin başı anahtar olarak kullanılır; böylece
    # yalnızca noktalama veya büyük/küçük harf farkı olan kopyalar da yakalanır.
    keys = corpus["text"].map(lambda t: normalize_text(t)[:_DEDUP_KEY_CHARS])
    corpus = corpus.assign(_key=keys)[keys.str.len() > 0]
    conflicting = corpus.groupby("_key")["label"].nunique()
    conflicting_keys = set(conflicting[conflicting > 1].index)
    corpus = corpus[~corpus["_key"].isin(conflicting_keys)]
    corpus = corpus.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)

    output = corpus.rename(
        columns={
            "text": DEFAULT_TEXT_COLUMN,
            "label": DEFAULT_LABEL_COLUMN,
            "source": SOURCE_COLUMN,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")

    manifest: dict[str, object] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(output_path),
        "sha256": _sha256(output_path),
        "n_samples": len(output),
        "removed": {
            "short_duplicate_or_conflicting": initial - len(output),
            "conflicting_label_groups": len(conflicting_keys),
        },
        "categories": {k: int(v) for k, v in output[DEFAULT_LABEL_COLUMN].value_counts().items()},
        "by_source": {k: int(v) for k, v in output[SOURCE_COLUMN].value_counts().items()},
        "sources": [
            {
                "name": SOURCES[name].name,
                "title": SOURCES[name].title,
                "url": SOURCES[name].url,
                "sha256": SOURCES[name].sha256,
                "license": SOURCES[name].license,
                "citation": SOURCES[name].citation,
            }
            for name in used
        ],
        "skipped_sources": skipped,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Derlem yazıldı: %s (%d kayıt)", output_path, len(output))
    return manifest
