import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from haberlens import corpus
from haberlens.corpus import CorpusError, Source, build_corpus, clean_text


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remote_csv(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "remote.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return path


def _source(path: Path, *, name: str = "remote", required: bool = True, sha: str | None = None):
    return Source(
        name=name,
        title="Test kaynağı",
        url=path.resolve().as_uri(),
        sha256=sha or _sha(path),
        license="test",
        citation="test",
        label_map={"spor": "Spor", "ekonomi": "Ekonomi"},
        reader=corpus._read_kemik,
        filename=f"{name}.csv",
        required=required,
    )


@pytest.fixture
def local_raw(tmp_path: Path) -> Path:
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "HABERLER": [
                "Galatasaray derbiyi kazandı ve liderliğe yükseldi",
                "Borsa güne yükselişle başladı",
            ],
            "ETIKET": ["Spor", "Ekonomi"],
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("﻿RADİKAL - Faiz kararı açıklandı", "Faiz kararı açıklandı"),
        ("HABERTÜRK – Seçim sonuçları", "Seçim sonuçları"),
        ("Deprem oldu (DHA) bölgede hasar yok", "Deprem oldu bölgede hasar yok"),
        ("  Çok   boşluklu\n\nmetin ", "Çok boşluklu metin"),
        ("İSTANBUL - Şehir haberi", "İSTANBUL - Şehir haberi"),  # şehir künyesi korunur
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_build_corpus_maps_cleans_and_deduplicates(
    tmp_path: Path, local_raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _remote_csv(
        tmp_path,
        [
            ("RADİKAL - Fenerbahçe deplasmanda üç puan aldı", "spor"),
            ("Fenerbahçe deplasmanda üç puan aldı!", "spor"),  # noktalama farklı kopya
            ("Merkez bankası faiz oranını değiştirmedi", "ekonomi"),
            ("Çelişkili etiketli aynı haber metni burada", "spor"),
            ("Çelişkili etiketli aynı haber metni burada", "ekonomi"),
            ("Konu belirtmeyen genel bir haber metni", "genel"),
            ("kısa", "spor"),
        ],
    )
    monkeypatch.setattr(
        corpus, "SOURCES", {"haberlens": corpus.SOURCES["haberlens"], "remote": _source(remote)}
    )
    output = tmp_path / "out" / "corpus.csv"
    manifest = build_corpus(raw_path=local_raw, cache_dir=tmp_path / "cache", output_path=output)

    frame = pd.read_csv(output)
    assert list(frame.columns) == ["HABERLER", "ETIKET", "KAYNAK"]
    assert len(frame) == 4
    assert "Fenerbahçe deplasmanda üç puan aldı" in frame["HABERLER"].tolist()
    assert not frame["HABERLER"].str.contains("RADİKAL").any()
    assert not frame["HABERLER"].str.contains("Çelişkili").any()
    assert manifest["removed"]["conflicting_label_groups"] == 1
    assert manifest["by_source"] == {"haberlens": 2, "remote": 2}
    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["sha256"] == _sha(output)
    assert (tmp_path / "cache" / "remote.csv").is_file()


def test_checksum_mismatch_is_rejected(
    tmp_path: Path, local_raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _remote_csv(tmp_path, [("Fenerbahçe deplasmanda üç puan aldı", "spor")])
    monkeypatch.setattr(corpus, "SOURCES", {"remote": _source(remote, sha="0" * 64)})
    with pytest.raises(CorpusError, match="bütünlük"):
        build_corpus(raw_path=local_raw, cache_dir=tmp_path / "c", output_path=tmp_path / "o.csv")
    assert not list((tmp_path / "c").glob("*.part"))


def test_optional_source_failure_is_skipped(
    tmp_path: Path, local_raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("x")
    broken = _source(missing, name="broken", required=False, sha="0" * 64)
    monkeypatch.setattr(
        corpus, "SOURCES", {"haberlens": corpus.SOURCES["haberlens"], "broken": broken}
    )
    manifest = build_corpus(
        raw_path=local_raw, cache_dir=tmp_path / "c", output_path=tmp_path / "o.csv"
    )
    assert "broken" in manifest["skipped_sources"]
    assert manifest["n_samples"] == 2


def test_unknown_source_name(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="Bilinmeyen kaynak"):
        build_corpus(sources=["yok"], output_path=tmp_path / "o.csv")


def test_archive_reader_without_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus.shutil, "which", lambda _: None)
    with pytest.raises(CorpusError, match="libarchive"):
        corpus._read_ttc3600(tmp_path / "x.rar")


@pytest.mark.skipif(corpus._find_archive_tool() is None, reason="libarchive (bsdtar) kurulu değil")
def test_archive_reader_extracts_category_folders(tmp_path: Path) -> None:
    archive = tmp_path / "ttc.zip"  # libarchive aynı komutla zip ve rar açar
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("TTC/spor/c (1).txt", "﻿Maç berabere bitti")
        zf.writestr("TTC/ekonomi/c (1).txt", "Dolar yükseldi")
    frame = corpus._read_ttc3600(archive)
    assert sorted(frame["raw_label"]) == ["ekonomi", "spor"]
    assert "Maç berabere bitti" in frame["text"].tolist()


def test_registered_sources_are_pinned() -> None:
    for source in corpus.SOURCES.values():
        assert set(source.label_map.values()) <= set(corpus.CATEGORIES)
        if source.url:
            assert len(source.sha256 or "") == 64
            assert "/main/" not in source.url and "/master/" not in source.url
