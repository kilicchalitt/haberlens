# Değişiklik Günlüğü

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenir.
Biçim [Keep a Changelog](https://keepachangelog.com/tr-TR/1.1.0/) standardına,
sürümleme [Semantic Versioning](https://semver.org/lang/tr/) kurallarına uyar.

## [1.1.0] — 2026-11-04

Odak: modelin **eğitimde görmediği yayın organlarının haberlerine** de uygulanabilmesi.
Ölçümler ve kararlar: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

### Eklendi

- `haberlens data build`: üç kaynaklı (haberlens, YTÜ Kemik 42K, TTC-3600) eğitim derlemini
  sabitlenmiş adreslerden indirir, SHA-256 ile doğrular, etiketleri ortak şemaya eşler,
  kaynak izlerini ve tekrarları temizler, manifesto yazar
- İki yeni kategori: **Dünya** ve **Kültür-Sanat** (toplam 9)
- Yeni mimariler: `svm` (kalibre Doğrusal SVM), `nb` (Complement Naive Bayes) ve varsayılan
  `ensemble` (LR + SVM + NB ağırlıklı yumuşak oylama)
- Başlık artırımı: uzun eğitim haberlerinin ilk 12 kelimesi ek örnek olarak eklenir
  (yalnızca eğitim bölümüne); görülmemiş kaynağın başlıklarında makro F1 +4,4 puan
- Yeni metrikler: ilk-2 doğruluk, kaynak bazlı skorlar, başlık benzeri kısa girdi performansı
- Benchmark'ta görülmemiş kaynak (leave-one-source-out) testi
- `TurkishTextNormalizer(max_chars=…)`: uzun metinlerde başlık + giriş bölümünün kullanılması
- Web arayüzüne Dünya ve Kültür-Sanat örnekleri; README'ye LinkedIn bağlantısı

### Değişti

- Varsayılan eğitim verisi `data/processed/news_corpus.csv` (önce `haberlens data build`)
- Varsayılan model `hybrid` yerine `ensemble`; model dosyası ~4,5 MB'tan ~96 MB'a büyüdü
- Çapraz doğrulama büyük derlemde varsayılan olarak kapalı (`--cv-folds` ile açılır) ve artık
  artırımı her katmanın içinde uygular (sızıntısız)
- `haberlens evaluate` makro F1'i yalnızca verideki kategoriler üzerinden hesaplar
- Docker imajı derlemi derleme sırasında oluşturur; CI duman testi `HABERLENS_CORPUS=raw`
  ile hızlı modda çalışır
- Tüm yüzdeler Türkçe ondalık virgülüyle yazılır

### Düzeltildi

- Kaynak bazlı makro F1, o kaynakta bulunmayan kategorileri de ortalamaya katarak skorları
  yapay olarak düşürüyordu
- Başka araçlarla yanlış küçültülmüş "İ" harfleri (`i̇`) kelimeleri ikiye bölüyordu

### Kaldırıldı

- Doğrulanamayan dondurulmuş BERT benchmark yolu ve `torch` / `transformers` isteğe bağlı
  bağımlılıkları (Word2Vec karşılaştırması korunuyor)

## [1.0.0] — 2026-11-04

Proje **HaberLens** adıyla yeniden yapılandırıldı ve üretime hazır hâle getirildi.

### Eklendi

- `haberlens` Python paketi (src düzeni, `pyproject.toml`, isteğe bağlı bağımlılık grupları)
- Türkçeye özgü metin normalizasyonu: `I/İ` dönüşümü, kesme işaretli ekler, Unicode noktalama,
  URL/e-posta/HTML temizliği, genişletilmiş etkisiz sözcük listesi
- Kelime + karakter n-gram TF-IDF birleşimi kullanan **hibrit model** (%98,4 makro F1)
- `NewsClassifier`: tekil/toplu tahmin, top-k olasılıklar, düşük güven (`uncertain`) bayrağı
  ve anahtar kelime açıklamaları
- Komut satırı aracı: `train`, `predict` (CSV/TXT toplu işlem), `evaluate`, `info`,
  `benchmark`, `serve`
- FastAPI REST servisi: tekil ve toplu sınıflandırma, model kartı, sağlık kontrolü,
  isteğe bağlı API anahtarı, CORS, istek sınırları, istek kimliği izleme
- Bağımlılıksız web arayüzü (açık/koyu tema, mobil uyumlu)
- Sürümlü model dosyası ve JSON model kartı (veri SHA-256, metrikler, kütüphane sürümleri)
- Tekrarlanabilir benchmark: BoW, TF-IDF, bigram, karakter n-gram, hibrit, Word2Vec,
  isteğe bağlı BERT
- Çok aşamalı Dockerfile, docker-compose, Makefile
- GitHub Actions CI (lint, Python 3.10-3.13 test matrisi, %90 kapsam eşiği, Docker duman testi),
  Dependabot, issue/PR şablonları
- Dokümantasyon: mimari, API referansı, model kartı, yol haritası, katkı rehberi,
  güvenlik politikası, atıf bilgisi (`CITATION.cff`)

### Değişti

- Veri setinin 7 kategorisinin tamamı kullanılıyor (önceden yalnızca 3)
- Veri seti `data/raw/turkish_news.csv` konumuna taşındı ve belgelendi
- Eğitim/test ayrımı artık katmanlı (stratified); metrikler makro F1 ve sınıf bazlı raporu içeriyor

### Düzeltildi

- **Veri sızıntısı:** vektörleştiriciler bölmeden önce tüm veriyle eğitiliyordu
- **Tekrar eden kayıtlar:** aynı metin hem eğitim hem test kümesine düşebiliyordu
- Türkçe tırnak ve kesme işaretleri (`“ ” ’`) noktalama temizliğinde atlanıyordu
- Her çalıştırmada tüm modellerin (BERT dahil) sıfırdan eğitilmesi gerekiyordu

### Kaldırıldı

- Tek dosyalık `main.py` betiği ve `input()` tabanlı etkileşimli demo
  (yerini `haberlens` CLI, REST API ve web arayüzü aldı)

## [0.1.0] — 2026-02-03

- İlk sürüm: Bag-of-Words, TF-IDF, N-Gram, Word2Vec ve BERT temsillerini 3 kategori
  (Ekonomi, Spor, Teknoloji) üzerinde karşılaştıran tek dosyalık betik

[1.1.0]: https://github.com/kilicchalitt/haberlens/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/kilicchalitt/haberlens/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/kilicchalitt/haberlens/releases/tag/v0.1.0
