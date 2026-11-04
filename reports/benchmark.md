# Karşılaştırmalı Değerlendirme (Benchmark)

- **Veri:** 4109 benzersiz haber metni, 7 dengeli kategori
- **Yöntem:** 5 katlı katmanlı çapraz doğrulama, sabit tohum (42)
- **Sınıflandırıcı:** Lojistik Regresyon (tüm yöntemlerde aynı)
- **Üretim:** `haberlens benchmark` · 2026-11-04

| Sıra | Yöntem | Doğruluk | Makro F1 | Süre (sn) |
| :-: | --- | :-: | :-: | :-: |
| 1 | **HaberLens Hybrid** (kelime + karakter) | %98,42 | %98,41 ± 0,32 | 35,8 |
| 2 | Karakter n-gram | %98,22 | %98,21 ± 0,32 | 31,8 |
| 3 | TF-IDF | %96,88 | %96,87 ± 0,45 | 16,5 |
| 4 | Bag-of-Words | %95,42 | %95,41 ± 0,43 | 8,9 |
| 5 | Word2Vec (ortalama) | %86,81 | %86,82 ± 0,79 | 39,9 |
| 6 | N-Gram (bigram) | %82,58 | %83,46 ± 1,19 | 24,2 |

> Varsayılan üretim mimarisi: `hybrid`. Süre sütunu tüm katmanlar için toplam eğitim + tahmin süresidir (BERT için gömme hesaplama süresi dahildir).

## Notlar

- **BERT** (`dbmdz/bert-base-turkish-cased`, dondurulmuş gömmeler) bu raporun üretildiği
  ortamda Hugging Face Hub'a erişim olmadığı için ölçülemedi. Kendi makinenizde
  `pip install -e ".[embeddings]"` ardından `haberlens benchmark --include-embeddings`
  komutuyla tabloya eklenir.
- **N-Gram (bigram)** yöntemi yalnızca kelime ikililerini kullandığı için kısa başlıklarda
  seyrek kalır; orijinal çalışmadaki yapılandırma karşılaştırma amacıyla korunmuştur.
- **Word2Vec** yalnızca bu veri setiyle (~4 bin kısa metin) eğitildiğinden anlamlı vektörler
  öğrenmek için yeterli veriye sahip değildir; büyük bir derlemle önceden eğitilmiş
  vektörlerle sonuç iyileşebilir.
