# Model Kartı — HaberLens Hybrid v1.0

## Model özeti

| | |
| --- | --- |
| **Görev** | Tek etiketli Türkçe haber metni sınıflandırma |
| **Kategoriler** | Ekonomi, Magazin, Sağlık, Siyaset, Spor, Teknoloji, Yaşam |
| **Mimari** | Kelime (1-2 gram) + karakter (2-5 gram) TF-IDF → Lojistik Regresyon (C=10) |
| **Boyut** | ~4,5 MB (`.joblib`, sıkıştırılmış) |
| **Donanım** | Yalnızca CPU; tekil tahmin ~2 ms |
| **Lisans** | MIT |
| **Geliştiren** | [Halit Kılıç](https://github.com/kilicchalitt) |

## Amaçlanan kullanım

- Haber sitelerinde ve içerik yönetim sistemlerinde (CMS) **otomatik kategori önerisi**
- Medya takip ve basın bülteni ajanslarında gelen haberlerin **ön sınıflandırılması**
- Haber arşivlerinin **toplu etiketlenmesi** ve kategori bazlı raporlama
- NLP eğitimi ve Türkçe metin sınıflandırma araştırmaları için **referans uygulama**

Model, editörün yerini almak için değil, iş yükünü azaltmak için tasarlanmıştır.
Düşük güvenli tahminler (`uncertain: true`) insan onayına yönlendirilmelidir.

## Kapsam dışı kullanım

- Kişiler hakkında karar verme, içerik sansürü veya otomatik moderasyon
- Haber dışı metin türleri (sosyal medya gönderileri, yorumlar, resmi yazışmalar) —
  performans düşebilir
- Yedi kategori dışında kalan konular (ör. kültür-sanat, eğitim, dünya) —
  model bunları mevcut kategorilerden birine atar

## Eğitim verisi

- 4.200 kayıtlık dengeli Türkçe haber veri seti; boş ve tekrar eden kayıtlar
  ayıklandıktan sonra **4.109 benzersiz metin** (bkz. [`data/README.md`](../data/README.md))
- Metinler çoğunlukla haber başlığı ve kısa özetlerden oluşur (ortalama ~107 karakter)

## Değerlendirme

Ayrılmış, katmanlı %20 test kümesi (822 metin) üzerinde:

| Metrik | Değer |
| --- | --- |
| Doğruluk | **%98,54** |
| Makro F1 | **%98,53** |
| 5 katlı çapraz doğrulama (makro F1) | %98,53 ± 0,49 |

| Kategori | Kesinlik | Duyarlılık | F1 |
| --- | :-: | :-: | :-: |
| Ekonomi | 0,991 | 0,964 | 0,977 |
| Magazin | 0,968 | 0,992 | 0,979 |
| Sağlık | 0,984 | 1,000 | 0,992 |
| Siyaset | 0,992 | 0,992 | 0,992 |
| Spor | 1,000 | 0,983 | 0,992 |
| Teknoloji | 0,966 | 0,975 | 0,971 |
| Yaşam | 1,000 | 0,992 | 0,996 |

### Karışıklık matrisi (satır: gerçek, sütun: tahmin)

| | Eko | Mag | Sağ | Siy | Spo | Tek | Yaş |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Ekonomi** | **106** | 1 | 0 | 0 | 0 | 3 | 0 |
| **Magazin** | 0 | **119** | 0 | 0 | 0 | 1 | 0 |
| **Sağlık** | 0 | 0 | **119** | 0 | 0 | 0 | 0 |
| **Siyaset** | 0 | 0 | 1 | **117** | 0 | 0 | 0 |
| **Spor** | 1 | 0 | 0 | 1 | **117** | 0 | 0 |
| **Teknoloji** | 0 | 2 | 1 | 0 | 0 | **115** | 0 |
| **Yaşam** | 0 | 1 | 0 | 0 | 0 | 0 | **117** |

En sık karışan çift **Ekonomi ↔ Teknoloji**'dir (ör. teknoloji şirketlerinin finansal
haberleri). Tüm yöntemlerin karşılaştırması için bkz. [`reports/benchmark.md`](../reports/benchmark.md).

## Sınırlamalar ve riskler

- **Kategori tanımları veri setine bağlıdır.** Bu veri setinde *Yaşam* kategorisi ağırlıklı
  olarak asayiş, kaza ve afet haberlerinden oluşur; hava durumu veya gündelik yaşam haberleri
  farklı kategorilere atanabilir.
- **Zamansal kayma.** Veri belirli bir döneme aittir; yeni kişi, kurum ve olaylar
  (ör. yeni bir teknoloji ürünü) modelin kelime dağarcığında bulunmayabilir. Kurumların
  modeli kendi güncel arşivleriyle periyodik olarak yeniden eğitmesi önerilir.
- **Kısa metin varsayımı.** Model başlık ve kısa özetlerle eğitilmiştir; çok uzun
  metinlerde de çalışır ancak bu senaryo ayrıca doğrulanmamıştır.
- **Tek etiket.** Birden fazla kategoriye ait haberler (ör. ekonomi politikası) için
  yalnızca en olası kategori döndürülür; `scores` alanı ikinci olasılığı görmeyi sağlar.
- **Önyargı.** Kaynak haber sitelerinin editoryal tercihleri modele yansıyabilir.

## Yeniden üretim

```bash
haberlens train          # models/haberlens.joblib + models/haberlens.json
haberlens benchmark      # reports/benchmark.md
```

Tüm rastgelelik kaynakları sabit tohum (`42`) ile sabitlenmiştir. Model dosyasının
yanındaki JSON kartı; veri setinin SHA-256 özetini, eğitim zamanını ve kütüphane
sürümlerini içerir.
