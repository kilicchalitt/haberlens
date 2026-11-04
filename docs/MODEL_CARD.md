# Model Kartı — HaberLens Ensemble v1.1

## Model özeti

| | |
| --- | --- |
| **Görev** | Tek etiketli Türkçe haber sınıflandırma (başlık veya tam metin) |
| **Kategoriler** | Dünya, Ekonomi, Kültür-Sanat, Magazin, Sağlık, Siyaset, Spor, Teknoloji, Yaşam |
| **Mimari** | Kelime (1-2 gram) + karakter (2-5 gram) TF-IDF → Lojistik Regresyon, kalibre Doğrusal SVM ve Complement Naive Bayes'in ağırlıklı yumuşak oylaması (1 : 1 : 0,5) |
| **Girdi** | Metnin ilk 1.500 karakteri (başlık + giriş) |
| **Eğitim verisi** | 30.435 benzersiz haber, 3 kaynak (bkz. [`data/README.md`](../data/README.md)) |
| **Boyut** | ~96 MB (`.joblib`); yükleme ~3-4 sn; çalışma belleği işlem (worker) başına ~450 MB |
| **Hız (CPU)** | Tekil tahmin + açıklama ~8 ms; toplu işlemde saniyede ~7.000 metin |
| **Eğitim süresi** | ~13 dk (4 çekirdek, test değerlendirmesi + tüm veriyle yeniden eğitim) |
| **Lisans** | MIT |
| **Geliştiren** | [Halit Kılıç](https://www.linkedin.com/in/halitkilic/) |

## Amaçlanan kullanım

- Haber sitelerinde ve içerik yönetim sistemlerinde (CMS) **otomatik kategori önerisi**
- Medya takip ve basın bülteni ajanslarında gelen haberlerin **ön sınıflandırılması**
- Haber arşivlerinin **toplu etiketlenmesi** ve kategori bazlı raporlama
- NLP eğitimi ve Türkçe metin sınıflandırma araştırmaları için **referans uygulama**

Model editörün yerini almak için değil, iş yükünü azaltmak için tasarlanmıştır.
Düşük güvenli tahminler (`uncertain: true`, varsayılan eşik 0,5) insan onayına
yönlendirilmelidir. Doğru kategori %97,5 oranında modelin **ilk iki önerisi**
arasındadır; arayüzlerde ikinci öneriyi göstermek önerilir.

## Kapsam dışı kullanım

- Kişiler hakkında karar verme, içerik sansürü veya otomatik moderasyon
- Haber dışı metin türleri (sosyal medya gönderileri, yorumlar, resmi yazışmalar)
- Dokuz kategori dışındaki konular (ör. eğitim, otomotiv, din): model bunları mevcut
  kategorilerden birine atar; düşük güven bayrağı bu durumları kısmen yakalar

## Değerlendirme

Katmanlı (kategori × kaynak) %20 test bölmesi, 6.087 haber. Başlık artırımı yalnızca
eğitim bölümüne uygulanmıştır.

| Metrik | Değer |
| --- | :-: |
| Doğruluk | **%91,08** |
| Makro F1 | **%86,98** |
| Ağırlıklı F1 | %91,01 |
| İlk-2 doğruluk | **%97,54** |
| Başlık benzeri kısa girdi (uzun haberlerin ilk 12 kelimesi) — makro F1 | %79,80 |

### Kaynak bazında

| Kaynak | Test örneği | Doğruluk | Makro F1 |
| --- | :-: | :-: | :-: |
| haberlens (kısa metin / başlık) | 817 | %93,88 | %95,61 |
| kemik42k (tam metin) | 4.820 | %90,75 | %83,79 |
| ttc3600 (tam metin) | 450 | %89,56 | %90,23 |

Modelin **eğitimde hiç görmediği** bir yayın organındaki başarısı için
[`reports/benchmark.md`](../reports/benchmark.md) → *Görülmemiş kaynak testi* bölümüne bakın.

### Kategori bazında

| Kategori | Kesinlik | Duyarlılık | F1 | Test örneği |
| --- | :-: | :-: | :-: | :-: |
| Dünya | 0,827 | 0,864 | 0,845 | 730 |
| Ekonomi | 0,885 | 0,895 | 0,890 | 802 |
| Kültür-Sanat | 0,794 | 0,780 | 0,787 | 232 |
| Magazin | 0,880 | 0,894 | 0,887 | 625 |
| Sağlık | 0,901 | 0,948 | 0,924 | 442 |
| Siyaset | 0,904 | 0,877 | 0,891 | 570 |
| Spor | 0,985 | 0,991 | 0,988 | 2.118 |
| Teknoloji | 0,887 | 0,838 | 0,862 | 327 |
| Yaşam | 0,853 | 0,676 | 0,755 | 241 |

### Karışıklık matrisi (satır: gerçek, sütun: tahmin)

| | Dün | Eko | Kül | Mag | Sağ | Siy | Spo | Tek | Yaş |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Dünya** | **631** | 31 | 2 | 13 | 10 | 19 | 6 | 8 | 10 |
| **Ekonomi** | 25 | **718** | 2 | 4 | 8 | 16 | 4 | 20 | 5 |
| **Kültür-Sanat** | 6 | 2 | **181** | 34 | 3 | 1 | 1 | 2 | 2 |
| **Magazin** | 9 | 2 | 34 | **559** | 4 | 4 | 9 | 0 | 4 |
| **Sağlık** | 2 | 9 | 0 | 1 | **419** | 2 | 1 | 2 | 6 |
| **Siyaset** | 41 | 14 | 3 | 3 | 3 | **500** | 4 | 1 | 1 |
| **Spor** | 5 | 7 | 0 | 3 | 2 | 2 | **2.099** | 0 | 0 |
| **Teknoloji** | 19 | 18 | 2 | 1 | 9 | 2 | 2 | **274** | 0 |
| **Yaşam** | 25 | 10 | 4 | 17 | 7 | 7 | 6 | 2 | **163** |

Hataların büyük kısmı **anlamca örtüşen** kategori çiftlerindedir:

- **Siyaset ↔ Dünya:** Dış politika haberleri (ör. "AB liderleri göç politikasını görüştü")
- **Kültür-Sanat ↔ Magazin:** Sanatçı haberleri (ör. "ünlü oyuncu hayatını kaybetti")
- **Ekonomi ↔ Dünya / Teknoloji:** Uluslararası ekonomi ve teknoloji şirketi haberleri
- **Yaşam:** En heterojen kategori; asayiş haberlerinin yanında yaşam tarzı haberleri de içerir

Bu durumlar tek etiketli sınıflandırmanın doğal sınırıdır; `scores` alanındaki ikinci
öneri çoğunlukla diğer geçerli kategoridir.

## Sınırlamalar ve riskler

- **Kategori tanımları veri setlerine bağlıdır.** Yaşam kategorisi ağırlıklı olarak
  asayiş, kaza ve afet haberleridir; hava durumu veya gündelik yaşam haberleri farklı
  kategorilere düşebilir.
- **Zamansal kayma.** Eğitim verisi ağırlıklı olarak 2012-2015 dönemine aittir; yeni kişi,
  kurum, ürün ve olaylar modelin kelime dağarcığında bulunmayabilir. Karakter n-gramları
  bu etkiyi azaltır ama ortadan kaldırmaz. Kurumların modeli kendi güncel arşivleriyle
  periyodik olarak yeniden eğitmesi önerilir.
- **Dengesiz sınıflar.** Spor kategorisi derlemin üçte birini oluşturur; Kültür-Sanat ve
  Yaşam en az temsil edilen ve en düşük F1'e sahip kategorilerdir.
- **Çok kısa girdiler.** 3-4 kelimelik başlıklarda güven düşer (ör. "Derbi berabere bitti"
  doğru sınıflanır ancak *belirsiz* olarak işaretlenir).
- **Önyargı.** Kaynak yayın organlarının editoryal tercihleri modele yansıyabilir.

## Yeniden üretim

```bash
haberlens data build     # derlemi indirir ve SHA-256 ile doğrular
haberlens train          # models/haberlens.joblib + models/haberlens.json
haberlens benchmark      # reports/benchmark.md
```

Tüm rastgelelik kaynakları sabit tohum (`42`) ile sabitlenmiştir. Model dosyasının
yanındaki JSON kartı; derlemin SHA-256 özetini, kaynak dağılımını, eğitim zamanını ve
kütüphane sürümlerini içerir.
