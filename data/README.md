# Veri

HaberLens, farklı yayın organlarından, farklı yıllardan ve farklı uzunluklardaki
(başlık ve tam metin) Türkçe haberleri birleştiren **çok kaynaklı bir derlemle**
eğitilir. Tek kaynaklı eğitimin neden yetersiz kaldığı
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) belgesinde ölçümleriyle anlatılmıştır.

```bash
haberlens data build      # indir → doğrula → eşle → temizle → data/processed/news_corpus.csv
```

| Dizin | İçerik | Git'te |
| --- | --- | :-: |
| `raw/turkish_news.csv` | HaberLens'in özgün veri seti | ✅ |
| `external/` | İndirilen üçüncü taraf kaynaklar (önbellek) | ❌ |
| `processed/news_corpus.csv` | Birleştirilmiş eğitim derlemi + `.json` manifesto | ❌ |

Üçüncü taraf veri setleri depoya **kopyalanmaz**; sabitlenmiş commit adreslerinden
indirilir ve SHA-256 özetiyle doğrulanır. Bir kaynak değişirse derleme hata verir.

## Kaynaklar

| Kaynak | İçerik | Derlemdeki kayıt | Lisans |
| --- | --- | :-: | --- |
| **haberlens** | Kısa haber metinleri ve başlıklar, 7 kategori | 4.086 | Araştırma ve eğitim amaçlı |
| **kemik42k** | YTÜ Kemik NLP Grubu 42 bin haber derlemi, tam metin | 24.099 | Akademik araştırma amaçlı |
| **ttc3600** | TTC-3600 kıyas veri seti (2015, 6 haber sitesi), tam metin | 2.250 | CC BY 4.0 |
| **Toplam** | | **30.435** | |

TTC-3600 bir RAR arşivi olarak dağıtılır; açmak için libarchive gerekir
(Debian/Ubuntu: `apt install libarchive-tools`, macOS ve Windows 10+: yerleşik `tar`).
Araç yoksa derleme bu kaynağı atlayıp uyarı vererek devam eder.

## Ortak kategori şeması

| Kategori | haberlens | kemik42k | ttc3600 | Derlem |
| --- | :-: | :-: | :-: | :-: |
| Dünya | — | `dunya` | — | 3.650 |
| Ekonomi | `Ekonomi` | `ekonomi` | `ekonomi` | 4.008 |
| Kültür-Sanat | — | `kultur-sanat` | `kultursanat` | 1.159 |
| Magazin | `Magazin` | `magazin` | — | 3.127 |
| Sağlık | `Sağlık` | `saglik` | `saglik` | 2.208 |
| Siyaset | `Siyaset` | `siyaset` | `siyaset` | 2.854 |
| Spor | `Spor` | `spor` | `spor` | 10.589 |
| Teknoloji | `Teknoloji` | `teknoloji` | `teknoloji` | 1.634 |
| Yaşam | `Yaşam` | `yasam` | — | 1.206 |

Kemik derlemindeki `genel`, `guncel` ve `turkiye` etiketleri **kullanılmaz**: bunlar konu
değil yayın bölümü adlarıdır ve içlerinde spor, siyaset ve asayiş haberleri karışıktır.
**Yaşam** kategorisi ağırlıklı olarak asayiş, kaza ve afet haberlerinden oluşur.

## Temizleme adımları

1. Unicode NFC normalizasyonu, BOM ve fazla boşlukların temizlenmesi
2. Metin başındaki yayın organı adlarının (`RADİKAL -`, `HABERTÜRK -` …) ve parantez
   içindeki ajans imzalarının (`(DHA)`, `(AA)` …) silinmesi: model kaynağı değil konuyu
   öğrenmelidir
3. 20 karakterden kısa kayıtların atılması
4. Normalize edilmiş metnin ilk 300 karakterine göre tekrarların ayıklanması (kaynaklar
   arası dahil; TTC-3600'ün 1.254 metni Kemik derleminde de bulunur)
5. Aynı metnin farklı etiketlerle geçtiği kayıtların tamamen çıkarılması

## Şema

| Sütun | Açıklama |
| --- | --- |
| `HABERLER` | Haber başlığı veya metni |
| `ETIKET` | Ortak kategori adı |
| `KAYNAK` | Kaydın geldiği kaynak (`haberlens`, `kemik42k`, `ttc3600`) |

## Kendi verinizle eğitim

Aynı sütun yapısına sahip (KAYNAK isteğe bağlı) herhangi bir CSV ile eğitilebilir;
sütun adları farklıysa komut satırından belirtilir:

```bash
haberlens train --data kurum_arsivi.csv --text-column baslik --label-column kategori
```

## Atıf

- Kılınç, D., Özçift, A., Bozyiğit, F., Yıldırım, P., Yücalar, F., & Borandağ, E. (2017).
  *TTC-3600: A new benchmark dataset for Turkish text categorization.* Journal of
  Information Science, 43(2), 174–185.
- Yıldız Teknik Üniversitesi Kemik Doğal Dil İşleme Grubu, *42 bin haber* derlemi.

Haber metinlerinin telif hakları ilgili yayın kuruluşlarına aittir. Veriler yalnızca
araştırma ve eğitim amacıyla kullanılmaktadır.
