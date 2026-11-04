# Veri Seti

| Özellik | Değer |
| --- | --- |
| Dosya | `raw/turkish_news.csv` |
| Kodlama | UTF-8, virgülle ayrılmış (CSV) |
| Kayıt sayısı | 4.200 (85 tekrar eden metin eğitimde otomatik ayıklanır) |
| Sınıf sayısı | 7 — her sınıfta 600 örnek (dengeli) |
| Ortalama uzunluk | ~107 karakter (19 – 319 aralığı) |

## Şema

| Sütun | Tip | Açıklama |
| --- | --- | --- |
| `HABERLER` | `str` | Haber başlığı veya kısa haber metni |
| `ETIKET` | `str` | Kategori etiketi |

## Kategoriler

`Ekonomi` · `Magazin` · `Sağlık` · `Siyaset` · `Spor` · `Teknoloji` · `Yaşam`

## Kendi verinizle eğitim

HaberLens, aynı iki sütunlu yapıya sahip herhangi bir CSV ile yeniden eğitilebilir.
Sütun adları farklıysa komut satırından belirtilebilir:

```bash
haberlens train --data kurum_arsivi.csv --text-column baslik --label-column kategori
```

## Kullanım notu

Veri seti yalnızca araştırma ve eğitim amaçlı paylaşılmıştır. Haber metinlerinin
telif hakları ilgili yayın kuruluşlarına aittir.
