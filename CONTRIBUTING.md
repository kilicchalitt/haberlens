# Katkı Rehberi

HaberLens'e katkıda bulunmak istediğiniz için teşekkürler! 🎉

## Geliştirme ortamı

```bash
git clone https://github.com/kilicchalitt/haberlens.git
cd haberlens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make dev        # pip install -e ".[dev]"
make train      # modeli eğit
make test       # testleri çalıştır
```

## İş akışı

1. Bir issue açarak değişikliği tartışın (küçük düzeltmeler için gerekmez).
2. `main` dalından yeni bir dal oluşturun: `git checkout -b feat/kisa-aciklama`.
3. Değişikliğinizi testleriyle birlikte yazın.
4. `make lint test` komutlarının hatasız geçtiğinden emin olun.
5. Pull request açın ve şablondaki kontrol listesini doldurun.

## Commit mesajları

Proje [Conventional Commits](https://www.conventionalcommits.org/tr/) standardını kullanır:

| Önek | Kullanım |
| --- | --- |
| `feat:` | Yeni özellik |
| `fix:` | Hata düzeltmesi |
| `docs:` | Yalnızca dokümantasyon |
| `test:` | Test ekleme/düzenleme |
| `refactor:` | Davranışı değiştirmeyen kod düzenlemesi |
| `build:` / `ci:` | Paketleme, Docker, CI |
| `chore:` | Diğer bakım işleri |

Örnek: `feat(api): add rate limiting to batch endpoint`

## Kod standartları

- Biçimlendirme ve lint: [`ruff`](https://docs.astral.sh/ruff/) (`make format`)
- Tüm genel fonksiyonlar tip ipuçları ve docstring içermelidir.
- Kullanıcıya dönük mesajlar Türkçe, kod tanımlayıcıları İngilizcedir.
- Test kapsamı %90'ın altına düşmemelidir (CI bunu denetler).

## Yeni bir model mimarisi eklemek

1. `src/haberlens/pipelines.py` içinde pipeline'ı döndüren bir fabrika fonksiyonu yazın.
2. Fonksiyonu `ARCHITECTURES` sözlüğüne bir `Architecture` kaydı olarak ekleyin.
3. `haberlens benchmark` ile mevcut yöntemlerle karşılaştırın ve sonucu PR'a ekleyin.

`tests/test_pipelines.py` her kayıtlı mimariyi otomatik olarak test eder.
