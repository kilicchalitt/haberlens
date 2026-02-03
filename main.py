import pandas as pd
import re
import string
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import warnings

                    
 # 1. VERİ SETİNİ YÜKLÜYORUZ VE HAZIRLIYORUZ
             

warnings.filterwarnings("ignore")  # saçma uyarıları kapattık

df = pd.read_csv('VeriSetim.csv')

# Sütun isimlerini düzenliyor
df = df.rename(columns={'HABERLER': 'metin', 'ETIKET': 'sinif'})

# Hedef sınıfları seçtirme
hedef_siniflar = ['Ekonomi', 'Spor', 'Teknoloji']
df = df[df['sinif'].isin(hedef_siniflar)]

# Veri setinden bilgi çektik(Her sınıftan en fazla 550 örnek aldırdık)
df = df.groupby('sinif').head(550)

# Karıştırma ve index sıfırlama yapıyoruz
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna()

print(f"Veri seti başarıyla yüklendi. Toplam kayıt: {len(df)}")
print("-" * 30)










    # 2.METİN ÖN İŞLEME YAPIYORUZ (boşluk temizleme, noktalama kaldırma, tokenizasyon vb.)
                    
stop_words = set([
    've', 'ile', 'bir', 'bu', 'şu', 'o', 'için', 'diye', 'da', 'de', 
    'mi', 'mu', 'ama', 'fakat', 'lakin', 'ki', 'daha', 'en', 'çok', 'gibi','fazla'
])

def metin_temizle(metin):
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    metin = metin.translate(str.maketrans('', '', string.punctuation))
    metin = re.sub(r'\d+', '', metin)
    kelimeler = metin.split()
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
    return " ".join(temiz_kelimeler)

df['temiz_metin'] = df['metin'].apply(metin_temizle)














    # 3. METİN TEMSİLLERİ HAZIRLIYORUZ
                     
print("Öznitelik çıkarımı yapılıyor (Feature Extraction)...")

# 1-) Bag-of-Words
bow_vectorizer = CountVectorizer(max_features=2000)
X_bow = bow_vectorizer.fit_transform(df['temiz_metin']).toarray()



# 2-) TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(df['temiz_metin']).toarray()





# 3-) N-Grams (Bigram)
ngram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=2000)
X_ngram = ngram_vectorizer.fit_transform(df['temiz_metin']).toarray()




# 4-) Word Embeddings (Word2Vec)
tokenized_data = df['temiz_metin'].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=2, workers=4)

def get_average_word2vec(tokens, model, vector_size):
    vector_list = [model.wv[word] for word in tokens if word in model.wv]
    if len(vector_list) == 0:
        return np.zeros(vector_size)
    return np.mean(vector_list, axis=0)

X_w2v = np.array([get_average_word2vec(tokens, w2v_model, 100) for tokens in tokenized_data])



# 5-) Transformers (BERT)
print("BERT modeli işleniyor, lütfen bekleyin...")
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_bert_embeddings(text_list):
    embeddings = []
    # Tekil tahminlerde de çalışması için basit kontrol yapıyoruz
    is_list = isinstance(text_list, list) or isinstance(text_list, pd.Series)
    input_texts = text_list if is_list else [text_list]

    for i, text in enumerate(input_texts):
        # Sadece toplu işlemde (eğitimde) ilerleme çubuğunu gösteriyor
        if len(input_texts) > 1 and i % 250 == 0 and i > 0: 
            print(f"  -> İşlenen: {i}/{len(input_texts)}")
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding.flatten())
    
    return np.array(embeddings)

X_bert = get_bert_embeddings(df['temiz_metin'])













   # 4. MODEL EĞİTİMİ VE DEĞERLENDİRME
                          
print("\nModeller eğitiliyor ve test ediliyor...")
print("-" * 50)

le = LabelEncoder()
y = le.fit_transform(df['sinif'])

methods = [
    ("Bag-of-Words", X_bow),
    ("TF-IDF", X_tfidf),
    ("N-Grams", X_ngram),
    ("Word2Vec", X_w2v),
    ("BERT", X_bert)
]

trained_models = {}

def train_and_evaluate(X, y, method_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model: {method_name:<15} | Doğruluk: %{acc*100:.2f}")
    return clf

for name, X_matrix in methods:
    trained_models[name] = train_and_evaluate(X_matrix, y, name)

print("*" * 50)












                        
    # 5. FİNAL: GELİŞMİŞ DEMO (TÜM MODELLERLE TAHMİN EDİYORUZ)
                    
def tahmin_et_hepsi(metin):
    try:
        if not metin or len(metin.strip()) < 3:
            print("Lütfen geçerli bir metin giriniz.")
            return

        temiz = metin_temizle(metin)
        
        print("\n" + "="*65)
        print(f"ANALİZ EDİLEN METİN: {metin[:80]}...")
        print("="*65)
        print(f"{'YÖNTEM':<20} | {'TAHMİN':<15} | {'GÜVEN SKORU':<10}")
        print("-" * 65)

        for method_name, clf in trained_models.items():
            vector = None
            
            # Her model farklı veri formatı istediği için burada onu ayarlıyoruz:
            if method_name == "Bag-of-Words":
                vector = bow_vectorizer.transform([temiz]).toarray()
            elif method_name == "TF-IDF":
                vector = tfidf_vectorizer.transform([temiz]).toarray()
            elif method_name == "N-Grams":
                vector = ngram_vectorizer.transform([temiz]).toarray()
            elif method_name == "Word2Vec":
                tokens = temiz.split()
                # Ortalama vektör alıp boyutu (1, 100) yapıyoruz
                avg_vec = get_average_word2vec(tokens, w2v_model, 100)
                vector = avg_vec.reshape(1, -1)
            elif method_name == "BERT":
                vector = get_bert_embeddings([temiz]) # Fonksiyon zaten array döner

            # Tahmin işlemi yapyırtık burada
            if vector is not None:
                tahmin_index = clf.predict(vector)[0]
                tahmin_sinif = le.inverse_transform([tahmin_index])[0]
                
                # Olasılıkları alıyor
                probs = clf.predict_proba(vector)[0]
                guven_skoru = probs[tahmin_index]
                
                print(f"{method_name:<20} | {tahmin_sinif:<15} | %{guven_skoru*100:.2f}")

        print("-" * 65 + "\n")

    except Exception as e:
        print(f"Hata oluştu: {e}")

print(df.head(10))
print("\nSİSTEM HAZIR! (Çıkmak için 'q' tuşuna basıp enterlayın)")
print("Haber metnini yapıştırıp Enter'a basabilirsiniz.")

while True:
    kullanici_girdisi = input(">> Haber Metni: ")
    if kullanici_girdisi.lower() == 'q':
        print("Çıkış yapılıyor...")
        break
    tahmin_et_hepsi(kullanici_girdisi)
    