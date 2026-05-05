# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ast
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# 1. VERİLERİ YÜKLE
try:
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
    print("Veriler başarıyla yüklendi...")
except FileNotFoundError:
    print("Hata: CSV dosyaları bulunamadı!")
    exit()

# 2. TEMİZLİK VE ÖZELLİK ÇIKARIMI
df = df[(df['vote_count'] > 50) & (df['revenue'] > 0)].copy()

TR_GENRES = {
    'Action': 'Aksiyon', 'Adventure': 'Macera', 'Fantasy': 'Fantastik', 
    'Science Fiction': 'Bilim Kurgu', 'Crime': 'Suç', 'Drama': 'Dram', 
    'Thriller': 'Gerilim', 'Animation': 'Animasyon', 'Family': 'Aile', 
    'Comedy': 'Komedi', 'Romance': 'Romantik'
}

all_genres = sorted(list(set(TR_GENRES.values())))

for g in all_genres:
    df[g] = df['genres'].apply(lambda x: 1 if any(TR_GENRES.get(i['name']) == g for i in ast.literal_eval(x)) else 0)

# 3. MODEL EĞİTİMİ (HAFİFLETİLMİŞ AYARLAR)
features = ['budget', 'runtime', 'popularity'] + all_genres
X = df[features].fillna(0)

# n_estimators=25 ve max_depth=8: Boyutu düşürürken doğruluğu korur
model_imdb = RandomForestRegressor(n_estimators=25, max_depth=8, min_samples_leaf=5, random_state=42)
model_revenue = RandomForestRegressor(n_estimators=25, max_depth=8, min_samples_leaf=5, random_state=42)

print("Modeller eğitiliyor (hafif sürüm)...")
model_imdb.fit(X, df['vote_average'])
model_revenue.fit(X, df['revenue'])

# 4. KAYDET (SIKIŞTIRMA AKTİF)
# compress=3: Dosyayı kaydederken zipler, yer tasarrufu sağlar[cite: 3, 4]
joblib.dump(model_imdb, 'model_imdb.pkl', compress=3)
joblib.dump(model_revenue, 'model_revenue.pkl', compress=3)
joblib.dump(all_genres, 'genres.pkl')
joblib.dump(features, 'features_list.pkl')

print("\n✅ Model dosyaları (.pkl) başarıyla oluşturuldu!")
size_imdb = os.path.getsize('model_imdb.pkl') / (1024 * 1024)
print(f"model_imdb.pkl boyutu: {size_imdb:.2f} MB")