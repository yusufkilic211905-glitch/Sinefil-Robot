# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:45:36 2026

@author: yusuf
DÜZELTILMIŞ - Tüm hatalar giderildi
"""
import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.ensemble import RandomForestRegressor

# 1. VERİLERİ YÜKLE
try:
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
except FileNotFoundError:
    print("Hata: CSV dosyaları bulunamadı! Lütfen klasörü kontrol edin.")
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

# 3. MODEL EĞİTİMİ
features = ['budget', 'runtime', 'popularity'] + all_genres
X = df[features].fillna(0)

model_imdb = RandomForestRegressor(n_estimators=100, random_state=42)
model_revenue = RandomForestRegressor(n_estimators=100, random_state=42)

model_imdb.fit(X, df['vote_average'])
model_revenue.fit(X, df['revenue'])

# 4. KAYDET
joblib.dump(model_imdb, 'model_imdb.pkl')
joblib.dump(model_revenue, 'model_revenue.pkl')
joblib.dump(all_genres, 'genres.pkl')
joblib.dump(features, 'features_list.pkl')

print("✅ Model dosyaları (.pkl) başarıyla oluşturuldu!")







