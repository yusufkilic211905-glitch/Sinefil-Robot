# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)

print("🤖 Render üzerinde yapay zeka modeli sıfırdan eğitiliyor, lütfen bekleyin...")

# 1. VERİLERİ YÜKLE VE TEMİZLE
try:
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
except Exception as e:
    print(f"Hata: CSV dosyaları okunamadı: {e}")
    df = pd.DataFrame()

if not df.empty:
    # Bütçe ve geliri 0'dan büyük olan, mantıklı verileri filtreliyoruz
    df = df[(df['vote_count'] > 30) & (df['budget'] > 0) & (df['revenue'] > 0)].copy()

# 2. MANUEL LİSTELER VE ÖZELLİK ÇIKARIMI
languages_list = ['TR', 'EN', 'FR', 'İTA', 'PT', 'DEU']
LANGUAGES_MAP = {'en': 'EN', 'tr': 'TR', 'fr': 'FR', 'it': 'İTA', 'pt': 'PT', 'de': 'DEU'}

companies_list = [
    'Warner Bros. Pictures', 'Walt Disney Pictures', 'Universal Pictures', 
    'Paramount Pictures', 'Columbia Pictures', '20th Century Studios', 
    'Marvel Studios', 'DreamWorks Animation', 'A24', 'Lucasfilm'
]

genres = ['Aksiyon', 'Macera', 'Fantastik', 'Bilim Kurgu', 'Suç', 'Dram', 'Gerilim', 'Animasyon', 'Aile', 'Komedi', 'Romantik']
TR_GENRES_MAP = {
    'Action': 'Aksiyon', 'Adventure': 'Macera', 'Fantasy': 'Fantastik', 
    'Science Fiction': 'Bilim Kurgu', 'Crime': 'Suç', 'Drama': 'Dram', 
    'Thriller': 'Gerilim', 'Animation': 'Animasyon', 'Family': 'Aile', 
    'Comedy': 'Komedi', 'Romance': 'Romantik'
}

# Veri işleme (Modelin anlayacağı 1 ve 0'lara dönüştürme)
if not df.empty:
    for g in genres:
        df[g] = df['genres'].apply(lambda x: 1 if any(TR_GENRES_MAP.get(i['name']) == g for i in ast.literal_eval(x)) else 0)

    for lang_code, lang_name in LANGUAGES_MAP.items():
        df[f"Lang_{lang_name}"] = (df['original_language'] == lang_code).astype(int)

    for comp in companies_list:
        df[f"Comp_{comp}"] = df['production_companies'].apply(lambda x: 1 if comp in str(x) else 0)

    lang_features = [f"Lang_{v}" for v in LANGUAGES_MAP.values()]
    comp_features = [f"Comp_{c}" for c in companies_list]
    features_list = ['budget', 'runtime', 'popularity'] + genres + lang_features + comp_features

    X = df[features_list].fillna(0)
    
    # Hafifletilmiş Eğitim (Render'da saniyeler içinde açılması için)
    model_imdb = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=42)
    model_revenue = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=42)
    
    model_imdb.fit(X, df['vote_average'])
    model_revenue.fit(X, df['revenue'])

    # Doğruluk Oranları (R² Skoru)
    imdb_acc = round(max(0, r2_score(df['vote_average'], model_imdb.predict(X))) * 100, 1)
    revenue_acc = round(max(0, r2_score(df['revenue'], model_revenue.predict(X))) * 100, 1)
    accuracy_data = {'imdb_accuracy': imdb_acc, 'revenue_accuracy': revenue_acc}
    print("✅ Yapay zeka başarıyla eğitildi ve sistem hazır!")
else:
    accuracy_data = {'imdb_accuracy': 0, 'revenue_accuracy': 0}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_imdb = None
    prediction_revenue = None
    
    if request.method == 'POST' and not df.empty:
        # Formdan gelen veriler
        budget = float(request.form.get('budget', 0))
        runtime = float(request.form.get('runtime', 0))
        popularity = float(request.form.get('popularity', 0))
        
        selected_genre = request.form.get('genre')
        selected_lang = request.form.get('language')
        selected_company = request.form.get('company')
        
        # Yapay zekaya verilecek girdi satırını oluşturma
        input_data = []
        for feature in features_list:
            if feature == 'budget': input_data.append(budget)
            elif feature == 'runtime': input_data.append(runtime)
            elif feature == 'popularity': input_data.append(popularity)
            elif feature in genres: input_data.append(1 if feature == selected_genre else 0)
            elif feature.startswith('Lang_'):
                input_data.append(1 if feature.replace('Lang_', '') == selected_lang else 0)
            elif feature.startswith('Comp_'):
                input_data.append(1 if feature.replace('Comp_', '') == selected_company else 0)
            else: input_data.append(0)
                
        # Tahminleri hesapla
        prediction_imdb = round(float(model_imdb.predict([input_data])[0]), 1)
        prediction_revenue = round(float(model_revenue.predict([input_data])[0] / 1000000), 1)

    return render_template('index.html', 
                           genres=genres, 
                           languages=languages_list, 
                           companies=companies_list,
                           prediction_imdb=prediction_imdb,
                           prediction_revenue=prediction_revenue,
                           accuracy=accuracy_data)

if __name__ == '__main__':
    app.run(debug=True)
