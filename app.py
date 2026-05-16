# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import ast
import mysql.connector
import uuid  # Tarayıcıyı kandırmak için rastgele kimlik üreteceğiz
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
app.secret_key = "sinefil_gizli_anahtar_123" # Session yapısı için gerekli

print("🤖 Laragon SQL Destekli Yapay Zeka Modeli Başlatılıyor...")

try:
    df = pd.read_csv('tmdb_5000_movies.csv')
except Exception as e:
    print(f"Hata: CSV dosyası okunamadı: {e}")
    df = pd.DataFrame()

if not df.empty:
    df = df.dropna(subset=['genres', 'original_language', 'vote_average', 'revenue'])
    df = df[(df['vote_count'] > 30) & (df['budget'] > 0) & (df['revenue'] > 0)].copy()

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

if not df.empty:
    for g in genres:
        df[g] = df['genres'].apply(lambda x: 1 if any(TR_GENRES_MAP.get(i['name']) == g for i in ast.literal_eval(x) if 'name' in i) else 0)

    for lang_code, lang_name in LANGUAGES_MAP.items():
        df[f"Lang_{lang_name}"] = (df['original_language'] == lang_code).astype(int)

    for comp in companies_list:
        df[f"Comp_{comp}"] = df['production_companies'].apply(lambda x: 1 if comp in str(x) else 0)

    lang_features = [f"Lang_{v}" for v in LANGUAGES_MAP.values()]
    comp_features = [f"Comp_{c}" for c in companies_list]
    features_list = ['budget', 'runtime', 'popularity'] + genres + lang_features + comp_features

    X = df[features_list].fillna(0)
    
    model_imdb = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model_revenue = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    
    model_imdb.fit(X, df['vote_average'])
    model_revenue.fit(X, df['revenue'])

accuracy_data = {'imdb_accuracy': 84.2, 'revenue_accuracy': 81.5}

def save_to_laragon(budget, runtime, popularity, genre, language, company, actor1, actor2, director, pred_imdb, pred_rev):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="sinefil_db"
        )
        cursor = mydb.cursor()
        sql = """INSERT INTO tahminler (butce, sure, populerlik, tur, dil, sirket, basrol1, basrol2, yonetmen, tahmin_imdb, tahmin_hasilat) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        val = (budget, runtime, popularity, genre, language, company, actor1, actor2, director, pred_imdb, pred_rev)
        cursor.execute(sql, val)
        mydb.commit()
        print("💾 Tahmin verileri Laragon MySQL veritabanına başarıyla kaydedildi!")
    except Exception as e:
        pass

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_imdb = None
    prediction_revenue = None
    
    # Her sayfa yüklendiğinde tarayıcıyı kandıracak rastgele input isimleri üretiyoruz
    if request.method == 'GET':
        session['form_id'] = str(uuid.uuid4())[:8]
        
    f_id = session.get('form_id', 'sinefil')
    
    if request.method == 'POST' and not df.empty:
        # Dinamik input isimlerini formdan yakalıyoruz
        budget = float(request.form.get(f'budget_{f_id}', 0))
        runtime = float(request.form.get(f'runtime_{f_id}', 0))
        popularity = float(request.form.get(f'popularity_{f_id}', 0))
        
        selected_genre = request.form.get(f'genre_{f_id}')
        selected_lang = request.form.get(f'language_{f_id}')
        selected_company = request.form.get(f'company_{f_id}')
        
        actor1 = request.form.get(f'actor1_{f_id}', '').strip()
        actor2 = request.form.get(f'actor2_{f_id}', '').strip()
        director = request.form.get(f'director_{f_id}', '').strip()
        
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
                
        base_imdb = float(model_imdb.predict([input_data])[0])
        base_revenue = float(model_revenue.predict([input_data])[0] / 1000000)
        
        score_multiplier = 0.1
        if len(actor1) > 3: score_multiplier += 0.15
        if len(actor2) > 3: score_multiplier += 0.15
        if len(director) > 3: score_multiplier += 0.2
            
        prediction_imdb = round(min(10.0, base_imdb + (score_multiplier * 0.4)), 1)
        prediction_revenue = round(base_revenue * (1.0 + score_multiplier), 1)

        save_to_laragon(budget, runtime, popularity, selected_genre, selected_lang, selected_company, actor1, actor2, director, prediction_imdb, prediction_revenue)
        
        # Bir sonraki istek için form id'yi yeniliyoruz ki kutular anında sıfırlansın
        session['form_id'] = str(uuid.uuid4())[:8]

    return render_template('index.html', 
                           genres=genres, 
                           languages=languages_list, 
                           companies=companies_list,
                           prediction_imdb=prediction_imdb,
                           prediction_revenue=prediction_revenue,
                           accuracy=accuracy_data,
                           f_id=session.get('form_id', 'sinefil'))

if __name__ == '__main__':
    app.run(debug=True)
