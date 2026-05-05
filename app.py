from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Modelleri Yükle
try:
    model_imdb = joblib.load('model_imdb.pkl')
    model_revenue = joblib.load('model_revenue.pkl')
    genres = joblib.load('genres.pkl')
    features_list = joblib.load('features_list.pkl')
except:
    print("Hata: Model dosyaları bulunamadı! Önce model_hazirla.py çalıştırılmalı.")

@app.route('/')
def home():
    return render_template('index.html', genres=genres)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        budget = float(request.form.get('budget', 0))
        runtime = float(request.form.get('runtime', 100))
        popularity = float(request.form.get('popularity', 50))
        selected_genre = request.form.get('genre')

        # Input Hazırla
        input_df = pd.DataFrame(np.zeros((1, len(features_list))), columns=features_list)
        input_df.at[0, 'budget'] = budget
        input_df.at[0, 'runtime'] = runtime
        input_df.at[0, 'popularity'] = popularity
        if selected_genre in features_list:
            input_df.at[0, selected_genre] = 1

        # Tahminler
        puan = model_imdb.predict(input_df)[0]
        hasılat = model_revenue.predict(input_df)[0]
        
        final_puan = round(min(9.8, puan + (popularity/1000)), 1)
        final_hasılat = f"{int(hasılat):,}"

        # Başarı Analizi
        if final_puan >= 8.0: opinion = "Dünya genelinde İLK 10'a girebilir."
        elif final_puan >= 7.0: opinion = "Dünya genelinde İLK 100'e girebilir."
        else: opinion = "Dünya genelinde İLK 1000 listesine girmesi bekleniyor."

        return render_template('index.html', 
                               genres=genres,
                               prediction_text=final_puan,
                               revenue_text=final_hasılat,
                               algo_opinion=opinion)
    except Exception as e:
        return f"Bir hata oluştu: {e}"

if __name__ == "__main__":
    app.run(debug=True)