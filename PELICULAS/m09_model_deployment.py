#!/usr/bin/python

import joblib
import pandas as pd

# Cargar el modelo y el vectorizador
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Lista de géneros en el mismo orden que fueron usados para entrenar el modelo
GENRE_LABELS = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

def predict_genres(plot):
    # Usar el vectorizador directamente en el argumento plot sin procesarlo
    input_data_encoded = vectorizer.transform([plot])
    predicted_genres_encoded = model.predict(input_data_encoded)
    predicted_genres = [GENRE_LABELS[i] for i, value in enumerate(predicted_genres_encoded[0]) if value == 1]
    return predicted_genres
