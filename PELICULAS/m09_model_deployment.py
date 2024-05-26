#!/usr/bin/python

import pandas as pd
import joblib

# Cargar el modelo y el vectorizador
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Aquí debes definir una lista con los nombres de los géneros en el mismo orden que fueron usados para entrenar el modelo
GENRE_LABELS = ['Accion', 'Aventura', 'Comedia', 'Crimen', 'Documental', 'Drama', 'Familia', 'Fantasia', 'Historia', 'Horror', 'Musica', 'Misterio', 'Romance', 'Ciencia Ficción', 'Película de TV', 'Thriller', 'Guerra', 'Western']

def predict_genres(year, genre, director):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'Year': [year],
        'Genre': [genre],
        'Director': [director]
    })

    # Vectorizar las características categóricas
    input_data_encoded = vectorizer.transform(input_data)

    # Predecir los géneros de la película
    predicted_genres_encoded = model.predict(input_data_encoded)
    
    # Decodificar las predicciones
    predicted_genres = [GENRE_LABELS[i] for i, value in enumerate(predicted_genres_encoded[0]) if value == 1]
    
    return predicted_genres
