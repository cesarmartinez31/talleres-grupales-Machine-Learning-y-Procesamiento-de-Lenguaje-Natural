#!/usr/bin/python

import pandas as pd
import joblib

# Cargar el modelo y el vectorizador
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_rating(year, genre, director):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'Year': [year],
        'Genre': [genre],
        'Director': [director]
    })

    # Vectorizar las características categóricas
    input_data_encoded = vectorizer.transform(input_data)

    # Predecir la calificación de la película
    predicted_genres = model.predict(input_data_encoded)  # Obtener la lista de géneros predichos
    
    return predicted_genres.tolist()  # Convertir a lista de Python
