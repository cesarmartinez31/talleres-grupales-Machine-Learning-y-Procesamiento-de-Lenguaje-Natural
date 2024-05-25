#!/usr/bin/python


import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Cargar el modelo y el vectorizador
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Inicializar la aplicación Flask
app = Flask(__name__)

# Definir la ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict_genre():
    # Obtener los datos de entrada del cuerpo de la solicitud POST
    data = request.json
    
    # Extraer la sinopsis de la película del cuerpo de la solicitud
    plot = data['plot']
    
    # Preprocesar el texto de la sinopsis
    clean_plot = preprocess_text(plot)
    
    # Vectorizar la sinopsis usando el vectorizador entrenado
    X_test_dtm = vectorizer.transform([clean_plot])
    
    # Realizar la predicción utilizando el modelo cargado
    predicted_probabilities = model.predict_proba(X_test_dtm)
    
    # Crear una lista de géneros y sus probabilidades predichas
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    predictions = {genre: proba for genre, proba in zip(genres, predicted_probabilities[0])}
    
    # Devolver las predicciones como respuesta en formato JSON
    return jsonify(predictions)

def preprocess_text(text):
    # Aquí iría tu función preprocess_text original para limpiar y procesar el texto de la sinopsis
    return text

if __name__ == '__main__':
    # Ejecutar la aplicación Flask en el puerto 5000
    app.run(port=5000, debug=True)