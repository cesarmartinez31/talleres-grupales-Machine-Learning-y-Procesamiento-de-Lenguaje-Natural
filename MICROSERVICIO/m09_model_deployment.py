#!/usr/bin/python

import pandas as pd
import joblib
import os

# Cargar el modelo y el codificador
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

def predict_price(year, mileage, state, make, model_name):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'Year': [year],
        'Mileage': [mileage],
        'State': [state],
        'Make': [make],
        'Model': [model_name]
    })

    # Codificar las características categóricas
    input_data_encoded = encoder.transform(input_data)

    # Predecir el precio del automóvil
    price = model.predict(input_data_encoded)[0]
    
    return price
        
