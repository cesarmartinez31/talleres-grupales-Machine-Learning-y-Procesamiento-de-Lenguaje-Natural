#!/usr/bin/python
from flask import Flask, request
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
from flask_cors import CORS
from m09_model_deployment import predict_genre

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas y orígenes

api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='API para predecir los géneros de una película dada su sinopsis')

ns = api.namespace('predict', 
     description='Predicción de géneros de películas')

resource_fields = api.model('Resource', {
    'result': fields.Raw,
})

@ns.route('/')
class MovieGenrePrediction(Resource):

    @api.expect(api.parser().add_argument('plot', type=str, required=True, help='Sinopsis de la película', location='args'))
    @api.marshal_with(resource_fields)
    def get(self):
        plot = request.args.get('plot', '')
        
        # Predecir los géneros de la película utilizando la función del archivo m09_model_deployment.py
        genre_probabilities = predict_genre(plot)
        
        return {
            'result': genre_probabilities
        }, 200
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
