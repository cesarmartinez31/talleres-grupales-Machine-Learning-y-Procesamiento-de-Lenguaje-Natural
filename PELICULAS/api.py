#!/usr/bin/python
from flask import Flask, request
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
from movie_model_deployment import predict_rating
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

api = Api(
    app, 
    version='1.0', 
    title='Movie Rating Prediction API',
    description='API to predict movie ratings')

ns = api.namespace('predict', 
    description='Movie Rating Predictor')

parser = api.parser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Release year of the movie', 
    location='args')
parser.add_argument(
    'Genre', 
    type=str, 
    required=True, 
    help='Genre of the movie', 
    location='args')
parser.add_argument(
    'Director', 
    type=str, 
    required=True, 
    help='Director of the movie', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

@ns.route('/')
class MovieRatingPredictor(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        year = args['Year']
        genre = args['Genre']
        director = args['Director']
        
        # Predecir la calificación de la película
        rating = predict_rating(year, genre, director)
        
        return {
            "result": rating
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
