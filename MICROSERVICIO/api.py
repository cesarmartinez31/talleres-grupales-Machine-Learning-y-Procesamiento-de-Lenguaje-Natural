#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
from m09_model_deployment import predict_price
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas y orígenes

api = Api(
    app, 
    version='1.0', 
    title='Used Car Price Prediction API',
    description='API to predict the price of used cars')

ns = api.namespace('predict', 
     description='Used Car Price Predictor')

parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Year of the car', 
    location='args')
parser.add_argument(
    'Mileage', 
    type=float, 
    required=True, 
    help='Mileage of the car', 
    location='args')
parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='State of the car', 
    location='args')
parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Make of the car', 
    location='args')
parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Model of the car', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

@ns.route('/')
class UsedCarPrice(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        year = args['Year']
        mileage = args['Mileage']
        state = args['State']
        make = args['Make']
        model = args['Model']
        
        # Predecir el precio del automóvil
        price = predict_price(year, mileage, state, make, model)
        
        return {
         "result": price
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
