from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)
api = Api(
    app, 
    version='1.0', 
    title='Vehicle Price Prediction API',
    description='Vehicle Price Prediction API'
)

ns = api.namespace('predict', description='Vehicle Price Predictor')

parser = api.parser()
parser.add_argument(
    'Year', type=int, required=True, help='Vehicle Year', location='args')
parser.add_argument(
    'Mileage', type=int, required=True, help='Vehicle Mileage', location='args')
parser.add_argument(
    'State', type=str, required=True, help='Vehicle State', location='args')
parser.add_argument(
    'Make', type=str, required=True, help='Vehicle Make', location='args')
parser.add_argument(
    'Model', type=str, required=True, help='Vehicle Model', location='args')

resource_fields = api.model('Resource', {
    'predicted_price': fields.Float,
})

# Load your trained XGBoost model
xgb_model = joblib.load('your_xgboost_model_file.pkl')

# Load label encoders for categorical features if needed
# label_encoders = joblib.load('label_encoders.pkl')

@ns.route('/')
class VehiclePriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        # Prepare input features as a DataFrame
        input_data = pd.DataFrame({
            'Year': [args['Year']],
            'Mileage': [args['Mileage']],
            'State': [args['State']],
            'Make': [args['Make']],
            'Model': [args['Model']]
        })
        
        # Encode categorical features if needed
        # for i, encoder in enumerate(label_encoders):
        #     input_data.iloc[:, i] = encoder.transform(input_data.iloc[:, i])
        
        # Make prediction
        predicted_price = xgb_model.predict(input_data)[0]
        
        return {'predicted_price': predicted_price}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
