#!/usr/bin/python
from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from m09_model_deployment import predict_genres

app = Flask(__name__)
CORS(app)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='API to predict movie genres'
)

ns = api.namespace('predict', description='Movie Genre Predictor')

parser = api.parser()
parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Plot of the movie', 
    location='args'
)

resource_fields = api.model('Resource', {
    'result': fields.List(fields.String)
})

@ns.route('/')
class MovieGenrePredictor(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']
        
        # Predecir los géneros de la película
        predicted_genres = predict_genres(plot)
        
        return {
            "result": predicted_genres
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

