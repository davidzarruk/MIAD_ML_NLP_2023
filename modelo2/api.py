#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app,
    version='1.0',
    title='Predicción del precio de carro usado',
    description='Predicción del precio de carro usado')

ns = api.namespace('predict',
     description='Valor del precio del carro a predecir')
   
parser = api.parser()

parser.add_argument(
    'Year', type=int, required=True, help='Year', location='args')
parser.add_argument(
    'Mileage', type=int, required=True, help='Mileage', location='args')
parser.add_argument(
    'State', type=str, required=True, help='State', location='args')
parser.add_argument(
    'Make', type=str, required=True, help='Make', location='args')
parser.add_argument(
    'Model', type=str, required=True, help='Model', location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

def predict_price(url):

    #clf = joblib.load('Price_Car_Grupo4.pkl') 
    clf = joblib.load('Price_Car_Grupo4.pkl')
    #a = url.split('-')
    url_ = pd.DataFrame(url).transpose()
    url_.columns=['Year', 'Mileage', 'State', 'Make', 'Model']
    url_[['Year', 'Mileage']]=url_[['Year', 'Mileage']].astype(float)
    enc = OrdinalEncoder()
    url_[['State','Make','Model']] = enc.fit_transform(url_[['State','Make','Model']])
    p1= clf.predict(url_)

    return p1

@ns.route('/')
class CarPrice(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        features = [args['Year'], args['Mileage'], args['State'], args['Make'], args['Model']]
       
        return {
            "result": predict_price(features)
        }, 200
 
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
