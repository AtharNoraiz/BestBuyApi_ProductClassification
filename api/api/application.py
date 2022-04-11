from flask import Flask
from flask import request
import flask
from flask_expects_json import expects_json
from .schema_json import schema
from flask import jsonify
from .categoriesService import products


app = Flask(__name__)

@app.route("/")
def heart_beat():

    return "BestBuy classification model is at your service",200
@expects_json(schema)
@app.route("/predict", methods = ['POST'])
def prediction():
    try:
        data = request.get_json()
        predictions = products.get_categories(data)
        return jsonify(predictions),200
    except Exception as e:
        return "bad request",400
