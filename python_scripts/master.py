# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:49:30 2021

@author: kanan
"""

from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import aggregate
import train
from tensorflow import keras
import datetime

app = Flask(__name__)
api = Api(app)

class Train(Resource):
    def post(self):
        data = request.get_json()
        start = datetime.datetime.now()
        model_path = train.train_model(data["clientNo"])
        end = datetime.datetime.now()
        train.convert_to_tflite(data["clientNo"])
        print(end-start)
        return model_path
class Aggregate(Resource):
    def post(self):
        start = datetime.datetime.now()
        weights = aggregate.fl_average(request.get_json()['model_directory'])
        model = aggregate.build_model(weights)
        model_path = aggregate.save_agg_model(model, request.get_json()['model_directory'])
        end = datetime.datetime.now()
        print(end-start)
        return model_path
class Demo(Resource):
    def post(self):
        data = request.get_json()
        print(data["clientNo"])
        return True
        #return jsonify({'data': data})
class Test(Resource):
    def post(self):
        model = keras.models.load_model(request.get_json()['model_path'])
        test_path = request.get_json()['test_images']
        accuracy = aggregate.test_Tfmodel(model, test_path)
        return accuracy
    
api.add_resource(Train, '/trainModel')  
api.add_resource(Aggregate, '/aggregateModel')
api.add_resource(Test, '/testModel')
api.add_resource(Demo, '/data')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # run our Flask app
