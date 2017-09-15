from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
from werkzeug import secure_filename
import cPickle as pickle
import pandas as pd
import os
import ast
from feature_engineering_test import feature_engineering
'''import feature_engineering_test'''
import requests
from MangoCLient import FraudMongo
from pymongo import MongoClient
import json
with open('model.pkl') as f:
    model = pickle.load(f)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt','pdf','csv','json'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
@app.route('/')
@app.route('/hello')
def index():
        if 'name' in request.args:
            return 'Hello ' + request.args['name']
        else:
            return 'Hello'
from flask import json

mongo=FraudMongo()
mongo.client.dbs.collec.remove({})

@app.route('/score')
def score():
    url = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"
    r = requests.get(url)
    first=r.json()
    if isinstance(first, dict):
        # print 'DICT'
        df=pd.DataFrame.from_dict(first, orient='index')
        df=df.transpose()
        orig_df = df.copy()
        featurizedDf=feature_engineering(df)
        # print featurizedDf.head()
        prediction= model.predict_proba(featurizedDf)[0][1]
        featurizedDf['probability_Fraud']=prediction
        orig_df['predictionFraud']=prediction
        mongo.insertOneToMongoDb(orig_df)
        return render_template('result.html',result=prediction,df=mongo.getCollectionAsDataframe().to_html())
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8093, debug=True)
