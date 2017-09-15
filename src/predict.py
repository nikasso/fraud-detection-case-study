'''
Reads in a single example from test_script_examples, vectorizes it, unpickles
the model, predicts the label, and outputs the label probability (print to
standard out is fine).
'''

import pandas as pd
import csv
from pymongo import MongoClient
import random
import cPickle as pickle
from model import modeling
from feature_engineering_test import feature_engineering



def open_get_examples(filename='data.json'):
    '''First steps, import data and returns a selection of example'''
    df = pd.read_json(filename).loc[:5]
    return feature_engineering(df)


def write_to_file(filename='test_script_examples.py'):
    '''Take selection of data and write the data to a smaller file.'''
    examples.to_json(filename)



def open_1_example(index, filename='test_script_examples.py'):
    '''Opens the subset of data and pulls one row at the index specified.
    Returns variables and label. Index must be no higher than 5, as the file has only 5 examples.'''
    df = pd.read_json(filename)
    example = df.loc[index]
    y = example[0]
    X = example[1:]
    return X, y

class FraudMongo(object):

    def __init__(self):
       self.client=MongoClient()

    def dataframeToMongo(self,df):
       self.dataframe=df
       self.collection=self.client.fraudDb.collection
       self.collection.insert_many(self.dataframe.to_dict(‘records’))


    def getDataframe(self):
       return self.dataframe

    def updateDataframe(self,newdf):
       self.dataframe=newdf
       self.dataframeToMongo(self.dataframe)

    def printDataframe(self):
       return self.dataframe.head()


if __name__ == '__main__':
    examples = open_get_examples()
    write_to_file()
    X, y = open_1_example(index=3)
    with open('model.pkl') as f:
        model = pickle.load(f)
    print model.predict_proba(X)
