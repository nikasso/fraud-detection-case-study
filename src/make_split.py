# Making train-test-split

import pandas as pd
from sklearn.model_selection import train_test_split

def make_split(df):
    ''' split the data and prep for putting it into the model'''
    y = df['fraud'].values
    X = df.drop('fraud', 1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test
