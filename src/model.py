'''
Builds the model based on the training data
'''

import random
import cPickle as pickle
import pandas as pd
from feature_engineering import feature_engineering
from make_split import make_split
from oversampling import oversampling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV

def get_data(datafile):
    # Load in data
    df = pd.read_json(datafile)
    # Perform data & cleaning and feature engineering
    df = feature_engineering(df)
    # Perform over-sampling of majority class
    df = oversampling(df)
    # Make train test split
    X_train, X_test, y_train, y_test = make_split(df)
    return X_train, X_test, y_train, y_test

def modeling(X_train, y_train):
    '''
    Instanciates different models & fits them to training data
    '''
    logmodel1 = LogisticRegression()
    rfmodel1 = RandomForestClassifier()
    gbmodel1 = GradientBoostingClassifier()
    models = [logmodel1, rfmodel1, gbmodel1]
    model_preds = {} # dictionary that will be of models and their y_preds
    for model in models:
        model.fit(X_train, y_train)
        model_preds[model] = model.predict(X_test) # y_pred for each model
    return model_preds

def score(model, y_test, y_pred):
    f = f1_score(y_test, y_pred, average='macro')
    p = precision_score(y_test, y_pred, average='macro')
    r = recall_score(y_test, y_pred, average='macro')
    print "Scores for Model {}".format(model)
    print "----------"*3
    print "F1 Score: {}".format(f)
    print "Precision Score: {}".format(p)
    print "Recall Score: {}".format(r)
    print "=========="*3 + "\n"

def optimized_model():
    '''
    Determines best params for Random Forest model, returns optimal model
    '''
    rf_grid = {'n_estimators': [10, 50, 100],
                'max_depth': [20, 40, 60],
                'max_features': ['sqrt'],
                'random_state': [42],
                'oob_score': [False]}
    grid_search = GridSearchCV(RandomForestClassifier(), rf_grid)
    grid_search.fit(X_train, y_train)
    search_params = grid_search.best_params_
    search_score = grid_search.best_score_
    rfmodel_best = grid_search.best_estimator_
    return rfmodel_best, search_params, search_score

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data('files/data.json')
    # Scoring
    model_preds = modeling(X_train, y_train)
    for model in model_preds.keys():
        y_pred = model_preds[model]
        score(model, y_test, y_pred)
    # RandomForest has best score, select that model
    rfmodel = model_preds.keys()[2]

    # Optimized Random Forest model
    rfmodel_best, search_params, search_score = optimized_model()
    y_pred = rfmodel_best.predict(X_test)
    score(rfmodel_best, y_test, y_pred)

    # Pickle model we choose (Optimized Random Forest model)
    with open('model.pkl', 'w') as f:
        pickle.dump(rfmodel_best, f)
