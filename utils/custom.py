import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# load data
def load_data():
    # load
    train_data = pd.read_csv('data/train_data_fe.csv')
    test_data = pd.read_csv('data/test_data_fe.csv')
    
    # drop unneeded columns
    train_data.drop(['stone_soil', 'id', 'img_id'], axis=1, inplace = True)
    test_data.drop(['stone_soil', 'id', 'img_id'], axis=1, inplace = True)

    # prep data
    X_train = train_data.drop(['stone_soil_enc'], axis=1)
    y_train = train_data['stone_soil_enc']
    X_test = test_data.drop(['stone_soil_enc'], axis=1)
    y_test = test_data['stone_soil_enc']

    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)

    return X_train, y_train, X_test, y_test

# define a function for scoring
def pred_metrics(y_test, y_pred) -> None:
    print("Accuracy:\t{}".format(accuracy_score(y_test, y_pred)))
    print("Precision:\t{}".format(precision_score(y_test, y_pred)))
    print("Recall:\t\t{}".format(recall_score(y_test, y_pred)))
    print("F1:\t\t{}".format(f1_score(y_test, y_pred)))

def cm_plot(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt = 'g', cmap = 'Blues')

def f1_cv(model, X_train, y_train):
    score = cross_val_score(model, X_train, y_train, scoring = "f1")
    print(score.round(decimals=4))
    print(score.mean().round(decimals=4))
    return score

def cv_metrics(model, X_train, y_train):
    accuracy_score = cross_val_score(model, X_train, y_train, scoring = 'accuracy').mean().round(3)
    precision_score = cross_val_score(model, X_train, y_train, scoring = 'precision').mean().round(3)
    recall_score = cross_val_score(model, X_train, y_train, scoring = 'recall').mean().round(3)
    f1_score = cross_val_score(model, X_train, y_train, scoring = 'f1').mean().round(3)
    scores = pd.Series(data = [accuracy_score, precision_score, recall_score, f1_score]
                         , index = ['accuracy', 'precision', 'recall', 'f1'])
    return scores