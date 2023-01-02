import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
def pred_metrics(real, pred) -> None:
    print("Accuracy:\t{}".format(accuracy_score(real, pred)))
    print("Precision:\t{}".format(precision_score(real, pred)))
    print("Recall:\t\t{}".format(recall_score(real, pred)))
    print("F1:\t\t{}".format(f1_score(real, pred)))