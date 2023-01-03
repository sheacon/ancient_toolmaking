import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# load data
def load_data(verbose=True):
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

    if verbose:
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

def cv_metrics2(model, X_train, y_train):
    shuffled_index = np.random.choice(X_train.shape[0], replace = False, size = X_train.shape[0])
    result = np.array_split(shuffled_index, 5)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for part in result:
        part_X_train = X_train.drop(part, axis='index')
        part_y_train = y_train.drop(part, axis='index')
        part_X_test = X_train.iloc[part]
        part_y_test = y_train.iloc[part]
        model.fit(part_X_train,part_y_train)
        part_y_pred = model.predict(part_X_test)
        tn, fp, fn, tp = confusion_matrix(part_y_test,part_y_pred).ravel()
        accuracy_score = (tp+tn)/(tp+tn+fp+fn)
        accuracy.append(accuracy_score)
        precision_score = tp/(tp+fp)
        precision.append(precision_score)
        recall_score = tp/(tp+fn)
        recall.append(recall_score)
        f1_score = 2*(precision_score*recall_score)/(precision_score+recall_score)
        f1.append(f1_score)
    scores = pd.Series(data = [np.asarray(accuracy).mean()
                                ,np.asarray(precision).mean()
                                ,np.asarray(recall).mean()
                                ,np.asarray(f1).mean()]
                         , index = ['accuracy', 'precision', 'recall', 'f1'])
    return scores





