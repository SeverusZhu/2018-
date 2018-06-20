from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from vecstack import stacking
from math import sqrt

import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def transformer(y, func=None):
    if func is None:
        return y
    else:
        return func(y)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, regression=True,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
    # Print type of task
    if regression and verbose > 0:
        print('task:   [regression]')
    elif not regression and verbose > 0:
        print('task:   [classification]')

    # Specify default metric for cross-validation
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        metric = accuracy_score

    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)

    # Split indices to get folds (stratified can be used only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(y_train, n_folds, shuffle=shuffle, random_state=random_state)
    else:
        kf = KFold(len(y_train), n_folds, shuffle=shuffle, random_state=random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))

    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))

        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], len(kf)))

        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train.loc[tr_index]
            y_tr = y_train.loc[tr_index]
            X_te = X_train.loc[te_index]
            y_te = y_train.loc[te_index]

            # Fit 1-st level model
            model = model.fit(X_tr, transformer(y_tr, func=transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(model.predict(X_te), func=transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(model.predict(X_test), func=transform_pred)

            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))

        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis=1)
        else:
            S_test[:, model_counter] = st.mode(S_test_temp, axis=1)[0].ravel()

        if verbose > 0:
            print('    ----')
            print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    return (S_train, S_test)

print("starting modeling.py")

data1 = pd.read_csv("../buptloc/data/0001.csv", low_memory = False)
data2 = pd.read_csv("../buptloc/data/0002.csv", low_memory = False)
data3 = pd.read_csv("../buptloc/data/0003.csv", low_memory = False)
data4 = pd.read_csv("../buptloc/data/0004.csv", low_memory = False)
data5 = pd.read_csv("../buptloc/data/0005.csv", low_memory = False)
data6 = pd.read_csv("../buptloc/data/0006.csv", low_memory = False)
data7 = pd.read_csv("../buptloc/data/0007.csv", low_memory = False)
data8 = pd.read_csv("../buptloc/data/0008.csv", low_memory = False)
data9 = pd.read_csv("../buptloc/data/0009.csv", low_memory = False)
#data10 = pd.read_csv("../buptloc/data/0010.csv", low_memory = False)
data10 = pd.read_csv("../buptloc/data/10_09_31.csv", low_memory = False)
#data11 = pd.read_csv("../buptloc/data/0011.csv", low_memory = False)
data11 = pd.read_csv("../buptloc/data/0011_0527.csv", low_memory = False)

#frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
frames = [data9, data10, data11]
data1 = pd.concat(frames)


data1 = DataFrame(data1)

test_data = pd.read_csv("../buptloc/data/test_12_1.csv", low_memory = False)

test_data = DataFrame(test_data)

train_data = data1.drop(['time_stamp'], axis = 1)
test_data = test_data.drop(['time_stamp'], axis = 1)


#  problems occurred while using xgb, turn into lgb instead
# 数据标准化
#train_data = preprocessing.scale(train_data)
train_data = DataFrame(train_data)
train_data.columns = ['loc_id', 'id_count', 'day', 'hour', 'month', 'dayofweek']

#test_data = preprocessing.scale(test_data)
test_data = DataFrame(test_data)
test_data.columns = ['loc_id', 'day', 'hour', 'month', 'dayofweek']

#X_train, X_test = train_test_split(train_data, test_size = 0.3, random_state = 1)
X_train = train_data.drop(['id_count'], axis = 1)
X_train = X_train.replace([np.inf, np.nan], 0).reset_index(drop = True)
X_test = test_data.replace([np.inf, np.nan], 0).reset_index(drop = True)

y_train = train_data["id_count"].reset_index(drop=True)
y_train = y_train.replace([np.inf, np.nan], 0).reset_index(drop = True)

#X_train = tr_user[features].replace([np.inf, np.nan], 0).reset_index(drop=True)
#X_test = ts_user[features].replace([np.inf, np.nan], 0).reset_index(drop=True)
#y_train = tr_user["loan_sum"].reset_index(drop=True)

# Caution! All models and parameter values are just
# demonstrational and shouldn't be considered as recommended.
# Initialize 1-st level models.
models = [
    ExtraTreesClassifier(random_state=0, n_jobs=-1,
                        n_estimators=300, max_depth=3),

    RandomForestClassifier(random_state=0, n_jobs=-1,
                          n_estimators=300, max_depth=3),

    XGBClassifier(seed=0, learning_rate=0.05,
                 n_estimators=300, max_depth=3),

    LGBMClassifier(num_leaves=8, learning_rate=0.05, n_estimators=300)
]

# Compute stacking features

S_train, S_test = stacking(models, X_train, y_train, X_test, regression=False, metric=mean_squared_error, n_folds=5,
                           shuffle=True, random_state=0, verbose=2)

# Fit 2-nd level model
model = LGBMClassifier(num_leaves=8, learning_rate=0.05, n_estimators=300)
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)

id_test = test_data['loc_id']
stacking_sub = pd.DataFrame({'loc_id': id_test, 'id_count': y_pred})
print(stacking_sub.describe())
stacking_sub.loc[stacking_sub["id_count"] < 0, "id_count"] = - stacking_sub["id_count"]
print('saving submission...')
#now_time = time.strftime("%m-%d %H_%M_%S", time.localtime())
stacking_sub[["loc_id", "id_count"]].to_csv("/home/severus/Documents/loc_stacking_3.csv", index=False,
                                                  header=False, float_format = '%d')


