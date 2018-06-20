# coding: utf-8

import pandas as pd
from pandas import concat, DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split, KFold
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn import preprocessing
import math

# 记录程序运行时间
start_time = time.time()

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


#  划分数据集
#  用sklearn.cross_validation进行训练数据集划分  7：3
train_xy,val = train_test_split(train_data, test_size = 0.3, random_state = 1)



#train_xy = train_xy.tolist()
#val = val.tolist()
train_feat = train_data
test_feat = test_data

# = test_feat['id_count']
predictors = [f for f in test_feat.columns if f not in ['id_count']]

print('开始训练...')

params_lgb = {
    'learning_rate': 0.001,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 18,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)


print('开始CV 5折训练...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle = True, random_state = 520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['id_count'], categorical_feature=['loc_id'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['id_count'])
    gbm = lgb.train(params_lgb,
                    lgb_train1,
                    num_boost_round = 50000,
                    valid_sets = lgb_train2,
                    verbose_eval = 100,
                    feval = evalerror,
                    early_stopping_rounds = 500)
    feat_imp = pd.Series(gbm.feature_importance(), index = predictors).sort_values(ascending = False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])

    # test_preds = DataFrame(test_preds)
    # for i in test_preds.columns:
    #     for j in range(len(test_preds)):
    #         if (test_preds[i] < 0)[j]:
    #             test_preds[i][j] = - test_preds[i][j]

print('线下得分：    {}'.format(mean_squared_error(train_feat['id_count'], train_preds) * 0.5))
#print('CV训练用时{}秒'.format(time.time() - t0))
submission = pd.DataFrame({'pred': test_preds.mean(axis = 1)})

#submission = submission.astype(float)

# submission.to_csv(r'sub_lgb_12_0527_1140.csv', header = 'id_cound',
#                   index = False, float_format = '%d')


#np.savetxt('submission_lgb.csv',np.c_[range(1,len(test_data) + 1), test_preds],
#                                      delimiter = ',', header = 'loc_id, time_stamp, id_count',
#                                      comments = '', fmt = '%d')

print("lgb success")



y = train_xy.id_count
X = train_xy.drop(['id_count'], axis = 1)
val_y = val.id_count
val_X = val.drop(['id_count'], axis = 1)

target = 'id_count'
#IDcol = 'member_id'

#xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X, label = val_y)
xgb_train = xgb.DMatrix(X, label = y)
xgb_test = xgb.DMatrix(test_data)

#xgboost模型
params_xgb = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'n_estimators': 1000,
        'gamma': 2,
        'max_depth': 10,            # default 6
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 6 ,
        'reg_alpha': 0.005,
        'eta': 0.3,                 # default, typical: 0.01 - 0.2
        'seed': 0,
        #'eval_metric':'auc'
        }

plst = list(params_xgb.items())
num_rounds = 10000
watchlist = [(xgb_train,'train'),(xgb_val,'val')]

model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds = 1000)
model.save_model('../buptloc/xgb.model')
print("best best_ntree_limit",model.best_ntree_limit)

xgb_preds = model.predict(xgb_test, ntree_limit = model.best_ntree_limit)
#xgb_preds = DataFrame(xgb_preds)

# for i in xgb_preds.columns:
#     for j in range(len(xgb_preds)):
#         if (xgb_preds[i] < 0)[j]:
#             xgb_preds[i][j] = - xgb_preds[i][j]

# inverse : problem occurred
#xgb_preds = preprocessing.StandardScaler.inverse_transform(xgb_preds, 'id_count')

submission = 0.7 * submission + 0.3 * xgb_preds

submission = DataFrame(submission)

submission.to_csv(r'sub_lgb_12_0527_1510.csv', header = 'id_cound',
                  index = False, float_format = '%d')

# np.savetxt('submission_xgb_0526_1630.csv',np.c_[range(1, len(test_data)+1), xgb_preds],
#                                       delimiter = ',', header = 'loc_id, id_count',
#                                       comments = '', fmt = '%d')

print("xgb success!")



