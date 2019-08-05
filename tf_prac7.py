#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 19:58:48 2019

@author: hihyun
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import train_test_split


df=pd.read_csv(r'/Users/hihyun/Desktop/Brightics/Q2_data/0731_total.csv',encoding='utf-8')
train, test = train_test_split(df,test_size=0.2)
data_y=train['sum']
data_x=train.drop(['sum','SELL_DATE'],axis=1)
test_x=test.drop(['SELL_DATE','sum'],axis=1)
test_y=test['sum']
my_model=xgb.XGBRegressor()
my_model.fit(data_x,data_y)
y_pred=my_model.predict(test_x)


df=pd.read_csv(r'/Users/hihyun/Desktop/Brightics/Q2_data/0731_total.csv',encoding='utf-8')
train, test = train_test_split(df,test_size=0.2)
data_y=train['sum']
data_x=train.drop(['sum','SELL_DATE'],axis=1)
test_x=test.drop(['SELL_DATE','sum'],axis=1)
test_y=test['sum']
my_model=xgb.XGBRegressor()
my_model.fit(data_x,data_y)
y_pred2=my_model.predict(test_x)



"""
#교차검증
kfold=KFold(n_splits=5, shuffle=True)
cv_score=cross_val_score(my_model,data_x,data_y, cv=kfold)
cv_score=cv_score.mean()
print('r2_score : {}'.format(cv_score))

"""