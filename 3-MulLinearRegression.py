# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:50:12 2018

@author: zhengyuv
"""

import pandas as pd
#import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 4].values

labelencoder = LabelEncoder()
X[ : , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#躲避虚拟变量陷阱
X = X[ : ,1: ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

print("R2系数为：", r2_score(Y_test, Y_pred))




