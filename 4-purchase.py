# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:39:01 2018

@author: zhengyuv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[ : ,2:4].values
Y = dataset.iloc[ : ,4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test )

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
print("准确率为：", acc)
















