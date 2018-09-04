# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:08:48 2018

@author: zhengyuv
"""
#1.数据预处理
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : , :1].values
Y = dataset.iloc[ : , 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#因为没有缺失值，所以不用处理缺失值，该数据集也不需要进行特征缩放

#2.训练模型
regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)

#3.预测结果
Y_pred = regressor.predict(X_test)

#4.可视化
#训练集结果可视化
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()
#测试集结果可视化
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.show()








