# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:33:18 2018

@author: zhengyuv
"""
#1.导入库
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#2.导入数据集
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
print("X")
print(X)
print("Y")
print(Y)

#3.处理丢失数据
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis=0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print("处理缺失值后")
print("X")
print(X)

#4.解析分类数据
labelencoder_X = LabelEncoder()
X[ : ,0] = labelencoder_X.fit_transform(X[ : , 0])
#创建虚拟变量
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("对数据重编码后")
print("X")
print(X)
print("Y")
print(Y)

#5.拆分数据集
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2,random_state=0)
print("拆分完数据集")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

#6.特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print("特征缩放后")
print("X_train")
print(X_train)
print("X_test")
print(X_test)









