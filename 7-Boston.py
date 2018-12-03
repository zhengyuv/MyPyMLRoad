# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:33:15 2018

@author: zhengyuv
"""
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

#prepare data
(train_data, train_label), (test_data, test_label) = boston_housing.load_data(path=r'D:\datasets\boston_housing.npz')

mean = train_data.mean(axis=0)
train_data = train_data-mean
std = train_data.std(axis=0)
train_data = train_data/std
test_data = test_data-mean
test_data = test_data/std

#model definition
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(13, )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model

#model validation
epochs_num = 100
k = 4
samples_num = len(train_data)//k
all_mae_histories = []
for i in range(k):
    print('第', i+1, '折:')
    
    val_data = train_data[i*samples_num:(i+1)*samples_num]
    val_label = train_label[i*samples_num:(i+1)*samples_num]
    
    partial_train_data = np.concatenate([train_data[:i*samples_num], 
                                        train_data[(i+1)*samples_num:]],
                                        axis=0)
    partial_test_data = np.concatenate([train_label[:i*samples_num], 
                                       train_label[(i+1)*samples_num:]], 
                                       axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data,
                        partial_test_data,
                        epochs=epochs_num,
                        batch_size=1,
                        verbose=1,
                        validation_data=(val_data, val_label))
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(epochs_num)]

plt.plot(range(1, epochs_num+1), average_mae_history)
plt.xlabel('epochs')
plt.ylabel('mae')
plt.show()

#prediction
model = build_model()
model.fit(train_data,
          train_label,
          epochs=80,
          batch_size=16,
          verbose=1)
test_mse, test_mae = model.evaluate(test_data, test_label)
print('测试集平均误差：',test_mae)