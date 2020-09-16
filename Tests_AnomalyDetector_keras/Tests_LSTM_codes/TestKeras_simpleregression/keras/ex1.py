#!/usr/bin/env python3

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sklearn.preprocessing as prep
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
# Generate dummy data
a,b = 1,0.5


import numpy as np

np.random.seed(230189)
# training
x_train = np.random.random((200, 1))
scaler = prep.MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
print(x_train)

#y_1 = a*x_train+b
y_train = a*x_train+b+0.1*np.random.random((200, 1))



#Test
x_test = np.random.random((10, 1))
scaler = prep.MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(x_test)
x_test = scaler.transform(x_test)

#y_test = a*x_test+b+0.1*np.random.random((10, 1))




plt.plot(x_train,y_train,'.', label = 'train')
#plt.plot(x_test,y_test,'.', label = 'test')

plt.legend()
plt.show()


def baseline_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=1))
    #model.add(Dense(64, activation='relu', input_dim=1))
    model.add(Dense(1, activation="linear"))
    optimizer = RMSprop(0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model



model1 = baseline_model()

model1.fit(x_train, y_train, epochs=40, verbose=1, validation_split=0.3)


y = model1.predict(x_train)

plt.figure()
plt.plot(x_train,y_train,'.', label='train')
plt.plot(x_train, y , '.', label='pred on training set')
plt.legend()
plt.show()
plt.close()



y_test = model1.predict(x_test)
plt.figure()
plt.plot(x_train,y_train,'.', label='train')
plt.plot(x_test, y_test , '*', label='test')
plt.legend()
plt.show()
plt.close()











'''
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, x_train, y_2, cv=kfold)
print(results)
'''
