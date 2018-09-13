# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
import os
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys
import pickle

np.random.seed(0)
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn_pandas import DataFrameMapper
import sklearn
import pandas as pd

MODEL_FILENAME = "yancheng_model.hdf5"
MODEL_LABELS_FILENAME = "yancheng_model_labels.dat"

data = pd.read_table('/info/ali/yancheng/train_20171215.txt',index_col=False)
(X_train, X_test, Y_train, Y_test) = train_test_split(data[['date','day_of_week','brand']], data[['cnt']], test_size=0.25, random_state=0)

train_data=data[['date','day_of_week','brand']]
train_label=data[['cnt']]

mapper = DataFrameMapper([
                          ('date', None),
                          ('day_of_week', sklearn.preprocessing.LabelBinarizer()),
                          ('brand', sklearn.preprocessing.LabelBinarizer()),
                        ]
                        )
encoded_X_train=mapper.fit_transform(X_train)
encoded_X_test=mapper.transform(X_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(mapper, f)


#test_data, test_label = get_data(100,low=1,high=100)

# Function to create model, required for KerasClassifier
def create_model(neurons=100,optimizer='rmsprop', init='glorot_uniform'):
  # 构建神经网络模型
  model = Sequential()
  # # 定义第一层
  # model.add(Dense(input_dim=3, units=neurons, activation="relu",kernel_initializer=init))
  # # model.add(Dense(units=30, activation="sigmoid"))
  # model.add(Dense(units=1, activation="softmax"))
  model.add(Dense(units=neurons, input_dim=encoded_X_train[0,:].size,activation="relu",init=init))
  # 选定loss函数和优化器
  model.add(Dense(50,activation='relu'))
  model.add(Dense(1,activation='tanh'))

  # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy"])
  # model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])
  return model

chooseParmIdc=False
if chooseParmIdc:

    # define the grid search parameters
    # grid search epochs, batch size and optimizer
    # model = KerasClassifier(build_fn=create_model, verbose=0)
    model = KerasRegressor(build_fn=create_model, verbose=0)
    optimizers = [
                    # 'rmsprop',
                    'sgd',
                ]
    init = [
            # 'glorot_uniform',
            'normal',
            # 'uniform',
            ]
    epochs = [100]
    batch_size = [10]
    shuffle=[True]
    validation_split=[0.2]
    verbose=[1]
    neurons = [100]

    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batch_size, init=init,
                      neurons=neurons,
                      verbose=verbose,shuffle=shuffle,validation_split=validation_split)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(encoded_X_train, Y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
else:
    model = create_model(neurons=100,optimizer='rmsprop', init='glorot_uniform')
    # model.fit(train_data, train_label, batch_size=20, epochs=100, shuffle=True, verbose=1, validation_split=0.2)
    model.fit(encoded_X_train, Y_train, batch_size=10, epochs=150, shuffle=True, verbose=1, validation_split=0.2)
    result=model.evaluate(encoded_X_test,Y_test,batch_size=1000)

    print('loss:%5.6f   acct:%5.6f'%(result[0],result[1]))

    # Save the trained model to disk
    model.save(MODEL_FILENAME)
    print(model.predict(encoded_X_train))