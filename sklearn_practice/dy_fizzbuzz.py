# -*- coding: UTF-8 -*-
import os
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys

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


DIGITS = 10


def binary_encode(num, digits=DIGITS):
  return [num >> i & 1 for i in range(digits)][::-1]


def label_encode(num):
  if num % 15 == 0:
    return [1, 0, 0, 0]
  elif num % 3 == 0:
    return [0, 1, 0, 0]
  elif num % 5 == 0:
    return [0, 0, 1, 0]
  else:
    return [0, 0, 0, 1]

def get_data(num, low=101, high=10000):
  binary_num_list = []
  label_list = []
  for i in range(num):
    n = np.random.randint(low, high, 1)[0]
    n = i
    binary_num_list.append(np.array(binary_encode(n)))
    label_list.append(np.array(label_encode(n)))
  return np.array(binary_num_list), np.array(label_list)

# 生成数据
train_data, train_label = get_data(1000)
test_data, test_label = get_data(100,low=1,high=100)

# encode class values as integers
lb = LabelBinarizer().fit(train_label)
encoded_train_label = lb.transform(train_label)
encoded_test_label = lb.transform(test_label)

# Function to create model, required for KerasClassifier
def create_model(neurons=1000,optimizer='adam', init='glorot_uniform'):
  # 构建神经网络模型
  model = Sequential()
  # 定义第一层
  model.add(Dense(input_dim=10, units=neurons, activation="relu",kernel_initializer=init))
  # model.add(Dense(units=30, activation="sigmoid"))
  model.add(Dense(units=4, activation="softmax"))
  # 选定loss函数和优化器
  model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
  return model

chooseParmIdc=True
if chooseParmIdc:

    # define the grid search parameters
    # grid search epochs, batch size and optimizer
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # model = KerasRegressor(build_fn=create_model, verbose=0)
    optimizers = [
                    # 'rmsprop',
                    'adam',
                ]
    init = [
            # 'glorot_uniform',
            'normal',
            # 'uniform',
            ]
    epochs = [100, 150]
    batch_size = [10, 20]
    shuffle=[True,False]
    validation_split=[0.2]
    verbose=[1]
    neurons = [10,100,1000,10000]

    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batch_size, init=init,
                      neurons=neurons,
                      verbose=verbose,shuffle=shuffle,validation_split=validation_split)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
else:
    model = create_model()
    # model.fit(train_data, train_label, batch_size=20, epochs=100, shuffle=True, verbose=1, validation_split=0.2)
    model.fit(train_data, train_label, batch_size=10, epochs=150, shuffle=True, verbose=1, validation_split=0.2)
    result=model.evaluate(test_data,test_label,batch_size=1000)

    print('loss:%5.6f   acct:%5.6f'%(result[0],result[1]))

    #
    test_data = np.array([binary_encode(i) for i in range(1, 101)])
    # pred=model.predict(test_data)
    pred=model.predict_classes(test_data)

    # init_lables = lb.inverse_transformnsform(pred)
    # print(init_lables)


    # Convert the one-hot-encoded prediction back to a normal letter
    results = []
    for i in range(1, 100):
        results.append('{}'.format(
                                    ['fizzbuzz', 'fizz', 'buzz', i][pred[i - 1]]
                                   )
                       )
    print(', '.join(results))
