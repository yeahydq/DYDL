# -*- coding: UTF-8 -*-
# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100]
batches = [5, 10]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Best: 0.744792 using {'epochs': 100, 'init': 'uniform', 'optimizer': 'adam', 'batch_size': 5}
# 0.688802 (0.013279) with: {'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'batch_size': 5}
# 0.669271 (0.043067) with: {'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'adam', 'batch_size': 5}
# 0.697917 (0.012075) with: {'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop', 'batch_size': 5}
# 0.700521 (0.015733) with: {'epochs': 50, 'init': 'normal', 'optimizer': 'adam', 'batch_size': 5}
# 0.717448 (0.012890) with: {'epochs': 50, 'init': 'uniform', 'optimizer': 'rmsprop', 'batch_size': 5}
# 0.717448 (0.006639) with: {'epochs': 50, 'init': 'uniform', 'optimizer': 'adam', 'batch_size': 5}
# 0.687500 (0.009568) with: {'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'batch_size': 5}
# 0.661458 (0.036966) with: {'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'adam', 'batch_size': 5}
# 0.735677 (0.015733) with: {'epochs': 100, 'init': 'normal', 'optimizer': 'rmsprop', 'batch_size': 5}
