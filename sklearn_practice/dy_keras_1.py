# -*- coding: UTF-8 -*-

import numpy as np
import sys

np.random.seed(1337)
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 生成数据
X = np.linspace(-1, 1, 200) #在返回（-1, 1）范围内的等差序列
np.random.shuffle(X)    # 打乱顺序
Y = 0.5  * X +2 + np.random.normal(0, 0.05, (200, )) #生成Y并添加噪声
# Y = 0.5 * X + 2 + np.random.random()/10
# plot
plt.scatter(X, Y)
# plt.show()

X_train, Y_train = X[:160], Y[:160]     # 前160组数据为训练数据集
X_test, Y_test = X[160:], Y[160:]      #后40组数据为测试数据集

# dick added
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=0)

# 构建神经网络模型
model = Sequential()
#
# model = Sequential([
#         Dense(1, activation='linear', input_shape=(1,), kernel_initializer='glorot_uniform')
#     ])
# 定义第一层, 由于是回归模型, 因此只有一层
model.add(Dense(input_dim=1, units=1))

# model.add(Dense(input_dim=1, units=30))
# model.add(Dense(units=30, activation="sigmoid"))
# model.add(Dense(units=1,activation='softmax'))
# 选定loss函数和优化器
# model.compile(loss='mse', optimizer='sgd')

# # Dick added
# model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
model.compile(loss="mse", optimizer=SGD(lr=0.1), metrics=["accuracy"])
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=3, epochs=100)
model.fit(X, Y, batch_size=10, epochs=100, shuffle=True, verbose=1, validation_split=0.2)


predicted_m = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]
print "\nm=%.2f b=%.2f\n" % (predicted_m, predicted_b)
result = model.evaluate(X_train, Y_train, batch_size=40)
print("%s:%s"%("Training set",result))

result = model.evaluate(X_test, Y_test, batch_size=40)
print("%s:%s"%("Testing set",result))
sys.exit(0)

# 训练过程
print('Training -----------')
for step in range(501):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

# 测试过程
print('Testing ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W)
print('biases=', b)

# 将训练结果绘出
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
# plt.show()