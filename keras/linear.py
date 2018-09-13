# -*- coding: UTF-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import random
from keras.layers.normalization import BatchNormalization

model=Sequential()

model.add(Dense(output_dim=1, input_dim=1,activation='linear'))
# model.add(BatchNormalization())
# model.add(Dense(output_dim=1))
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=["accuracy"])

def creat_data_line(w,b,min,max,num):
    """
    y=w*x+b+noise
    :param w:  生成直线的斜率
    :param b:   生成直线的偏置
    :param min: x的最小值
    :param max: x的最大值
    :param num: 随机数的个数
    :return: x，y
    """
    x=np.zeros(num)
    y=np.zeros(num)
    for i in range(1,num):
        x[i]=random.uniform(min,max)
        y[i]=w*x[i]+(b+random.uniform(-0.02,0.02))
        #print(x[i],y[i])
    return x,y
#产生数据：y=x+3 x属于（0,5），20000个数据
x,y=creat_data_line(1,3,5,10,20000)
hist=model.fit(x,y,batch_size=1)
x2,y2=creat_data_line(1,3,5,10,200)
score=model.evaluate(x2,y2,batch_size=16)
print(score)
print( model.predict([3]))
