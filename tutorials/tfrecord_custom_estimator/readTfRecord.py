# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from util.tfrecorder import TFrecorder
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import shutil


def createDir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

tfr = TFrecorder()
def input_fn_maker(path, data_info_path, shuffle=False, batch_size = 1, epoch = 1, padding = None):
    def input_fn():
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset=tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle = shuffle,
                            batch_size = batch_size, epoch = epoch, padding =padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn

padding_info = ({'image':[784,],'label':[]})
test_input_fn = input_fn_maker('mnist_tfrecord/test/',  'mnist_tfrecord/data_info.csv', shuffle=True,
                               padding = padding_info)
train_input_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', shuffle=True, batch_size = 512,
                               padding = padding_info)
train_eval_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', batch_size = 512,
                               padding = padding_info)
test_inputs = test_input_fn()

#++++++++++++++++++
# Method 1
#++++++++++++++++++
#写入图片路径
outImagePath = './decode/'
shutil.rmtree(outImagePath)
createDir(outImagePath)

with tf.Session() as sess:
    for i in range(10):
        rst=sess.run(test_inputs)
        print(test_inputs['image'].shape)
        print(rst['label'])
        plt.imshow(rst['image'].reshape((28,28)),cmap=plt.cm.gray)
        # plt.savefig(swd + str(i) + '_''Label_' + str(rst['label']) + '.jpg')
        plt.show()
        img = Image.fromarray(rst['image'].reshape(28,28)*255).convert('L')  # 这里Image是之前提到的
        img.save(outImagePath + str(i) + '_''Label_' + str(rst['label'][0]) + '.jpg')  # 存下图片

#++++++++++++++++++
# Method 2
#++++++++++++++++++
# sess =tf.InteractiveSession()
# print(test_inputs['image'].eval().shape)
# plt.imshow(test_inputs['image'].eval().reshape((28,28)),cmap=plt.cm.gray)
# sess.close()
