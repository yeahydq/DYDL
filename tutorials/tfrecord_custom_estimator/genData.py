# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from util.tfrecorder import TFrecorder
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def createDir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

dirs = ['mnist_tfrecord', 'mnist_tfrecord/train','mnist_tfrecord/test']

for dir in dirs:
    createDir(dir)

# info of data
df = pd.DataFrame({'name':['image','label'],
                  'type':['float32','int64'],
                  'shape':[(784,),()],
                  'isbyte':[False,False],
                  "length_type":['fixed','fixed'],
                  "default":[np.NaN,np.NaN]})

data_info_path = 'mnist_tfrecord/data_info.csv'
df.to_csv(data_info_path,index=False)

# write
tfr = TFrecorder()

def writeData(dataset, outputPath,num_examples_per_file=1000):
    path = outputPath
    num_so_far = 0

    writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' % (path, num_so_far, num_examples_per_file))
    # write mutilple examples
    for i in np.arange(dataset.num_examples):
        features = {}
        # write image of one example
        tfr.feature_writer(df.iloc[0], dataset.images[i], features)
        # write label of one example
        tfr.feature_writer(df.iloc[1], dataset.labels[i], features)

        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
        if i % num_examples_per_file == 0 and i != 0:
            writer.close()
            num_so_far = i
            writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' % (path, num_so_far, i + num_examples_per_file))
            print('saved %s%s_%s.tfrecord' % (path, num_so_far, i + num_examples_per_file))
    writer.close()

writeData(mnist.train,'mnist_tfrecord/train/train',1000)
writeData(mnist.test,'mnist_tfrecord/test/test',1000)
