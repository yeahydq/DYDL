# -*- coding: UTF-8 -*-
# !/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tfutil import *
from util.tfrecorder import TFrecorder

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = 'longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value'.split(',')
LABEL_COLUMN = 'median_house_value'
DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

SCALE = 100000

# Create the custom estimator
def custom_estimator(features, mode):
    # 0. Extract data from feature columns
    # reshape 784维的图片到28x28的平面表达，1为channel数
    features['image'] = tf.reshape(features['image'],[-1,28,28,1])
    # shape: [None,28,28,1]
    conv1 = tf.layers.conv2d(
        inputs=features['image'],
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv1')
    # shape: [None,28,28,32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name= 'pool1')
    # shape: [None,14,14,32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv2')
    # shape: [None,14,14,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name= 'pool2')
    # shape: [None,7,7,64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name= 'pool2_flat')
    # shape: [None,3136]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name= 'dense1')
    # shape: [None,1024]
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # shape: [None,1024]
    logits = tf.layers.dense(inputs=dropout, units=10, name= 'output')
    # shape: [None,10]
    predictions = {
        "image":features['image'],
        "conv1_out":conv1,
        "pool1_out":pool1,
        "conv2_out":conv2,
        "pool2_out":pool2,
        "pool2_flat_out":pool2_flat,
        "dense1_out":dense1,
        "logits":logits,
        "classes": tf.argmax(input=logits, axis=1),
        # "labels": features['label'],
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    # 2. Loss function, raining/eval ops
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        labels = features['label']
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        # optimizer = tf.train.FtrlOptimizer(learning_rate=1e-3)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=1e-3,
            optimizer=optimizer)
        eval_metric_ops = {
            # "crossEntropy": loss,
            "accuracy": tf.metrics.accuracy(labels=features['label'], predictions=predictions["classes"])
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops = None

    # 3. Create predictions
    # predictions_dict = {"classes": predictions}
    predictions_dict=predictions["classes"]

    # 4. Create export outputs
    #export_outputs = {"regression_export_outputs": tf.estimator.export.RegressionOutput(value=predictions)}
    # export_outputs = {"predict_export_outputs": tf.estimator.export.PredictOutput(outputs=predictions["classes"])}
    export_outputs = {
        'classes': tf.estimator.export.PredictOutput({"probabilities": predictions["probabilities"], "classes": predictions["classes"]})}

    # 5. Return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs
        #       training_hooks=[summary_hook]
    )

# Build the estimator
def build_estimator(model_dir):
    """
     Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model.
  """

    # # Input columns
    # feature_columns = {
    #     colname: tf.feature_column.numeric_column(colname) \
    #     for colname in 'housing_median_age,median_income,num_rooms,num_bedrooms,persons_per_house'.split(',')
    # }
    # # Bucketize lat, lon so it's not so high-res; California is mostly N-S, so more lats than lons
    # feature_columns['longitude'] = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'),
    #                                                                    np.linspace(-124.3, -114.3, 5).tolist())
    # feature_columns['latitude'] = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'),
    #                                                                   np.linspace(32.5, 42, 10).tolist())
    #
    # # Feature cross (特征交叉)
    # ploc = tf.feature_column.crossed_column([feature_columns['longitude'], feature_columns['latitude']],
    #                                         100)  # pickup localtion
    # feature_columns['dy_embed'] = tf.feature_column.embedding_column(ploc, 10)


    estimator = tf.estimator.Estimator(
        model_fn=custom_estimator,
        model_dir=model_dir,
        # params={'parm_feature_columns': list(feature_columns.values())},
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=100,
            save_summary_steps=100
        )
    )

    # add extra evaluation metric for hyperparameter tuning
    # estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
    return estimator


# Create feature engineering function that will be used in the input and serving input functions
def add_engineered(features):
    # this is how you can do feature engineering in TensorFlow
    # features['num_rooms'] = features['total_rooms'] / features['households']
    # features['num_bedrooms'] = features['total_bedrooms'] / features['households']
    # features['persons_per_house'] = features['population'] / features['households']

    return features


# Create serving input function to be able to serve predictions
def serving_input_fn():

    reciever_tensors = {
        # The size of input image is flexible.
        'image': tf.placeholder(tf.float32, [None, None, None, 1]),
    }

    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        'image': tf.image.resize_images(reciever_tensors['image'], [28, 28]),
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)
    #
    # inputs = {'image': tf.placeholder(tf.float32, [None, 28, 28, 1])}
    # features = inputs # as-is
    # return tf.estimator.export.ServingInputReceiver(features, inputs)

def read_dataset(filename, mode, batch_size=512):
    import pandas as pd
    df = pd.read_csv(filename, sep=",")

    INPUT_COLUMNS = [
        "housing_median_age", "median_income", "num_rooms", "num_bedrooms", "persons_per_house", "longitude",
        "latitude"
    ]
    df = add_engineered(df)
    msk = np.random.rand(len(df)) < 0.8
    traindf = df[msk]
    evaldf = df[~msk]
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.inputs.pandas_input_fn(x=traindf[INPUT_COLUMNS],
                                                         y=traindf["median_house_value"] / SCALE,
                                                         num_epochs=None,
                                                         batch_size=batch_size,
                                                         shuffle=True)
    else:
        return tf.estimator.inputs.pandas_input_fn(x=evaldf[INPUT_COLUMNS],
                                                        y=evaldf["median_house_value"] / SCALE,  # note the scaling
                                                        num_epochs=1,
                                                        batch_size=batch_size,
                                                        shuffle=False)

tfr = TFrecorder()
def input_fn_maker(path, data_info_path, shuffle=False, batch_size = 128, epoch = 100, padding = None):
    def input_fn():
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset=tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle = shuffle,
                            batch_size = batch_size, epoch = epoch, padding =padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn

padding_info = ({'image':[784,],'label':[]})
test_input_fn = input_fn_maker('mnist_tfrecord/test/',  'mnist_tfrecord/data_info.csv',batch_size = 512,
                               padding = padding_info)
train_input_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', shuffle=True, batch_size = 128,
                               padding = padding_info)
train_eval_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', batch_size = 512,
                               padding = padding_info)

# Create estimator train and evaluate function
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'])
    # 1/2 train spec
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=args['train_steps'])
    # 2/2 evaluation spec
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=train_eval_fn,
        steps=args['eval_steps'],
        throttle_secs=args['eval_throttle_secs'], # Do not re-evaluate unless the last evaluation was started at least this many seconds ago
        exporters=exporter
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# If we want to use TFRecords instead of CSV
def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))

