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

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = 'longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value'.split(',')
LABEL_COLUMN = 'median_house_value'
DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

SCALE = 100000

# Create the custom estimator
def custom_estimator(features, labels, mode, params):
    # 0. Extract data from feature columns
    input_layer = tf.feature_column.input_layer(features, params['parm_feature_columns'])

    # 1. Define Model Architecture
    fc0 = tf.layers.dense(input_layer, 30, activation=None, name='fc0')
    fc1 = tf.layers.dense(fc0, 20, tf.nn.relu, name='fc1')
    fc2 = tf.layers.dense(fc1, 20, activation=None, name='fc2')
    fc3 = fc_layer(fc2, 20, "fc3", actvation=tf.nn.relu)
    fc4 = fc_layer(fc3, 20, "fc4")
    predictions = tf.layers.dense(fc4, 1, activation=None, name='predictions')

    #   predictions = tf.layers.dense(input_layer,1,activation=None)

    # 2. Loss function, training/eval ops
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.expand_dims(tf.cast(labels, dtype=tf.float32), -1)
        loss = tf.losses.mean_squared_error(labels, predictions)
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.2)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=0.2,
            optimizer=optimizer)
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(labels * SCALE, predictions * SCALE)
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops = None

    # 3. Create predictions
    predictions_dict = {"predicted": predictions}

    # 4. Create export outputs
    export_outputs = {"regression_export_outputs": tf.estimator.export.RegressionOutput(value=predictions)}

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
def build_estimator(model_dir, nbuckets, hidden_units):
    """
     Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model.
  """

    # Input columns
    feature_columns = {
        colname: tf.feature_column.numeric_column(colname) \
        for colname in 'housing_median_age,median_income,num_rooms,num_bedrooms,persons_per_house'.split(',')
    }
    # Bucketize lat, lon so it's not so high-res; California is mostly N-S, so more lats than lons
    feature_columns['longitude'] = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'),
                                                                       np.linspace(-124.3, -114.3, 5).tolist())
    feature_columns['latitude'] = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'),
                                                                      np.linspace(32.5, 42, 10).tolist())

    # Feature cross (特征交叉)
    ploc = tf.feature_column.crossed_column([feature_columns['longitude'], feature_columns['latitude']],
                                            100)  # pickup localtion
    feature_columns['dy_embed'] = tf.feature_column.embedding_column(ploc, 10)


    estimator = tf.estimator.Estimator(
        model_fn=custom_estimator,
        model_dir=model_dir,
        params={'parm_feature_columns': list(feature_columns.values())},
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
    features['num_rooms'] = features['total_rooms'] / features['households']
    features['num_bedrooms'] = features['total_bedrooms'] / features['households']
    features['persons_per_house'] = features['population'] / features['households']
    return features


# Create serving input function to be able to serve predictions
def serving_input_fn():
    # Dick: including 'new add' feature
    # feature_placeholders = {
    #     # All the real-valued columns                                 (0~1 is dayofweek and hourofday)
    #     column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:]
    # }
    # feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
    # feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])

    # Input columns
    feature_placeholders = {
        colname: tf.placeholder(tf.float32, [None]) \
        for colname in 'longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value'.split(',')
    }

    # Dick: not include the 'new add' feature
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(add_engineered(features), feature_placeholders)


# Create input function to load data into datasets, return dict with fieldName/value
def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            # remove label column
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).skip(1).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels

    return _input_fn


# Create estimator train and evaluate function
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'].split(' '))
    # 1/2 train spec
    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(
            filename=args['train_data_paths'],
            mode=tf.estimator.ModeKeys.TRAIN,
            batch_size=args['train_batch_size']),
        max_steps=args['train_steps'])
    # 2/2 evaluation spec
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(
            filename=args['eval_data_paths'],
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=args['eval_batch_size']),
        steps=100,
        exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# If we want to use TFRecords instead of CSV
def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)
    }

