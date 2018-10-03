# -*- coding: UTF-8 -*-
import tensorflow as tf
import re

def fc_layer(bottom, neurons,name,actvation=None,reTrain=False):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = tf.Variable(tf.random_normal(shape=[dim,neurons], mean=0, stddev=1), name="weights")
        biases = tf.Variable(tf.constant(0.1, shape=[neurons]), name="biases")

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        if actvation is not None:
            fc=actvation(fc)
        _activation_summary(fc)
    return fc

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/', '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))
