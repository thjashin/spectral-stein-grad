#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import os
import logging

import numpy as np
import tensorflow as tf
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib import layers


def set_plot():
    import matplotlib as _mpl
    _mpl.use('TkAgg')
    from matplotlib import pyplot as _plt
    _plt.rcParams['font.size'] = 16
    _plt.rcParams['axes.labelsize'] = 1 * _plt.rcParams['font.size']
    _plt.rcParams['axes.titlesize'] = 2 * _plt.rcParams['font.size']
    _plt.rcParams['legend.fontsize'] = 1.5 * _plt.rcParams['font.size']
    _plt.rcParams['xtick.labelsize'] = 1 * _plt.rcParams['font.size']
    _plt.rcParams['ytick.labelsize'] = 1 * _plt.rcParams['font.size']
    return _plt


def create_session(log_device_placement):
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=log_device_placement)
    return tf.Session(config=config)


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    :param tower_grads: List of lists of (gradient, variable) tuples.
        The outer list is over individual gradients. The inner list is over
        the gradient calculation for each tower.
    :return: List of pairs of (gradient, variable) where the gradient has
        been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        if grad_and_vars[0][0] is None:
            continue
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_losses(tower_losses):
    """
    Calculate the average loss or other quantity for all towers.

    :param tower_losses: A list of lists of quantities. The outer list is over
        towers. The inner list is over losses or other quantities for each
        tower.
    :return: A list of quantities that have been averaged over all towers.
    """
    ret = []
    for quantities in zip(*tower_losses):
        ret.append(tf.add_n(quantities) / len(quantities))
    return ret


@add_arg_scope
def conv2d_transpose(
        inputs,
        out_shape,
        kernel_size=(5, 5),
        stride=(1, 1),
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=layers.xavier_initializer(),
        scope=None,
        reuse=None):
    batchsize = tf.shape(inputs)[0]
    in_channels = int(inputs.get_shape()[-1])

    output_shape = tf.stack([batchsize, out_shape[0],
                             out_shape[1], out_shape[2]])
    filter_shape = [kernel_size[0], kernel_size[1], out_shape[2], in_channels]

    with tf.variable_scope(scope, 'Conv2d_transpose', [inputs], reuse=reuse):
        w = tf.get_variable('weights', filter_shape,
                            initializer=weights_initializer)

        outputs = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                            strides=[1, stride[0], stride[1], 1])
        outputs.set_shape([None] + out_shape)

        if not normalizer_fn:
            biases = tf.get_variable('biases', [out_shape[2]],
                                     initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs


def kl_normal_normal(mean1, logstd1, mean2, logstd2):
    return logstd2 - logstd1 + (tf.exp(2 * logstd1) + (mean1 - mean2) ** 2) / \
        (2 * tf.exp(2 * logstd2)) - 0.5


def setup_logger(name, src, result_path, filename="log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(result_path, filename)
    makedirs(log_path)
    info_file_handler = logging.FileHandler(log_path)
    info_file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler)
    logger.info(src)
    with open(src) as f:
        logger.info(f.read())
    return logger


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :return: `uint8` numpy array
        The output image.
    """
    makedirs(filename)
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
