#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from utils import dataset
import numpy as np
import tensorflow as tf
import zhusuan as zs
from six.moves import range
from tensorflow.contrib import layers

from utils.utils import conv2d_transpose
from utils.utils import save_image_collections, setup_logger, kl_normal_normal

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("n_z", 8, """Dimension of the latent space.""")


# Define model parameters
ngf = 16


def deconv_res_block(input_, out_shape, resize=False):
    if not resize:
        lx_z = conv2d_transpose(input_, out_shape, kernel_size=(3, 3),
                                stride=(1, 1))
        lx_z = conv2d_transpose(lx_z, out_shape, kernel_size=(3, 3),
                                stride=(1, 1), activation_fn=None)
        lx_z += input_
    else:
        lx_z = conv2d_transpose(input_, input_.get_shape().as_list()[1:],
                                kernel_size=(3, 3), stride=(1, 1))
        lx_z = conv2d_transpose(lx_z, out_shape, kernel_size=(3, 3),
                                stride=(2, 2), activation_fn=None)
        residual = conv2d_transpose(input_, out_shape, kernel_size=(3, 3),
                                    stride=(2, 2), activation_fn=None)
        lx_z += residual
    lx_z = tf.nn.relu(lx_z)
    return lx_z


def conv_res_block(input_, out_channel, resize=False):
    if not resize:
        lz_x = layers.conv2d(input_, out_channel, 3, stride=1)
        lz_x = layers.conv2d(lz_x, out_channel, 3, stride=1, activation_fn=None)
        lz_x += input_
    else:
        lz_x = layers.conv2d(input_, out_channel, 3, stride=2)
        lz_x = layers.conv2d(lz_x, out_channel, 3, stride=1, activation_fn=None)
        residual = layers.conv2d(input_, out_channel, 3, stride=2,
                                 activation_fn=None)
        lz_x += residual
    lz_x = tf.nn.relu(lz_x)
    return lz_x


@zs.reuse('model')
def vae(observed, n, n_x, n_z, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1,
                      n_samples=n_particles)
        lx_z = layers.fully_connected(z, 7 * 7 * ngf * 2)
        lx_z = tf.reshape(lx_z, [-1, 7, 7, ngf * 2])
        lx_z = deconv_res_block(lx_z, [7, 7, ngf * 2])
        lx_z = deconv_res_block(lx_z, [14, 14, ngf * 2], resize=True)
        lx_z = deconv_res_block(lx_z, [14, 14, ngf * 2])
        lx_z = deconv_res_block(lx_z, [28, 28, ngf], resize=True)
        lx_z = deconv_res_block(lx_z, [28, 28, ngf])
        lx_z = conv2d_transpose(lx_z, [28, 28, 1], kernel_size=(3, 3),
                                stride=(1, 1), activation_fn=None)
        x_logits = tf.reshape(lx_z, [n_particles, -1, n_x])
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model, x_logits


@zs.reuse('variational')
def q_net(x, n_z, n_particles):
    with zs.BayesianNet() as variational:
        lz_x = 2 * tf.to_float(x) - 1
        lz_x = tf.reshape(lz_x, [-1, 28, 28, 1])
        lz_x = layers.conv2d(lz_x, ngf, 3, stride=1)
        lz_x = conv_res_block(lz_x, ngf)
        lz_x = conv_res_block(lz_x, ngf * 2, resize=True)
        lz_x = conv_res_block(lz_x, ngf * 2)
        lz_x = conv_res_block(lz_x, ngf * 2, resize=True)
        lz_x = conv_res_block(lz_x, ngf * 2)
        lz_x = layers.flatten(lz_x)
        lz_x = layers.fully_connected(lz_x, 500)
        z_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', z_mean, logstd=z_logstd, group_ndims=1,
                      n_samples=n_particles)
    return z


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join('data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    n_x = x_train.shape[1]
    n_z = FLAGS.n_z

    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_input = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x = tf.to_int32(tf.random_uniform(tf.shape(x_input)) <= x_input)
    n = tf.shape(x)[0]

    qz = q_net(x, n_z, n_particles)
    # log_qz = qz.log_prob(qz)
    model, _ = vae({'x': x, 'z': qz}, n, n_x, n_z, n_particles)
    log_px_qz = model.local_log_prob('x')
    eq_ll = tf.reduce_mean(log_px_qz)

    kl = kl_normal_normal(
        qz.distribution.mean, qz.distribution.logstd, 0., 0.)
    kl_term = tf.reduce_mean(tf.reduce_sum(kl, -1))
    lower_bound = eq_ll - kl_term
    cost = -lower_bound

    # log_pz = model.local_log_prob('z')
    # kl_term_est = tf.reduce_mean(log_qz - log_pz)
    # cost = kl_term

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)
    infer_op = optimizer.minimize(cost)

    # Generate images
    n_gen = 100
    _, x_logits = vae({}, n_gen, n_x, n_z, 1)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    # Define training parameters
    lb_samples = 1
    learning_rate = 1e-4
    epochs = 3000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_image_freq = 10
    save_model_freq = 100
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    result_path = "results/vae_conv_{}_".format(n_z) + \
        time.strftime("%Y%m%d_%H%M%S")

    saver = tf.train.Saver(max_to_keep=10)
    logger = setup_logger('vae_conv', __file__, result_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            logger.info('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer_op, lower_bound],
                    feed_dict={x_input: x_batch,
                               learning_rate_ph: learning_rate,
                               n_particles: lb_samples})
                lbs.append(lb)

            time_epoch += time.time()
            logger.info(
                'Epoch {} ({:.1f}s): Lower bound = {}'
                .format(epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples})
                    test_lbs.append(test_lb)
                time_test += time.time()
                logger.info('>>> TEST ({:.1f}s)'.format(time_test))
                logger.info('>> Test lower bound = {}'
                            .format(np.mean(test_lbs)))

            if epoch % save_image_freq == 0:
                logger.info('Saving images...')
                images = sess.run(x_gen)
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name)

            if epoch % save_model_freq == 0:
                logger.info('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                logger.info('Done')


if __name__ == "__main__":
    main()
