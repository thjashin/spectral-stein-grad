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

from estimator import SteinScoreEstimator, SpectralScoreEstimator
from utils.utils import conv2d_transpose
from utils.utils import create_session, average_gradients, average_losses
from utils.utils import save_image_collections, setup_logger


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("n_z", 8, """Dimension of the latent space.""")
tf.flags.DEFINE_string("estimator", "spectral", """The KL term estimator.""")
tf.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use""")
tf.flags.DEFINE_boolean('log_device_placement', False,
                        """Whether to log device placement.""")


# Define model parameters
ngf = 16


# Estimator parameters
n_est = 100
eta = 0.001


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
    return model, z, x_logits


@zs.reuse('variational')
def implicit(x, n_z, n_particles):
    with zs.BayesianNet() as variational:
        lz_x = 2 * tf.to_float(x) - 1
        lz_x = tf.reshape(lz_x, [-1, 28, 28, 1])
        lz_x = layers.conv2d(lz_x, ngf, 3, stride=1)
        lz_x = conv_res_block(lz_x, ngf)
        lz_x = conv_res_block(lz_x, ngf * 2, resize=True)
        lz_x = conv_res_block(lz_x, ngf * 2)
        lz_x = conv_res_block(lz_x, ngf * 2, resize=True)
        lz_x = conv_res_block(lz_x, ngf * 2)
        h_mean = layers.flatten(lz_x)
        h_logstd = tf.get_variable(
            "h_logstd", shape=[7 * 7 * ngf * 2], dtype=tf.float32,
            initializer=tf.constant_initializer(0.))
        h = zs.Normal('h', h_mean, logstd=h_logstd, group_ndims=1,
                      n_samples=n_particles)
        lz_x = layers.fully_connected(h, 500)
        z = layers.fully_connected(lz_x, n_z, activation_fn=None)
    return z


q_net = implicit


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
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

    def build_tower_graph(x, id_):
        tower_x = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
                    (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        n = tf.shape(tower_x)[0]

        # qz_samples: [n_particles, n, n_z]
        qz_samples = q_net(tower_x, n_z, n_particles)
        # Use a single particle for the reconstruction term
        observed = {'x': tower_x, 'z': qz_samples[:1]}
        model, z, _ = vae(observed, n, n_x, n_z, 1)
        # log_px_qz: [1, n]
        log_px_qz = model.local_log_prob('x')
        eq_ll = tf.reduce_mean(log_px_qz)
        # log_p_qz: [n_particles, n]
        log_p_qz = z.log_prob(qz_samples)
        eq_joint = eq_ll + tf.reduce_mean(log_p_qz)

        if FLAGS.estimator == "stein":
            estimator = SteinScoreEstimator(eta=eta)
        elif FLAGS.estimator == "spectral":
            estimator = SpectralScoreEstimator(n_eigen=None, eta=None,
                                               n_eigen_threshold=0.99)
        else:
            raise ValueError("The chosen estimator is not recognized.")

        qzs = tf.transpose(qz_samples, [1, 0, 2])
        dlog_q = estimator.compute_gradients(qzs)
        entropy_surrogate = tf.reduce_mean(
            tf.reduce_sum(tf.stop_gradient(-dlog_q) * qzs, -1))
        cost = -eq_joint - entropy_surrogate
        grads_and_vars = optimizer.compute_gradients(cost)

        return grads_and_vars, eq_joint

    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                grads, tower_eq_joint = build_tower_graph(x, i)
                tower_losses.append([tower_eq_joint])
                tower_grads.append(grads)

    eq_joint = average_losses(tower_losses)[0]
    grads = average_gradients(tower_grads)
    infer_op = optimizer.apply_gradients(grads)

    # Generate images
    n_gen = 100
    _, _, x_logits = vae({}, n_gen, n_x, n_z, 1)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    # Define training parameters
    learning_rate = 1e-4
    epochs = 3000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_image_freq = 10
    save_model_freq = 100
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    result_path = "results/vae_conv_{}_{}".format(
        n_z, FLAGS.estimator) + time.strftime("_%Y%m%d_%H%M%S")

    saver = tf.train.Saver(max_to_keep=10)
    logger = setup_logger('vae_conv_' + FLAGS.estimator, __file__, result_path)

    with create_session(FLAGS.log_device_placement) as sess:
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
            eq_joints = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, eq_joint_ = sess.run(
                    [infer_op, eq_joint],
                    feed_dict={x_input: x_batch,
                               learning_rate_ph: learning_rate,
                               n_particles: n_est},
                )

                eq_joints.append(eq_joint_)

            time_epoch += time.time()
            logger.info(
                'Epoch {} ({:.1f}s): log joint = {}'
                .format(epoch, time_epoch, np.mean(eq_joints)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_eq_joints = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_eq_joint = sess.run(
                        eq_joint, feed_dict={x: test_x_batch,
                                             n_particles: n_est})
                    test_eq_joints.append(test_eq_joint)
                time_test += time.time()
                logger.info('>>> TEST ({:.1f}s)'.format(time_test))
                logger.info('>> Test log joint = {}'
                            .format(np.mean(test_eq_joints)))

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
