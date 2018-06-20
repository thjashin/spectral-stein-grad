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

from estimator import SteinScoreEstimator, SpectralScoreEstimator, \
    entropy_gradients
from utils.utils import save_image_collections, setup_logger

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("estimator", "spectral", """The KL term estimator.""")


# Define model parameters
n_z = 32
ngf = 64


# Estimator parameters
n_est = 100
eta = 0.001


@zs.reuse('model')
def vae(observed, n, n_z, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, std=1., n_samples=n_particles,
                      group_ndims=1)
        lx_z = layers.fully_connected(z, num_outputs=ngf * 8 * 4 * 4,
                                      activation_fn=None)
        lx_z = tf.reshape(lx_z, [-1, 4, 4, ngf * 8])
        lx_z = layers.conv2d_transpose(lx_z, ngf * 4, 5, stride=2)
        lx_z = layers.conv2d_transpose(lx_z, ngf * 2, 5, stride=2)
        lx_z = layers.conv2d_transpose(lx_z, ngf, 5, stride=2)
        lx_z = layers.conv2d_transpose(lx_z, 3, 5, stride=2,
                                       activation_fn=tf.nn.sigmoid)
        x_mean = tf.reshape(lx_z, [-1, n, 64, 64, 3])
        x_logstd = tf.get_variable(
            "h_logstd", shape=[64, 64, 3], dtype=tf.float32,
            initializer=tf.constant_initializer(0.))
        x = zs.Normal('x', x_mean, logstd=x_logstd, group_ndims=3)
    return model, z, x_mean


@zs.reuse('variational')
def implicit(x, n_z, n_particles):
    with zs.BayesianNet() as variational:
        lz_x = layers.conv2d(x, ngf * 1, 5, stride=2)
        lz_x = layers.conv2d(lz_x, ngf * 2, 5, stride=2)
        lz_x = layers.conv2d(lz_x, ngf * 4, 5, stride=2)
        lz_x = layers.conv2d(lz_x, ngf * 8, 5, stride=2)
        h_mean = layers.flatten(lz_x)
        h_logstd = tf.get_variable(
            "h_logstd", shape=[4 * 4 * ngf * 8], dtype=tf.float32,
            initializer=tf.constant_initializer(0.))
        h = zs.Normal('h', h_mean, logstd=h_logstd, group_ndims=1,
                      n_samples=n_particles)
        lz_x = layers.fully_connected(h, 500)
        z = layers.fully_connected(lz_x, n_z, activation_fn=None)
    return z


@zs.reuse('variational')
def mean_field(x, n_z, n_particles):
    with zs.BayesianNet() as variational:
        lz_x = layers.conv2d(x, ngf * 1, 5, stride=2)
        lz_x = layers.conv2d(lz_x, ngf * 2, 5, stride=2)
        lz_x = layers.conv2d(lz_x, ngf * 4, 5, stride=2)
        lz_x = layers.conv2d(lz_x, ngf * 8, 5, stride=2)
        lz_x = layers.flatten(lz_x)
        lz_x = layers.fully_connected(lz_x, 500)
        z_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', z_mean, logstd=z_logstd, group_ndims=1,
                      n_samples=n_particles)
    return z


q_net = implicit


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load celebA
    data_path = os.path.join('data', 'celebA', 'img_align_celeba.zip')
    celeba = dataset.CelebADataset(data_path)

    x = tf.placeholder(tf.float32, shape=[None] + celeba.data_dims, name='x')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    n = tf.shape(x)[0]

    qz_samples = q_net(x, n_z, n_particles)
    # Use a single particle for the reconstruction term
    observed = {'x': x, 'z': qz_samples[:1]}
    model, z, _ = vae(observed, n, n_z, 1)
    # [1, n]
    log_px_qz = model.local_log_prob('x')
    eq_ll = tf.reduce_mean(log_px_qz)
    # [n_particles, n]
    log_p_qz = z.log_prob(qz_samples)
    eq_joint = eq_ll + tf.reduce_mean(log_p_qz)

    if FLAGS.estimator == "stein":
        estimator = SteinScoreEstimator(eta=eta)
    elif FLAGS.estimator == "spectral":
        estimator = SpectralScoreEstimator(n_eigen=None, eta=None,
                                           n_eigen_threshold=0.99)
    else:
        raise ValueError("The chosen estimator is not recognized.")

    optimizer = tf.train.AdamOptimizer(3e-4)

    entropy_grads_and_vars = entropy_gradients(
        optimizer, estimator, tf.transpose(qz_samples, [1, 0, 2]))
    entropy_dict = dict([(v, g) for g, v in entropy_grads_and_vars
                         if g is not None])

    joint_grads_and_vars = optimizer.compute_gradients(-eq_joint)
    joint_dict = dict([(v, g) for g, v in joint_grads_and_vars
                       if g is not None])

    def combine_grads(v):
        ret = 0.
        if v in entropy_dict:
            ret += -entropy_dict[v]
        if v in joint_dict:
            ret += joint_dict[v]
        return ret

    grads_and_vars = [(combine_grads(v), v) for v in tf.trainable_variables()]
    infer_op = optimizer.apply_gradients(grads_and_vars)

    # Generate images
    n_gen = 100
    _, _, x_mean = vae({}, n_gen, n_z, None)
    x_gen = tf.reshape(x_mean, [-1] + celeba.data_dims)

    # Interpolation
    # [n, n_z]
    x_start = x[:8]
    x_end = x[8:16]
    z_start = qz_samples[0, :8, :]
    z_end = qz_samples[0, 8:16, :]
    # [1, 8, 1]
    alpha = tf.reshape(tf.linspace(0., 1., 8), [1, 8, 1])
    # [n, 1, n_z]
    z_start = tf.expand_dims(z_start, 1)
    z_end = tf.expand_dims(z_end, 1)
    # [n, 8, n_z]
    z_interp = alpha * z_start + (1. - alpha) * z_end
    z_interp = tf.reshape(z_interp, [-1, n_z])
    _, _, x_interp = vae({'z': z_interp}, 64, n_z, None)
    x_interp = tf.reshape(x_interp, [-1] + celeba.data_dims)

    # Define training parameters
    epochs = 25
    batch_size = 64
    iters = celeba.train_size // batch_size
    save_image_freq = 1
    print_freq = 100
    save_model_freq = 5
    test_freq = 1
    test_batch_size = 500
    test_iters = celeba.test_size // test_batch_size
    result_path = "results/vae_celeba_" + FLAGS.estimator + \
        time.strftime("_%Y%m%d_%H%M%S")

    saver = tf.train.Saver(max_to_keep=10)
    logger = setup_logger('vae_celeba_' + FLAGS.estimator, __file__,
                          result_path)

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
            eq_joints = []
            time_iters = []
            for t in range(1, iters + 1):
                time_iter = -time.time()
                x_batch = celeba.next_batch(batch_size)
                _, eq_joint_ = sess.run(
                    [infer_op, eq_joint],
                    feed_dict={x: x_batch, n_particles: n_est})
                # logger.info('Iter {}: lb = {}, kl = {}, true_kl = {}'
                #             .format(t, lb, kl, true_kl))
                eq_joints.append(eq_joint_)
                time_iter += time.time()
                time_iters.append(time_iter)

                if t % print_freq == 0:
                    logger.info(
                        'Epoch={} Iter={} ({}s): log joint = {}'
                        .format(epoch, t, np.mean(time_iters),
                                np.mean(eq_joints[-print_freq:])))

            logger.info(
                'Epoch {}: log joint = {}'
                .format(epoch, np.mean(eq_joints)))

            interp_images = []
            start_images = []
            end_images = []
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_eq_joints = []
                for t in range(test_iters):
                    test_x_batch = celeba.next_test_batch(test_batch_size)
                    test_eq_joint, interp_image, start_image, end_image = \
                        sess.run(
                            [eq_joint, x_interp, x_start, x_end],
                            feed_dict={x: test_x_batch, n_particles: n_est})
                    test_eq_joints.append(test_eq_joint)
                    interp_images.append(interp_image)
                    start_images.append(start_image)
                    end_images.append(end_image)
                time_test += time.time()
                logger.info('>>> TEST ({:.1f}s)'.format(time_test))
                logger.info('>> Test log joint = {}'
                            .format(np.mean(test_eq_joints)))

                logger.info('Saving interpolations...')
                interp_name = os.path.join(result_path,
                                           "interp.epoch.{}.png".format(epoch))
                save_image_collections(interp_images[0], interp_name,
                                       scale_each=True, shape=(8, 8))
                if epoch == 1:
                    save_image_collections(
                        start_images[0], interp_name + ".start.png",
                        scale_each=True, shape=(8, 1))
                    save_image_collections(
                        end_images[0], interp_name + ".end.png",
                        scale_each=True, shape=(8, 1))

            if epoch % save_image_freq == 0:
                logger.info('Saving images...')
                images = sess.run(x_gen)
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name, scale_each=True)

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
