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
from zhusuan.evaluation import AIS

from utils.utils import setup_logger

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dir", "", """The result directory.""")
tf.flags.DEFINE_integer("seed", 1234, """Random seed.""")


if __name__ == "__main__":
    seed = FLAGS.seed
    tf.set_random_seed(seed)

    # Load MNIST
    data_path = os.path.join('data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    np.random.seed(seed)
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    test_idx = np.arange(x_test.shape[0])
    np.random.shuffle(test_idx)
    x_test = x_test[test_idx[:2048]]
    n_x = x_test.shape[1]

    # Define model parameters
    from .vae_conv import vae
    n_z = int(FLAGS.dir.split('_')[2])

    # Define training/evaluation parameters
    if "vae_conv_" in FLAGS.dir:
        test_batch_size = 256
    elif "vae_" in FLAGS.dir:
        test_batch_size = 512
    test_iters = x_test.shape[0] // test_batch_size
    test_n_temperatures = 1000
    test_n_leapfrogs = 10
    test_n_chains = 5
    result_path = FLAGS.dir

    # Build the computation graph
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [test_n_chains, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        ret = vae(observed, n, n_x, n_z, test_n_chains)
        model = ret[0]
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    def ais_log_prior(observed):
        z = observed['z']
        ret = vae({'z': z}, n, n_x, n_z, test_n_chains)
        model = ret[0]
        return model.local_log_prob('z')

    ret = vae({}, n, n_x, n_z, test_n_chains)
    model = ret[0]
    pz_samples = model.outputs('z')
    z = tf.Variable(tf.zeros([test_n_chains, test_batch_size, n_z]),
                    name="z", trainable=False)
    hmc = zs.HMC(step_size=1e-6, n_leapfrogs=test_n_leapfrogs,
                 adapt_step_size=True, target_acceptance_rate=0.65,
                 adapt_mass=True)
    ais = AIS(ais_log_prior, log_joint, {'z': pz_samples}, hmc,
              observed={'x': x_obs}, latent={'z': z},
              n_chains=test_n_chains, n_temperatures=test_n_temperatures)

    model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope="model")
    variational_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope="variational")
    saver = tf.train.Saver(max_to_keep=10,
                           var_list=model_var_list + variational_var_list)
    logger = setup_logger("vae_eval", __file__, result_path,
                          filename="eval.log.{}".format(seed))

    # Run the evaluation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        if ckpt_file is not None:
            logger.info('Restoring model from {}...'.format(ckpt_file))
            saver.restore(sess, ckpt_file)

            # AIS evaluation
            logger.info('Start evaluation...')
            time_ais = -time.time()
            test_ll_lbs = []
            for t in range(test_iters):
                time_iter = -time.time()
                test_x_batch = x_test[t * test_batch_size:
                                      (t + 1) * test_batch_size]
                ll_lb = ais.run(sess, feed_dict={x: test_x_batch})
                time_iter += time.time()
                logger.info('Test batch {} ({:.1f}s): AIS lower bound = {}'
                            .format(t, time_iter, ll_lb))
                test_ll_lbs.append(ll_lb)
            time_ais += time.time()
            test_ll_lb = np.mean(test_ll_lbs)
            logger.info(
                '>> Test log likelihood ({:.1f}s)\n'
                '>> AIS lower bound = {}'.format(time_ais, test_ll_lb))
