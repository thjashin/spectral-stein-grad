#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np

from utils import setup_logger
from utils import dataset


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dir", "", """The result directory.""")


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load celebA
    data_path = os.path.join('data', 'celebA', 'img_align_celeba.zip')
    celeba = dataset.CelebADataset(data_path)

    # Define model parameters
    n_z = 32
    if "spectral" or "stein" in FLAGS.dir:
        from vae.vae_celeba_implicit import vae
    else:
        from vae.vae_celeba import vae

    # Define training/evaluation parameters
    n_gen = 100
    iters = 10000 // n_gen
    result_path = FLAGS.dir

    # Generate images
    ret = vae({}, n_gen, n_z, None)
    x_mean = ret[-1]
    x_gen = tf.reshape(x_mean, [-1] + celeba.data_dims)

    model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope="model")
    saver = tf.train.Saver(max_to_keep=10,
                           var_list=model_var_list)
    logger = setup_logger("vae_celeba_gen", __file__, result_path,
                          filename="gen.log")

    # Run the evaluation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(5, 30, 5):
            # Restore from the checkpoint
            ckpt_file = "vae.epoch.{}.ckpt".format(epoch)
            logger.info('Restoring model from {}...'.format(ckpt_file))
            saver.restore(sess, os.path.join(result_path, ckpt_file))

            # Generation
            logger.info('Start generation...')
            images = []
            for t in range(iters):
                time_iter = -time.time()
                image = sess.run(x_gen)
                time_iter += time.time()
                logger.info('Gen batch {} ({:.1f}s)'.format(t, time_iter))
                images.append(image)

            # Save to npz
            time_save = -time.time()
            images = np.vstack(images)
            np.savez_compressed(
                os.path.join(result_path, 'gen.{}.npz'.format(epoch)),
                images=images)
            time_save += time.time()

            logger.info('{} generations saved. ({:.1f}s)'
                        .format(images.shape[0], time_save))
