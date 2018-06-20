#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
import skimage

from utils import fid
from utils import setup_logger

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dir", "", """The result directory.""")


def main():
    logger = setup_logger("celeba_fid", __file__, FLAGS.dir,
                          filename="fid.log")

    # training set statistics
    stats_path = os.path.join('data', 'fid_stats_celeba.npz')

    # inception path
    inception_path = os.path.join('data', 'inception')

    # download the inception network
    inception_path = fid.check_or_download_inception(inception_path)

    # load pre-calculated training set statistics
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    print('mu_real', mu_real.shape)
    print('sigma_real', sigma_real.shape)
    f.close()

    print("FID:")
    # load the graph into the current TF graph
    fid.create_inception_graph(inception_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(5, 30, 5):
            # path to generated images
            gen_path = os.path.join(FLAGS.dir, 'gen.{}.npz'.format(epoch))
            # loads all images into memory (this might require a lot of RAM!)
            gen_data = np.load(gen_path)
            images = gen_data['images'].astype(np.float32)
            images = skimage.img_as_ubyte(images)
            print('gen:', images.shape)
            gen_data.close()

            mu_gen, sigma_gen = fid.calculate_activation_statistics(
                images, sess, batch_size=200, verbose=True)

            fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen,
                                                       mu_real, sigma_real)
            logger.info("{} {}".format(epoch, fid_value))


if __name__ == "__main__":
    main()
