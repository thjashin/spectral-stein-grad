#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .base import ScoreEstimator


class SteinScoreEstimator(ScoreEstimator):
    def __init__(self, eta):
        self._eta = eta
        super(SteinScoreEstimator, self).__init__()

    def compute_gradients(self, samples, x=None):
        # samples: [..., M, x_dim]
        # x: [..., 1, x_dim]
        M = tf.shape(samples)[-2]
        # kernel_width: [...]
        kernel_width = self.heuristic_kernel_width(samples, samples)
        # K: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        K, grad_K1, grad_K2 = self.grad_gram(samples, samples,
                                             kernel_width)
        # K_inv: [..., M, M]
        Kinv = tf.matrix_inverse(K + self._eta * tf.eye(M))
        # H_dh: [..., M, x_dim]
        H_dh = tf.reduce_sum(grad_K2, axis=-2)
        # grads: [..., M, x_dim]
        grads = - tf.matmul(Kinv, H_dh)
        if x is None:
            return grads
        else:
            assert_single_x = tf.assert_equal(
                tf.shape(x)[-2], 1,
                message="Only support single-particle out-of-sample extension.")
            with tf.control_dependencies([assert_single_x]):
                # Kxx: [..., 1, 1]
                Kxx = self.gram(x, x, kernel_width)
            # Kxq: [..., 1, M]
            Kxq = self.gram(x, samples, kernel_width)
            # Kxq @ K_inv: [..., 1, M]
            KxqKinv = tf.matmul(Kxq, Kinv)
            # term1: [..., 1, 1]
            term1 = -1. / (Kxx + self._eta -
                           tf.matmul(KxqKinv, Kxq, transpose_b=True))
            # grad_Kqx2: [..., M, 1, x_dim]
            Kqx, grad_Kqx1, grad_Kqx2 = self.grad_gram(samples, x, kernel_width)
            # term2: [..., 1, x_dim]
            term2 = tf.matmul(Kxq, grads) - tf.matmul(KxqKinv + 1.,
                                                      tf.squeeze(grad_Kqx2, -2))
            # ret: [..., 1, x_dim]
            return tf.matmul(term1, term2)
