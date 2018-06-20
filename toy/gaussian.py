#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import zhusuan as zs
import numpy as np

from estimator import SteinScoreEstimator, SpectralScoreEstimator


q_mean = 0.
q_logstd = 0.
q_std = np.exp(q_logstd)
q_precision = 1. / q_std ** 2


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    from matplotlib import cm, colors, rc
    from matplotlib import pyplot as plt
    rc('text', usetex=True)

    # print(plt.style.available)
    import seaborn as sns
    sns.set()
    sns.set_color_codes()
    sns.set_style("white")

    eta = 1.
    n_eigen = 6
    M = 100
    lower_box = -5
    upper_box = 5

    q = zs.distributions.Normal(q_mean, logstd=q_logstd)
    # samples: [M]
    samples = q.sample(n_samples=M)
    # log_q_samples: [M]
    log_q_samples = q.log_prob(samples)
    # x: [N]
    x = tf.placeholder(tf.float32, shape=[None])
    # log_qx: [N]
    log_qx = q.log_prob(x)
    # true_dlog_qx: [N]
    true_dlog_qx = tf.map_fn(lambda i: tf.gradients(q.log_prob(i), i)[0], x)

    stein = SteinScoreEstimator(eta=eta)
    stein_dlog_q_samples = stein.compute_gradients(samples[..., None])
    # stein_dlog_q_samples: [M]
    stein_dlog_q_samples = tf.squeeze(stein_dlog_q_samples, -1)

    def stein_dlog(y):
        stein_dlog_qx = stein.compute_gradients(samples[..., None],
                                                x=y[..., None, None])
        # stein_dlog_qx: []
        stein_dlog_qx = tf.squeeze(stein_dlog_qx, axis=(-1, -2))
        return stein_dlog_qx

    # stein_dlog_qx: [N]
    stein_dlog_qx = tf.map_fn(stein_dlog, x)

    spectral = SpectralScoreEstimator(n_eigen=n_eigen)
    spectral_dlog_qx = spectral.compute_gradients(samples[..., None],
                                                  x=x[..., None])
    # spectral_dlog_qx: [N]
    spectral_dlog_qx = tf.squeeze(spectral_dlog_qx, -1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        xs = np.linspace(lower_box, upper_box, M)
        log_qxs, true_dlog_qxs, spectral_dlog_qxs, stein_dlog_qxs, \
            stein_dlog_basis, samples_ = \
                sess.run([log_qx, true_dlog_qx, spectral_dlog_qx, stein_dlog_qx,
                          stein_dlog_q_samples, samples],
            feed_dict={x: xs})

        plt.figure(figsize=(8, 6))
        plt.plot(xs, log_qxs, label=r"$\log q(x)$", linewidth=2., color='m')
        plt.plot(xs, true_dlog_qxs, label=r"$\nabla_x\log q(x)$", linewidth=2.)
        plt.plot(xs, spectral_dlog_qxs,
                 label=r"$\hat{\nabla}_x\log q(x)$, Spectral", linewidth=2.)
        plt.plot(xs, stein_dlog_qxs, "--",
                 label=r"$\hat{\nabla}_x\log q(x)$, Stein$^+$", linewidth=2.)

        sample_idx = np.argsort(samples_)
        samples_ = samples_[sample_idx]
        stein_dlog_basis = stein_dlog_basis[sample_idx]
        plt.scatter(samples_, stein_dlog_basis, marker='x', s=30, alpha=0.7,
                    label=r"$\hat{\nabla}_x\log q(x)$, Stein", color='r')
        plt.xlim(lower_box, upper_box)
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 1, 4, 3, 2]
        l = plt.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            fontsize=17, ncol=2, loc="lower center", columnspacing=0.5,
            borderaxespad=0.2,
        )
        plt.setp(l.texts, family='serif', usetex=True)
        plt.xlabel('x', fontsize=20)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)

        sns.despine()
        plt.show()
        # plt.savefig('gaussian.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
