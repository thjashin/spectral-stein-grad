#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
from utils.utils import setup_logger

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dir", "", """The result directory.""")


def main():
    logger = setup_logger("eval_stats", __file__, FLAGS.dir,
                          filename="summary.log")
    lbs = []
    for seed in range(1234, 1234 + 10):
        filename = os.path.join(FLAGS.dir, "eval.log.{}".format(seed))
        with open(filename, "rb") as f:
            text = f.read().decode("utf-8")
            lb = float(text.strip().split("\n")[-1].split("=")[-1].strip())
            logger.info(str(lb))
            lbs.append(lb)

    logger.info("{}+-{}".format(np.mean(lbs), np.std(lbs)))


if __name__ == "__main__":
    main()
