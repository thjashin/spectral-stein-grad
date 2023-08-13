#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip
import zipfile
import tarfile

import numpy as np
import skimage
from skimage import io, transform
import six
from six.moves import urllib, range
from six.moves import cPickle as pickle


def standardize(data_train, data_test):
    """
    Standardize a dataset to have zero mean and unit standard deviation.

    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.

    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.

    :param x: 1-D Numpy array of type int.
    :param depth: A int.

    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def load_mnist_realval(path, one_hot=True, dequantify=False):
    """
    Loads the real valued MNIST dataset.

    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).

    :return: The MNIST dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
        x_test, t_transform(t_test)


def load_uci_datasets(path, rng, delimiter=None, dtype=np.float32):
    data = np.loadtxt(path, delimiter=delimiter)
    data = data.astype(dtype)
    permutation = rng.choice(np.arange(data.shape[0]),
                             data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_val = permutation[size_train: size_test]
    index_test = permutation[size_test:]

    # np.savetxt("index_train.txt", index_train, fmt='%d')
    # np.savetxt("index_test.txt", index_test, fmt='%d')

    X_train, y_train = data[index_train, :-1], data[index_train, -1]
    X_val, y_val = data[index_val, :-1], data[index_val, -1]
    X_test, y_test = data[index_test, :-1], data[index_test, -1]

    return X_train, y_train, X_val, y_val, X_test, y_test


class CelebADataset(object):
    def __init__(self, path="data/celebA/img_align_celeba.zip", crop=True):
        self.f = zipfile.ZipFile(path, 'r')
        self.data_files = [i for i in self.f.namelist() if i[-1] != '/']
        if len(self.data_files) < 100000:
            print("Only %d images found for celebA, is this right?" % len(
                self.data_files))
            exit(-1)
        self.train_size = int(np.floor(len(self.data_files) * 0.8))
        self.test_size = len(self.data_files) - self.train_size
        self.train_img = self.data_files[:self.train_size]
        self.test_img = self.data_files[self.train_size:]

        self.train_idx = 0
        self.test_idx = 0
        self.data_dims = [64, 64, 3]

        self.train_cache = np.ndarray((self.train_size, 64, 64, 3),
                                      dtype=np.float32)
        self.train_cache_top = 0
        self.test_cache = np.ndarray((self.test_size, 64, 64, 3),
                                     dtype=np.float32)
        self.test_cache_top = 0
        self.range = [-1.0, 1.0]
        self.is_crop = crop
        self.name = "celebA"

    """ Return [batch_size, 64, 64, 3] data array """
    def next_batch(self, batch_size):
        # sample_files = self.data[0:batch_size]
        prev_idx = self.train_idx
        self.train_idx += batch_size
        if self.train_idx > self.train_size:
            self.train_idx = batch_size
            prev_idx = 0

        if self.train_idx < self.train_cache_top:
            return self.train_cache[prev_idx:self.train_idx, :, :, :]
        else:
            sample_files = self.train_img[prev_idx:self.train_idx]
            sample = [self.get_image(sample_file, self.is_crop)
                      for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            self.train_cache[prev_idx:self.train_idx] = sample_images
            self.train_cache_top = self.train_idx
            return sample_images

    def next_test_batch(self, batch_size):
        prev_idx = self.test_idx
        self.test_idx += batch_size
        if self.test_idx > self.test_size:
            self.test_idx = batch_size
            prev_idx = 0

        if self.test_idx < self.test_cache_top:
            return self.test_cache[prev_idx:self.test_idx, :, :, :]
        else:
            sample_files = self.test_img[prev_idx:self.test_idx]
            sample = [self.get_image(sample_file, self.is_crop)
                      for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            self.test_cache[prev_idx:self.test_idx] = sample_images
            self.test_cache_top = self.test_idx
            return sample_images

    def batch_by_index(self, batch_start, batch_end):
        sample_files = self.data_files[batch_start:batch_end]
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def get_image(self, image_path, is_crop=True):
        file = self.f.open(image_path)
        raw = skimage.img_as_float(io.imread(file))
        image = CelebADataset.transform(raw, is_crop=is_crop)
        return image

    @staticmethod
    def center_crop(x, crop_h, crop_w=None, resize_w=64):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        return transform.resize(x[j:j + crop_h, i:i + crop_w],
                                [resize_w, resize_w])

    @staticmethod
    def full_crop(x):
        if x.shape[0] <= x.shape[1]:
            lb = int((x.shape[1] - x.shape[0]) / 2)
            ub = lb + x.shape[0]
            x = transform.resize(x[:, lb:ub], [64, 64])
        else:
            lb = int((x.shape[0] - x.shape[1]) / 2)
            ub = lb + x.shape[1]
            x = transform.resize(x[lb:ub, :], [64, 64])
        return x

    @staticmethod
    def transform(image, npx=148, is_crop=True, resize_w=64):
        # npx : # of pixels width/height of image
        if is_crop:
            cropped_image = CelebADataset.center_crop(image, npx,
                                                      resize_w=resize_w)
        else:
            cropped_image = CelebADataset.full_crop(image)
        return cropped_image

    def reset(self):
        self.idx = 0

def load_cifar10(path, normalize=True, dequantify=False, one_hot=True):
    """
    Loads the cifar10 dataset.

    :param path: Path to the dataset file.
    :param normalize: Whether to normalize the x data to the range [0, 1].
    :param dequantify: Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :param one_hot: Whether to use one-hot representation for the labels.

    :return: The cifar10 dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset(
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', path)

    data_dir = os.path.dirname(path)
    batch_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.isfile(os.path.join(batch_dir, 'data_batch_5')):
        with tarfile.open(path) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, data_dir)

    train_x, train_y = [], []
    for i in range(1, 6):
        batch_file = os.path.join(batch_dir, 'data_batch_' + str(i))
        with open(batch_file, 'rb') as f:
            if six.PY2:
                data = pickle.load(f)
            else:
                data = pickle.load(f, encoding='latin1')
            train_x.append(data['data'])
            train_y.append(data['labels'])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    test_batch_file = os.path.join(batch_dir, 'test_batch')
    with open(test_batch_file, 'rb') as f:
        if six.PY2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
        test_x = data['data']
        test_y = np.asarray(data['labels'])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    if dequantify:
        train_x += np.random.uniform(0, 1,
                                     size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0, 1, size=test_x.shape).astype('float32')
    if normalize:
        train_x = train_x / 256
        test_x = test_x / 256

    train_x = train_x.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_x = test_x.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    t_transform = (lambda x: to_one_hot(x, 10)) if one_hot else (lambda x: x)
    return train_x, t_transform(train_y), test_x, t_transform(test_y)
