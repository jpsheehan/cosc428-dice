""" Contains functions for loading model data. """

import os
import tensorflow as tf
import numpy as np
import cv2

DIR_TRAINING_DATA = "./training_data/"
DIR_TEST_DATA = "./test_data/"


def get_label(filename):
    """ Returns the zero-based label. """
    return int(os.path.basename(filename).split("_")[0]) - 1


def is_valid_file(filename):
    """ Returns true if the file contains a valid image. """
    return len(os.path.basename(filename).split("_")) == 3


def get_image(filename):
    """ Gets a normalised greyscale image from a file. """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    return img


def get_filenames(dirname):
    """ Returns a shuffled list of valid filenames. """
    filenames = os.listdir(dirname)
    filenames = filter(is_valid_file, filenames)
    filenames = map(lambda filename: os.path.join(
        dirname, filename), filenames)
    filenames = list(filenames)
    np.random.shuffle(filenames)
    return filenames


def load_data(dirname):
    """ Loads image and label data from a directory. """
    filenames = get_filenames(dirname)

    images = np.array([get_image(filename) for filename in filenames])

    labels = tf.keras.utils.to_categorical(
        np.array([get_label(filename) for filename in filenames]))

    return images, labels


def load_training_data():
    """ Returns training images and labels. """
    return load_data(DIR_TRAINING_DATA)


def load_test_data():
    """ Returns the test images and labels. """
    return load_data(DIR_TEST_DATA)
