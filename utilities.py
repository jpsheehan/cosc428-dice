#!/usr/bin/python

import numpy as np
import cv2 as cv
import math
import os
import os.path
import time
import random

CAMERA_SURFACE_FRONT = 0
CAMERA_SURFACE_BACK = 1
CAMERA_LOGITECH = 2


class Camera:
    """ Provides an interface to read undistorted camera frames. """

    def __init__(self, camera_id, camera_matrix_file='./camera_matrix.npy', distortion_coeff_file='./distortion_coeff.npy'):
        """ Initialises the camera with distortion information. """

        # open video capture device
        self.cap = cv.VideoCapture(camera_id)

        # open files containing distortion information
        self.camera_matrix = np.loadtxt(camera_matrix_file)
        self.distortion_coeff = np.loadtxt(distortion_coeff_file)

        # read a single frame to get the width and height
        _ret, frame = self.cap.read()
        self.frame_height, self.frame_width = frame.shape[:2]

        # create a new camera matrix and distortion
        self.new_camera_matrix, self.roi = cv.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeff, (self.frame_width, self.frame_height), 1, (self.frame_width, self.frame_height))
        _mapx, _mapy = cv.initUndistortRectifyMap(
            self.camera_matrix, self.distortion_coeff, None, self.new_camera_matrix, (self.frame_width, self.frame_height), cv.CV_16SC2)

    def __del__(self):
        """ Releases the camera. """
        self.cap.release()

    def read(self):
        """ Returns an undistorted and cropped frame from the camera. """
        # get a frame from the camera
        _ret, raw_img = self.cap.read()

        # undistort the frame
        processed_img = cv.undistort(raw_img, self.camera_matrix,
                                     self.distortion_coeff, None, self.new_camera_matrix)

        # crop the frame
        x, y, w, h = self.roi
        processed_img = processed_img[y:y+h, x:x+w]

        return processed_img


class Die:

    def __init__(self, img):
        self.img = img
        self.value = None

    def __str__(self):
        if self.value is None:
            return "?"
        else:
            return str(self.value)

    def compute_value(self):
        pass


def cropCopy(img, x, y, w, h):
    """ Returns a cropped copy of the image. """
    return img[y:y+h, x:x+w].copy()


def crop(img, x, y, w, h):
    """ Returns the cropped image. """
    return img[y:y+h, x:x+w]


def isolate_die_face(img, rect, angle):
    """ Isolates the die face from the rest of the image. """
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = rect

    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]

    # the width and height are the differences between the maximum and minimum coordinates
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)

    if w == 0 or h == 0:
        return None

    # crop out our unrotated die face
    rot_img = cropCopy(img, min(xs), min(ys), w, h)

    if rot_img.size == 0:
        return None

    # rotate the die face so it is the right way up
    rot_mat = cv.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1.0)
    rot_img = cv.warpAffine(rot_img, rot_mat, (w, h))

    # now the rotated die has some padding from the rotation
    # calculate the width of the padding and crop it out
    rot_w = int(math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
    rot_h = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    rot_img = crop(rot_img, int((w - rot_w) / 2),
                   int((h - rot_h) / 2), rot_w, rot_h)

    return rot_img


def save_die_face(img, folder='die_images'):
    """ Saves the die face to disk. """
    if not os.path.exists(folder):
        os.mkdir(folder)

    # generate a unique filename for the image
    filename = None
    while True:
        filename = str(int(time.time() * 1000)) + "_" + \
            str(random.randint(0, 10000)) + ".png"
        filename = os.path.join(folder, filename)
        if not os.path.exists(filename):
            break

    cv.imwrite(filename, img)

    return filename


def pipeline(img, *fns):
    """ Allows for chaining of image processing functions without intermediate variables. """
    intermediate = {}
    for fn in fns:
        intermediate[fn.__doc__] = img = fn(img)
        # img = fn(img)
    return img, intermediate
