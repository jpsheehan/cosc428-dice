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

    if len(rot_img) == 0:
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


def pipeline(img, *fns):
    for fn in fns:
        img = fn(img)
    return img


def greyscale(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def blur(img):
    # img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.medianBlur(img, 5)
    # img = cv.bilateralFilter(img, 3, 150, 150)
    return img


def threshold(img):
    _ret, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    return img


def edges(img):
    img = cv.Canny(img, 200, 400)
    return img


def main():
    cam = Camera(CAMERA_LOGITECH)

    while True:

        # read image
        img = cam.read()

        # copy the original image for annotation
        img_annotated = img.copy()

        # process the image
        img_processed = pipeline(img, greyscale, threshold, blur, edges)

        # detect contours
        contours, _ = cv.findContours(
            img_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # filter out small contours
        contours = list(filter(lambda c: cv.contourArea(c) > 100, contours))

        die_faces = []

        # filter contours by area
        contours = filter(lambda c: cv.contourArea(c) > 500, contours)

        for i, contour in enumerate(contours):

            # print(i, cv.contourArea(contour))

            _, _, angle = rot_rect = cv.minAreaRect(contour)
            rot_rect = cv.boxPoints(rot_rect)
            rot_rect = np.int0(rot_rect)

            rect = cv.boundingRect(contour)

            die_face = isolate_die_face(img, rot_rect, angle)

            if die_face is None:
                continue

            img_annotated = cv.drawContours(
                img_annotated, [rot_rect], 0, (255, 0, 0), 2)

            img_annotated = cv.putText(
                img_annotated, "ID: {}".format(i + 1), (rect[0] + rect[2] + 5, rect[1] + 5), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            die_faces.append(die_face)

            # cv.imshow(str(i), die_face)

        cv.imshow('Processed', img_processed)
        cv.imshow('Original with Contours', img_annotated)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
        elif cv.waitKey(1) & 0xff == ord(' '):
            for die_face in die_faces:
                save_die_face(die_face)
            # cv.displayOverlay("Original with Contours", "Saved {} images".format(
            #     str(len(die_faces))), 2000)
            print("Saved {} images".format(str(len(die_faces))))

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
