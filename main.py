import numpy as np
import cv2 as cv

from utilities import *
from colors import *


class OpenCVGui:

    def __init__(self):
        self.name = "Nice Dice Advice Device"

    def _nop(self):
        pass

    def run(self):
        cv.namedWindow(self.name)
        cv.createButton("Save Images", self.on_button_save)

    def on_button_save(self):
        pass


def greyscale(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def blur(img):
    # img = cv.GaussianBlur(img, (3, 3), 0)
    # img = cv.medianBlur(img, 3)
    img = cv.bilateralFilter(img, 3, 150, 150)
    return img


def threshold(img):
    # _ret, img = cv.threshold(img, 90, 128, cv.THRESH_BINARY)
    return img


def edges(img):
    img = cv.Canny(img, 100, 400)
    return img


def get_contours(img):
    # find all the contours
    contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # filter out small contours outside of our expected size range
    contours = filter(lambda c: cv.contourArea(
        c) > 1000 and cv.contourArea(c) < 3500, contours)

    return contours


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
        contours = get_contours(img_processed)

        die_faces = []

        for i, contour in enumerate(contours):

            _, _, angle = rot_rect = cv.minAreaRect(contour)
            rot_rect = cv.boxPoints(rot_rect)
            rot_rect = np.int0(rot_rect)

            rect = cv.boundingRect(contour)

            die_face = isolate_die_face(img, rot_rect, angle)

            if die_face is None:
                continue

            img_annotated = cv.drawContours(
                img_annotated, [rot_rect], 0, COLOR_RED, 2)

            img_annotated = cv.putText(
                img_annotated, "ID: {}".format(i + 1), (rect[0] + rect[2] + 5, rect[1] + 5), cv.FONT_HERSHEY_PLAIN, 1, COLOR_GREEN)

            die_faces.append(die_face)

        # print the number of dice found on the image
        noun = "dice"
        if len(die_faces) == 1:
            noun = "die"
        cv.putText(img_annotated, "Found {} {}!".format(str(len(die_faces)), noun),
                   (0, img_annotated.shape[0]), cv.FONT_HERSHEY_PLAIN, 3, COLOR_WHITE, 2)

        cv.imshow('Processed', img_processed)
        cv.imshow('Original with Contours', img_annotated)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
        elif cv.waitKey(1) & 0xff == ord(' '):
            for die_face in die_faces:
                save_die_face(die_face)
            print("Saved {} images".format(str(len(die_faces))))

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
