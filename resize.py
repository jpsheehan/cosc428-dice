import cv2
import os

import utilities
import main as main_file

DIR_NAME = "./die_images/"


def main():
    """ Resizes the images and converts them to grayscale. """
    for filename in os.listdir(DIR_NAME):
        path = os.path.join(DIR_NAME, filename)
        img = cv2.imread(path)

        w = None
        h = None
        if len(img.shape) == 3:
            w, h, _ = img.shape
        else:
            w, h = img.shape

        changed = False
        if w != utilities.IMG_SIZE[0] or h != utilities.IMG_SIZE[1]:
            img = cv2.resize(img, utilities.IMG_SIZE)
            changed = True

        # if len(img.shape) != 2:
        #     img = main_file.do_greyscale(img, None, None, None)
        #     changed = True

        if changed:
            cv2.imwrite(path, img)
            print("altered", path)


if __name__ == "__main__":
    main()
