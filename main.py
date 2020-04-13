# import utilities
from Gui import Gui, Param, Widget

import numpy as np
import cv2 as cv

from utilities import *
from colors import *

P_BLUR_KERNEL = "Kernel Size"

P_CANNY_THRESHOLD_1 = "Lower Threshold"
P_CANNY_THRESHOLD_2 = "Upper Threshold"

P_THRESHOLD_THRESHOLD = "Threshold"
P_THRESHOLD_MAX_VAL = "Max. Value"

P_DENOISE_KERNEL = "Kernel Size"

P_HOUGH_RHO = "Rho"
P_HOUGH_THRESHOLD = "Threshold"
P_HOUGH_MIN_LINE_LENGTH = "Min. Line Length"
P_HOUGH_MAX_LINE_GAP = "Max. Line Gap"

cam = Camera(CAMERA_LOGITECH)


def do_camera(_img, _params, _imgs, _state):
    return cam.read()


def do_greyscale(img, _params, _imgs, _state):
    """Greyscale"""
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def do_denoising(img, params, _imgs, _state):
    kernel_size = params[P_DENOISE_KERNEL]
    kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (kernel_size, kernel_size))
    img = cv.morphologyEx(img, cv.MORPH_ERODE, kernel)
    return img


def do_blur(img, params, _imgs, _state):
    """Blurred"""
    kernel = params[P_BLUR_KERNEL] * 2 + 1
    img = cv.GaussianBlur(img, (kernel, kernel), 0)
    # img = cv.medianBlur(img, kernel)
    # img = cv.bilateralFilter(img, kernel, 150, 150)
    return img


def do_threshold(img, params, _imgs, _state):
    """Threshold"""
    thresh = params[P_THRESHOLD_THRESHOLD]
    max_val = params[P_THRESHOLD_MAX_VAL]
    _ret, img = cv.threshold(img, thresh, max_val, cv.THRESH_BINARY)
    return img


def do_edges_canny(img, params, _imgs, _state):
    """Canny Edge Detection"""
    thresh1 = params[P_CANNY_THRESHOLD_1]
    thresh2 = params[P_CANNY_THRESHOLD_2]
    edges = cv.Canny(img, thresh1, thresh2)
    return edges


def do_display_edges(edges, _params, imgs, _state):
    img_annotated = imgs[0].copy()
    contours = get_raw_contours(edges)
    img_annotated = cv.drawContours(
        img_annotated, contours, -1, COLOR_GREEN, 3)
    return img_annotated


def do_edges_hough(img, params, _imgs, _state):
    """Hough Line Detection"""
    rho = params[P_HOUGH_RHO]
    thresh = params[P_HOUGH_THRESHOLD]
    min_line_length = params[P_HOUGH_MIN_LINE_LENGTH]
    max_line_gap = params[P_HOUGH_MAX_LINE_GAP]
    lines = cv.HoughLinesP(
        img, rho, np.pi/180, thresh, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def do_annotation(edges, _params, imgs, state):

    img_annotated = imgs[0].copy()

    contours = get_contours(edges)
    state["faces"] = []

    for i, contour in enumerate(contours):

        _, _, angle = rot_rect = cv.minAreaRect(contour)
        rot_rect = cv.boxPoints(rot_rect)
        rot_rect = np.int0(rot_rect)

        # TODO: FILTER ONLY SQUARE(ISH) RECTANGLES

        rect = cv.boundingRect(contour)

        face = isolate_die_face(imgs[0], rot_rect, angle)

        if face is None:
            continue

        img_annotated = cv.drawContours(
            img_annotated, [rot_rect], 0, COLOR_RED, 2)

        img_annotated = cv.putText(
            img_annotated, "ID: {}".format(i + 1), (rect[0] + rect[2] + 5, rect[1] + 5), cv.FONT_HERSHEY_PLAIN, 1, COLOR_GREEN)

        state["faces"].append(face)

    # print the number of dice found on the image
    noun = "dice"
    if len(state["faces"]) == 1:
        noun = "die"
    cv.putText(img_annotated, "Found {} {}!".format(str(len(state["faces"])), noun),
               (0, img_annotated.shape[0]), cv.FONT_HERSHEY_PLAIN, 3, COLOR_WHITE, 2)

    return img_annotated


def get_raw_contours(edges):
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_contours(edges):
    # find all the contours
    contours = get_raw_contours(edges)

    # filter out small contours outside of our expected size range
    contours = filter(lambda c: cv.contourArea(c) > 1000, contours)

    return contours


def key_handler(key, _imgs, state):
    """ Handles the key presses. """
    if key == ord('s') or key == ord('S'):
        print("Saving dice faces...")
        for i, face in enumerate(state.get("faces", [])):
            filename = save_die_face(face)
            print("Saved die #{} to \"{}\"".format(i + 1, filename))

        state["gui_paused"] = False


def main():
    """ Creates the GUI and runs the pipeline. """

    gui = Gui(key_handler=key_handler)

    widget_camera = Widget("Camera", do_camera, show_window=False)
    gui.widgets.append(widget_camera)

    widget_greyscale = Widget("Greyscale", do_greyscale, show_window=False)
    gui.widgets.append(widget_greyscale)

    # widget_denoising = Widget("Denoising", do_denoising)
    # widget_denoising.params.append(Param(P_DENOISE_KERNEL, 1, 20, 1))
    # gui.widgets.append(widget_denoising)

    widget_threshold = Widget("Threshold", do_threshold)
    widget_threshold.params.append(Param(P_THRESHOLD_THRESHOLD, 0, 255, 150))
    widget_threshold.params.append(Param(P_THRESHOLD_MAX_VAL, 0, 255, 255))
    gui.widgets.append(widget_threshold)

    widget_blur = Widget("Blur", do_blur)
    widget_blur.params.append(Param(P_BLUR_KERNEL, 0, 20, 1))
    gui.widgets.append(widget_blur)

    widget_edges_canny = Widget(
        "Canny Edge Detection", do_edges_canny, display_function=do_display_edges)
    widget_edges_canny.params.append(Param(P_CANNY_THRESHOLD_1, 1, 600, 100))
    widget_edges_canny.params.append(Param(P_CANNY_THRESHOLD_2, 1, 600, 400))
    gui.widgets.append(widget_edges_canny)

    # widget_edges_hough = Widget("Hough Line Detection", do_edges_hough, show_image=False)
    # widget_edges_hough.params.append(Param(P_HOUGH_RHO, 1, 10, 1))
    # widget_edges_hough.params.append(Param(P_HOUGH_THRESHOLD, 1, 500, 100))
    # widget_edges_hough.params.append(
    #     Param(P_HOUGH_MIN_LINE_LENGTH, 1, 300, 10))
    # widget_edges_hough.params.append(Param(P_HOUGH_MAX_LINE_GAP, 1, 300, 10))
    # gui.widgets.append(widget_edges_hough)

    widget_annotate = Widget("Annotated Camera Frame", do_annotation)
    gui.widgets.append(widget_annotate)

    gui.show()


if __name__ == "__main__":
    main()
