# import utilities
from Gui import Gui, Param, Widget

import numpy as np
import cv2 as cv
import tensorflow as tf

from train import load_model
from utilities import *
from colors import *

P_BLUR_KERNEL = "Kernel Size"

P_CANNY_THRESHOLD_1 = "Lower Threshold"
P_CANNY_THRESHOLD_2 = "Upper Threshold"

P_THRESHOLD_BLOCK_SIZE = "Block Size"
P_THRESHOLD_MAX_VAL = "Max. Value"
P_THRESHOLD_CONSTANT = "Constant"

P_DENOISE_KERNEL = "Kernel Size"

P_HOUGH_RHO = "Rho"
P_HOUGH_THRESHOLD = "Threshold"
P_HOUGH_MIN_LINE_LENGTH = "Min. Line Length"
P_HOUGH_MAX_LINE_GAP = "Max. Line Gap"

cam = Camera(0)
# cam = Image("captures/04.jpg")

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
    img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
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
    block_size = params[P_THRESHOLD_BLOCK_SIZE] * 2 + 1
    max_val = params[P_THRESHOLD_MAX_VAL]
    constant = params[P_THRESHOLD_CONSTANT]
    _, img = cv.threshold(img, max_val, 255, cv.THRESH_BINARY)
    # img = cv.adaptiveThreshold(img, max_val, cv.ADAPTIVE_THRESH_MEAN_C,
    #         cv.THRESH_BINARY, block_size, constant)
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

def get_edge_length(p1, p2):
    """ Get the squared edge length between two points. """
    return np.sqrt(abs(p2[1] - p1[1]) ** 2 + abs(p2[0] - p1[0]) ** 2)

def is_square(rot_rect, error):
    """ Returns true if the rotated rectangle is square (within error percent). """
    [p1, p2, p3, p4] = rot_rect
    e1 = get_edge_length(p1, p2)
    e2 = get_edge_length(p2, p3)
    diff = abs(e1 - e2) / max(e1, e2)
    return diff <= error


def do_annotation(edges, _params, imgs, state):

    img_annotated = imgs[0].copy()

    contours = get_contours(edges)
    state["faces"] = []

    for i, contour in enumerate(contours):

        _, _, angle = rot_rect = cv.minAreaRect(contour)
        rot_rect = cv.boxPoints(rot_rect)
        rot_rect = np.int0(rot_rect)

        # check if the rectangle is a square +/- 20%
        if not is_square(rot_rect, 0.20):
            continue

        rect = cv.boundingRect(contour)

        face = isolate_die_face(imgs[0], rot_rect, angle)

        if face is None:
            continue

        estimated_value, probability = get_prob(face, state["model"])

        # if the probability is too low, we cannot be sure
#         if probability < 1000:
 #            continue

        img_annotated = cv.drawContours(
            img_annotated, [rot_rect], 0, COLOR_RED, 2)

        # img_annotated = cv.putText(
        #     img_annotated, "{} ({}%)".format(estimated_value, probability), (rect[0] + rect[2] + 5, rect[1] + 5), cv.FONT_HERSHEY_PLAIN, 3, COLOR_GREEN, thickness=3)

        img_annotated = cv.putText(
            img_annotated, "{}".format(estimated_value), (rect[0] + rect[2] + 5, rect[1] + 5), cv.FONT_HERSHEY_PLAIN, 3, COLOR_GREEN, thickness=3)

        state["faces"].append(face)

    # print the number of dice found on the image
    noun = "dice"
    if len(state["faces"]) == 1:
        noun = "die"
    # cv.putText(img_annotated, "Found {} {}!".format(str(len(state["faces"])), noun),
               # (0, img_annotated.shape[0]), cv.FONT_HERSHEY_PLAIN, 3, COLOR_WHITE, 2)

    return img_annotated


def get_raw_contours(edges):
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_contours(edges):
    # find all the contours
    contours = get_raw_contours(edges)

    # filter out small contours outside of our expected size range
    contours = filter(lambda c: cv.contourArea(c) > 200, contours)

    return contours


def key_handler(key, _imgs, state):
    """ Handles the key presses. """
    if key == ord('s') or key == ord('S'):
        print("Saving dice faces...")
        for i, face in enumerate(state.get("faces", [])):
            filename = save_die_face(face)
            print("Saved die #{} to \"{}\"".format(i + 1, filename))

        state["gui_paused"] = False


def get_prob(img, model):
    img = do_greyscale(img, None, None, None)
    img = cv.resize(img, IMG_SIZE)
    probs = model(np.array([img])).numpy()[0]
    m = None
    v = None
    for i, p in enumerate(probs):
        if m is None or p > m:
            m = p
            v = i + 1
    return v, m


def main():
    """ Creates the GUI and runs the pipeline. """

    gui = Gui(key_handler=key_handler)
    gui.state["model"] = load_model()

    widget_camera = Widget("Camera", do_camera, show_window=True, show_controls=False)
    gui.widgets.append(widget_camera)

    widget_greyscale = Widget("Greyscale", do_greyscale, show_window=False, show_controls=False)
    gui.widgets.append(widget_greyscale)

    # widget_denoising = Widget("Denoising", do_denoising)
    # widget_denoising.params.append(Param(P_DENOISE_KERNEL, 1, 20, 1))
    # gui.widgets.append(widget_denoising)

    widget_threshold = Widget("Threshold", do_threshold, show_controls=False)
    widget_threshold.params.append(Param(P_THRESHOLD_BLOCK_SIZE, 0, 50, 5))
    widget_threshold.params.append(Param(P_THRESHOLD_MAX_VAL, 0, 255, 160))
    widget_threshold.params.append(Param(P_THRESHOLD_CONSTANT, -20, 20, 0))
    gui.widgets.append(widget_threshold)

    widget_blur = Widget("Blur", do_blur, show_controls=False)
    widget_blur.params.append(Param(P_BLUR_KERNEL, 0, 20, 2))
    gui.widgets.append(widget_blur)

    widget_edges_canny = Widget(
        "Canny Edge Detection", do_edges_canny, display_function=do_display_edges, show_controls=False)
    widget_edges_canny.params.append(Param(P_CANNY_THRESHOLD_1, 1, 600, 110))
    widget_edges_canny.params.append(Param(P_CANNY_THRESHOLD_2, 1, 600, 320))
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
