import numpy as np
import cv2 as cv
import math

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
        self.mapx, self.mapy = cv.initUndistortRectifyMap(
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


def get_contour_bounds(contour):
    min_x = (None, None)
    max_x = (None, None)
    min_y = (None, None)
    max_y = (None, None)
    for [[x, y]] in contour:
        if min_x[0] is None or x < min_x[0]:
            min_x = (x, y)
        if min_y[1] is None or y < min_y[1]:
            min_y = (x, y)
        if max_x[0] is None or x > max_x[0]:
            max_x = (x, y)
        if max_y[1] is None or y > max_y[1]:
            max_y = (x, y)

    # take the of the maximum length between two adjacent points
    lengths = [(min_y[0] - min_x[0], min_y[1] - min_x[1]), (max_x[0] - min_y[0], max_x[1] - min_y[1]),
               (max_y[0] - max_x[0], max_y[1] - max_x[1]), (min_x[0] - max_y[0], min_x[1] - max_y[1])]

    lengths = list(map(lambda p: p[0] * p[0] + p[1] * p[1], lengths))
    max_length_squared = max(lengths)
    max_length_index = lengths.index(max_length_squared)
    max_length = math.sqrt(max_length_squared)

    a = None
    b = None

    if max_length_index == 0:
        a = min_y
        b = min_x
    elif max_length_index == 1:
        a = max_x
        b = min_y
    elif max_length_index == 2:
        a = max_y
        b = max_x
    elif max_length_index == 3:
        a = min_x
        b = max_y
    else:
        assert False

    # the angle between a and b gives us the angle of the dice relative to the x axis
    angle = math.atan((a[1] - b[1]) / (a[0] - b[0]))
    angle_deg = abs(math.degrees(angle))

    c = (int(math.cos(-angle) * max_length +
             a[0]), int(math.sin(-angle) * max_length + a[1]))

    d = (int(math.cos(-angle) * max_length +
             b[0]), int(math.sin(-angle) * max_length + b[1]))

    return (min_x, min_y, max_x, max_y, angle_deg)


def get_die_value(img, rect):
    """ Returns the value of the die. """

    # copy and crop
    try:
        x, y, w, h = rect
        die = img.copy()
        die = die[y:y+h, x:x+w]
        # print(rect)

        circles = cv.HoughCircles(die, cv.HOUGH_GRADIENT,
                                  1, 20, param1=50, param2=4, minRadius=3, maxRadius=4)

        for circle in circles:
            print(circle)
            # die = cv.circle(die)

        if len(circles) > 0:
            return len(circles)
        return None

    except Exception as ex:
        return None


def main():
    cam = Camera(CAMERA_LOGITECH)

    while True:

        # read image
        img = cam.read()

        # apply gaussian blur
        img_blurred = cv.GaussianBlur(img, (3, 3), 0)

        # convert to greyscale
        img_grey = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)

        # detect edges
        img_edges = cv.Canny(img_grey, 200, 400)

        # detect contours
        contours, _ = cv.findContours(
            img_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # filter out small contours
        contours = list(filter(lambda c: cv.contourArea(c) > 100, contours))

        for i, contour in enumerate(contours):
            # a, b, c, d, theta = get_contour_bounds(contour)
            # img = cv.rectangle(img, bounds[0], bounds[1], (0, 255, 0))

            # thumbs[i] = img.copy()

            # # crop
            # thumbs[i] = thumbs[i][b[1]:d[1], a[0]:c[0]]

            # # rotate
            # cropped_height, cropped_width = thumbs[i].shape[:2]
            # rot = cv.getRotationMatrix2D(
            #     (cropped_width / 2.0, cropped_height / 2.0), theta, 1)
            # thumbs[i] = np.cross(rot, thumbs[i])

            rect = cv.boundingRect(contour)

            value = get_die_value(img_grey, rect)

            value_str = str(value)
            if value is None:
                value_str = "?"

            img = cv.rectangle(
                img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255))

            img = cv.putText(
                img, value_str, (rect[0] + rect[2] + 5, rect[1] + 5), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            # img = cv.drawMarker(img, a, (255, 0, 0))
            # img = cv.drawMarker(img, b, (255, 255, 0))
            # img = cv.drawMarker(img, c, (0, 255, 255))
            # img = cv.drawMarker(img, d, (255, 0, 255))
            # img = cv.putText(img, "ID: {}, {:.2f} degrees".format(
            #     i, theta), (c[0] + 10, c[1] - 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # overlay remaining contours on original image
        # img = cv.drawContours(img, contours, -1, (255, 0, 0), 3)

        # for k, v in thumbs.items():
        #     cv.imshow("ID: {}".format(k), v)

        cv.imshow('Canny Edge Detection', img_edges)
        cv.imshow('Original with Contours', img)

        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
