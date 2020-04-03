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


def crop(img, x, y, w, h):
    return img[y:y+h, x:x+w]


def main():
    cam = Camera(CAMERA_SURFACE_FRONT)

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

        img_annotated = img.copy()

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

            (_, _, angle) = rot_rect = cv.minAreaRect(contour)
            rot_rect = cv.boxPoints(rot_rect)
            rot_rect = np.int0(rot_rect)

            rect = cv.boundingRect(contour)

            # value = get_die_value(img_grey, rect)

            # value_str = str(value)
            # if value is None:
            #     value_str = "?"

            img_annotated = cv.rectangle(
                img_annotated, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255))

            img_annotated = cv.drawContours(
                img_annotated, [rot_rect], 0, (255, 0, 0), 2)

            # (x1, y1), _, (x2, y2), _ = rot_rect
            # w = x2 - x1
            # h = y2 - y1
            # size = math.floor(math.sqrt(w ** 2 + h ** 2))
            # tmp = np.zeros((size, size, 3), np.uint8)

            # mask = tmp.copy()
            # mask = cv.fillPoly(mask, [rot_rect], (255, 255, 255))
            # mask = crop(mask, rect[0], rect[1],
            #             rect[2], rect[3])

            # tmp = cv.copyTo(img, mask, tmp)

            xs = [x for [x, _] in rot_rect]
            ys = [y for [_, y] in rot_rect]
            p1 = (min(xs), min(ys))
            p2 = (max(xs), max(ys))
            w = p2[0] - p1[0]
            h = p2[1] - p1[1]
            print(p1, p2)
            tmp = crop(img, p1[0], p1[1], w, h)

            rot_mat = cv.getRotationMatrix2D(
                (int((p1[0] + w)/2), int((p1[1] + h)/2)), angle, 1.0)
            tmp = cv.warpAffine(img, rot_mat, (w, h))

            cv.imshow("tmp", tmp)
            break

            # img = cv.putText(
            #     img, value_str, (rect[0] + rect[2] + 5, rect[1] + 5), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

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
