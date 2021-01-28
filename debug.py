"""
Debug utils.
"""
import itertools
from copy import copy
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np

import get_points

DEBUG = True  # Enable or disable debug images
COUNTER = itertools.count()
DEBUG_SAVE_DIR = "debug/"


def rand_color():
    """Returns a random rgb color."""
    return randint(0, 255), randint(0, 255), randint(0, 255)


class DebugImage:
    """
    Represents a debug image. Can draw points and lines and save the
    resulting image.
    """

    def __init__(self, img):
        if DEBUG:
            if isinstance(img, tuple):
                img = np.zeros((img[0], img[1], 3), np.uint8)
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self.img = copy(img)

    def lines(self, _lines, color=(0, 0, 255), size=2):
        """Draw lines in the image."""
        if DEBUG:
            for li1, li2 in _lines:
                cv2.line(self.img, tuple(li1), tuple(li2), color, size)
        return self

    def hough_lines(self, _lines, color=(0, 0, 255), size=2):
        if DEBUG:
            for distance, angle in _lines:
                a = np.cos(angle)
                b = np.sin(angle)
                x0 = a * distance
                y0 = b * distance
                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * a)
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * a)
                cv2.line(self.img, (x1, y1), (x2, y2), color, size)
        return self

    def plot_lines_peaks(self, lines):

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        angles = ['0', '30', '60', '90', '120', '150']
        plot_lines = [0] * len(angles)

        for _, angle in lines:
            i = get_points.det_intervall(angle)
            plot_lines[i] += 1
        print(plot_lines)
        ax.bar(angles, plot_lines)
        plt.show()

    def points(self, _points, color=(0, 0, 255), size=10):
        """Draw points in the image."""
        if DEBUG:
            for point in _points:
                cv2.circle(self.img, (int(point[0]), int(point[1])), size,
                           color, -1)
        return self

    def save(self, filename, prefix=True):
        """Save the image."""
        global COUNTER
        if DEBUG:
            if prefix:
                __prefix = "__debug_" + "%04d" % int(next(COUNTER)) + "_"
            else:
                __prefix = ""

            cv2.imwrite(DEBUG_SAVE_DIR + __prefix + filename + ".jpg",
                        self.img)
        print(DEBUG_SAVE_DIR + __prefix + filename + ".jpg")
