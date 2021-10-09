"""
Debug utils.
"""
import itertools
from copy import copy
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detectboard import get_board

DEBUG = False  # Enable or disable debug images
COUNTER = itertools.count()
DEBUG_SAVE_DIR = "/home/joking/PycharmProjects/Chess_Recognition/debug/"


def reset_counter():
    """ Reset Counter"""
    global COUNTER
    COUNTER = itertools.count()


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
        if DEBUG:
            angles = ['0', '30', '60', '90', '120', '150']
            plot_lines = [0] * len(angles)

            for _, angle in lines:
                i = get_board.det_intervall(angle, 0)
                plot_lines[i] += 1

            # plt.figure(figsize=(12, 7))
            fig, ax = plt.subplots(1, 1)
            # Passing the parameters to the bar function, this is the main function which creates the bar plot
            plt.bar(angles, plot_lines)

            plt.title("Bar plot representing angles of the lines in an image")
            ax.bar(angles, plot_lines, color="blue", zorder=3)
            ax.grid(zorder=0)
            plt.xlabel("Angle in degree")
            plt.ylabel("Number of lines in image")
            # fig.savefig('comparison.png', dpi=200)
            plt.show()

    def points(self, _points, color=rand_color(), size=3):
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
