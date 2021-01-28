from unittest import TestCase

import cv2

import get_board
import get_slid


class TestGetBoard(TestCase):
    def test_get_board(self):
        for i in range(5, 6):
            path = "/home/joking/PycharmProjects/Chess_Recognition/data/chessboards/" + str(i) + ".jpg"

            print(path)
            cropped = get_board.get_board(path, show=True)
            squares, counter = get_board.get_squares(cropped, show=False)#
            print(len(squares))

    # probleme counter -> winkel stimmt nicht auf dem brett zB 5
    # wenn linien nicht ganz durchgezogen sind, warum sind die linien nicht ganz durchgezogen 8
    # play with cluster distance

    # good 1 4 5 8 12
    # too many 2 3 9 10 11 13
    # too few 6 7  14 16 17 18 19
    # weird board 2 3  99  10 11 13
    # [(242.5, 30.958654), (242.5, 282.63727), (435.54462, 1597.8215), (434.62378, 1808.9536)]

    def test_detect_slid(self):
        for i in range(10, 20):
            print(i)
            path = "/home/joking/PycharmProjects/Chess_Recognition/data/chessboards/" + str(i) + ".jpg"
            squares, board = get_slid.get_board_slid(path)
            temp_path = "/home/joking/PycharmProjects/Chess_Recognition/data/chessboards/tmp/" + str(i) + ".jpg"
            img = cv2.imread(temp_path)
            cv2.imshow("board", board)
            cv2.waitKey(0)