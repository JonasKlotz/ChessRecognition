import copy
import glob
import time

import cv2
import natsort
import numpy as np
import pandas as pd

import get_points
from get_slid import get_board_slid

res_dir = "/home/joking/Projects/Chessrecognition/Data/Results/Board Recognition/trash/"


def draw_and_save(index, img, points, path, slid=True):
    img2 = copy.copy(img)
    for point in points:
        cv2.circle(img2, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)

    if slid:
        save_path = path + str(index + 1) + "_slid.jpg"
    else:
        save_path = path + str(index + 1) + "_points.jpg"
    cv2.imwrite(save_path, img2)
    print("printed to ", save_path)


def evaluate_board(index, path):
    img = cv2.imread(path, 1)
    elapsed_time_slid, elapsed_time_points = -1, -1
    try:
        # Evaluate SLID
        start_time = time.time()
        squares, board_img, corners = get_board_slid(path)
        elapsed_time_slid = time.time() - start_time
        print("Processing SLID took ", elapsed_time_slid, "seconds...")
        draw_and_save(index, img, corners, res_dir, slid=True)

    except:
        print("SLID failed")

    try:
        # Evaluate GETPOINTS
        start_time = time.time()
        corners2 = get_points.get_points(img=img)
        elapsed_time_points = time.time() - start_time
        print("Processing GETPOINTS took ", elapsed_time_points, "seconds...")
        draw_and_save(index, img, corners2, res_dir, slid=False)
    except:
        print("points failed")

    return elapsed_time_slid, elapsed_time_points


if __name__ == '__main__':

    data_dir = "/home/joking/Projects/Chessrecognition/Data/chessboards/board_recog_2/"
    res_slid_list, res_points_list = [], []

    filenames = natsort.natsorted(glob.glob(data_dir + '*.jpg'))

    for i, filename in enumerate(filenames):
        print(filename)
        res_slid, res_points = evaluate_board(i, filename)
        res_points_list.append(res_points)
        res_slid_list.append(res_slid)

    data = np.column_stack((res_points_list, res_slid_list))
    print(data)
    pd.DataFrame(data, columns=["Ours", "CPS"]).to_csv(res_dir + '/results.csv', index=False)
