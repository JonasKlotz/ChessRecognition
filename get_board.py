#!/usr/bin/env python
# coding: utf-8

import math
## IMPORTS
from collections import defaultdict

import cv2  # For Sobel etc
import numpy as np
import scipy.cluster as clstr
import scipy.spatial as spatial
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

import debug
import utility
from calculate_fen.get_board_colors import find_field_colour


def auto_canny(image, sigma=0.33):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def hor_vert_lines(lines):
    """
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    """
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v


def intersections(h, v):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)


def cluster(points, max_dist=60):
    """
    Given a list of points, returns a list of cluster centers.
    """
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), clusters)
    return list(clusters)


def four_point_transform(img, points, square_length=1816):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))


def closest_x_points(points, loc, x):
    """
    Returns the list of x closest points, sorted by distance from loc.
    """

    p = points.copy()
    if loc in p:
        p.remove(loc)
    p.sort(key=lambda x: spatial.distance.euclidean(loc, x))
    return p[:x]


def closest_point(points, loc):
    """
    Returns the closest point, sorted by distance from loc.
    """

    p = points.copy()
    if loc in p: p.remove(loc)
    p.sort(key=lambda x: spatial.distance.euclidean(loc, x))
    return p[0]


def guess_corners(points, img_dim):
    board_corners = []
    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]

    for point in img_corners:
        board_corners.append(closest_point(points, point))
    return board_corners


def find_corners(points, img_dim):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    center_point = closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center_point)
    center_adjacent_point = closest_point(points, center_point)
    points.append(center_point)
    grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))

    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    tolerance = 0.25  # bigger = more tolerance
    for img_corner in img_corners:
        while True:
            cand_board_corner = closest_point(points, img_corner)
            points.remove(cand_board_corner)
            cand_board_corner_adjacent = closest_point(points, cand_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner),
                                                          np.array(cand_board_corner_adjacent))
            if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(cand_board_corner)
                board_corners.append(cand_board_corner)
                break
    return board_corners


def get_points(img, show=False):
    # if show:
    #    fig = plt.figure(figsize=(10,10))
    #    plt.imshow(img)
    #    plt.show()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(np.uint8(gray), 50, 150, apertureSize=3)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 200)
    lines = np.reshape(lines, (-1, 2))

    h, v = hor_vert_lines(lines)
    if len(h) < 9 or len(v) < 9:
        print('too few lines')

    points = intersections(h, v)
    print(img.shape)
    # Cluster intersection points TODO:
    #  maxdist = ca 1/16tel der bild länge hälfte von einem quadrat
    # schwierig weil dann fängt es nicht so gut ab falls das board nicht direkt getroffen wird ...
    # anzahl punkte sicherstellen
    # aus punkten quares generieren
    half_square_len = img.shape[0] / 8  # (vlt bisschen weniger)
    points = cluster(points, max_dist=100)
    if show:
        utility.print_points(points, img)
    return points


def get_board(path, show=False):
    """
    takes picture as inputs
    transforms to only board
    """
    img = cv2.imread(path, 1)
    if show:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.show()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(np.uint8(gray), 50, 150, apertureSize=3)
    # fig = plt.figure(figsize=(10,10))
    # plt.imshow( canny)
    # plt.show()

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 200)
    lines = np.reshape(lines, (-1, 2))

    h, v = hor_vert_lines(lines)
    if len(h) < 9 or len(v) < 9:
        print('too few lines')

    points = intersections(h, v)

    # Cluster intersection points TODO 1 cluster zuviel??
    points = cluster(points)
    if show:
        utility.print_points(points, img)
    # Find corners

    img_dim = np.shape(gray)
    img_dim = (img_dim[1], img_dim[0])
    # todo corners
    board_corners = guess_corners(points, img_dim)
    # board_corners = find_corners(points, img_dim)

    if show:
        utility.print_points(board_corners, img)

    new_img = four_point_transform(img, board_corners)
    if show:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(new_img)
        plt.show()
    return new_img


def get_squares(img, show=False):
    points = get_points(img, show=False)  #
    v_points = sorted(points)  # größer heißt höher im bild, key=lambda x: x[0]
    utility.print_points(v_points, img)

    # betrachte abstand in x koordinate
    standard_dis = abs((v_points[0][0] - v_points[1][0]))  # spatial.distance.euclidean(v_points[0], v_points[1])
    print("dis: ", standard_dis)
    thresh = (standard_dis + 1) * 4

    counter = 0  # soll 9 sein
    p_len = len(points)

    # find transition to next line, (after counter points)
    for i in range(p_len - 1):
        dis = abs((v_points[i][0] - v_points[i + 1][0]))
        counter += 1
        if abs(dis - standard_dis) > thresh:
            break

    # print("counter soll 9 sein: ", counter)

    # split in line array
    lists = utility.chunks(v_points, counter)
    lines = []
    # sort lines after y coordinate
    for line in lists:
        lines.append(sorted(line, key=lambda x: x[1]))

    print()
    squares = []
    for i in range(len(lines) - 1):  # für jede line außer die letzte
        for k in range(len(lines[i]) - 1):  # für jeden punkt außer den letzten
            square_points = [lines[i][k], lines[i][k + 1], lines[i + 1][k + 1], lines[i + 1][k]]
            sq = four_point_transform(img, square_points, square_length=150)  # was ist input für ki 150²
            squares.append(sq)
            if show:
                # print_points(square_points, img)
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(sq)
                plt.show()
    board_img = img

    if len(squares) == 72:  # > 64
        squares = remove_wrong_outline(squares, counter)
        board_img = utility.combine_squares_board_image(squares)

    squares = resize_squares(squares)
    squares = rearrange_squares(squares)

    return squares, board_img


"""
@:returns resized square list
"""


def resize_squares(square_list, d=150):
    resized_list = []
    for square in square_list:
        resized_list.append(cv2.resize(square, (d, d)))
    return resized_list


def rearrange_squares(squares):
    new_squares = [np.zeros(8)] * 64
    for k in range(8):
        for i in range(8):
            old_square = squares[(i * 8) + k]
            new_squares[i + (8 * k)] = old_square
    return new_squares


def remove_wrong_outline(squares, counter):
    """ removes an outline that is wrong. works when eG 72 squares are found and all on extra col/row on the outside

    finds edge rows/cols on the outside
    examines how many "same colored elements" are contained

    """
    end = len(squares)  ## durch 8 teilen??
    print("länge: ", end)
    step = counter - 1  # normalfall 9 punkte - 1 = felder
    zeros = np.zeros(step)
    ones = np.ones(step)

    first_col, first_row, last_col, last_row = [], [], [], []  # color array der entsprechenden zeile/spalte
    f_c_index, f_r_index, l_c_index, l_r_index = [], [], [], []  # zugehörige indizes
    # TODO OPTIMIERUNG

    # finde randspalten
    for i in range(step):
        # erste spalte
        img = squares[(i)]
        first_col.append(find_field_colour(img, False))
        f_c_index.append(i)

        # erste zeile
        img = squares[(i) * step]
        first_row.append(find_field_colour(img, False))
        f_r_index.append(i * step)

        # letzte spalte
        img = squares[(-i - 1)]
        last_col.append(find_field_colour(img, False))
        l_c_index.append((-i - 1))

        # letzte zeile
        img = squares[(i + 1) * step - 1]
        last_row.append(find_field_colour(img, False))
        l_r_index.append((i + 1) * step - 1)

        # thresh ab wann cut??

    # print("1.Spalte", first_col)
    # print("1.Reihe", first_row)
    # print("9.Spalte", last_col)
    # print("9. Reihe", last_row)

    test_squares = squares.copy()

    first_col_max = max(np.sum(first_col == zeros), np.sum(first_col == ones))
    first_row_max = max(np.sum(first_row == zeros), np.sum(first_row == ones))
    last_col_max = max(np.sum(last_col == zeros), np.sum(last_col == ones))
    last_row_max = max(np.sum(last_row == zeros), np.sum(last_row == ones))

    maxima = [first_col_max, first_row_max, last_col_max, last_row_max]
    print(maxima)

    # first_col contains most same elements
    if len(squares) > 64:
        if 0 == np.argmax(maxima):
            f_c_index.sort(reverse=True)
            for index in f_c_index:
                del squares[index]

        # first_crow contains most same elements
        if 1 == np.argmax(maxima):
            f_r_index.sort(reverse=True)
            for index in f_r_index:
                del squares[index]

        # last_col contains most same elements
        if 2 == np.argmax(maxima):
            l_c_index.sort()  # as this one only contains negatves real sort necessary
            for index in l_c_index:
                del squares[index]

        # last row contains most same elements
        if 3 == np.argmax(maxima):
            l_r_index.sort(reverse=True)
            for index in l_r_index:
                del squares[index]

    # 0-step = 1-> erste spalte
    # 0-step = i*step -> erste zeile

    # step = (i+1)*step-1 -> letzte zeile
    # end-step - end -> letzte spalte

    # wenn alle gleiche farbe haben remove oder vlt wenn > 5 gleich da in einer r/c
    print("The new number of squares is now: ", len(squares))
    return squares


def combine_points2(all_points, img):
    """ combines the given point lists
    only accepts points that are in at least two of the point arrays
    returns clustered and reduced version with more probable points
    """

    debug.DebugImage(img) \
        .points(all_points, color=(0, 255, 0)) \
        .save("get_points_unclustered_points", prefix=True)

    alfa = math.sqrt(cv2.contourArea(np.array(all_points)) / 81) * 3  # how do i optimally choose alfa ???
    X = DBSCAN(eps=alfa).fit(all_points)

    # splits all points in different lists regarding labels form dbscan
    # 0 index = -1 1st index = 1
    labeled_points = [all_points[X.labels_ == l] for l in np.unique(X.labels_)]

    # -1 labels
    deb_img = debug.DebugImage(img)
    for i in range(len(np.unique(X.labels_))):
        deb_img = deb_img.points(labeled_points[i], color=debug.rand_color())
    deb_img.save("get_points_clustered_points", prefix=True)


def combine_points(points_list):
    """ combines the given point lists
    only accepts points that are in at least two of the point arrays
    :param points_list list of np point arrays
    returns clustered and reduced version with more probable points
    """
    _ANALYST_RADIUS = 80
    final_points = []
    # points_list = points.tolist()
    # 4 mal points
    for i, points in enumerate(points_list):
        # für jeden point dadrin
        for point in points:
            found = False
            # für alle anderen point arrays
            for k, set_to_check in enumerate(points_list):
                # wenn das selbe set
                if k == i or len(set_to_check) <= 1: continue  # problem nur der punkt selber drinne
                neighbour = closest_point(set_to_check, point)
                distance = spatial.distance.euclidean(neighbour, point)
                # print(distance)
                if distance <= _ANALYST_RADIUS:
                    final_points.append(neighbour)
                    set_to_check.remove(neighbour)
                    found = True
            if found:
                final_points.append(point)
                # points.remove(point)
    # check if close point in at least one other array
    final_points = cluster(final_points)
    return final_points
