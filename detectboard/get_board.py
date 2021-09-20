import logging
from collections import Counter
from collections import defaultdict
from copy import copy

import cv2
import numpy as np
import scipy.cluster as clstr
import scipy.spatial as spatial
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

import debug


def simplify_image(img, limit, grid, iters):
    """
    Simplify image using CLAHE algorithm (adaptive histogram
    equalization)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for _ in range(iters):
        img = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid).apply(img)
    if limit != 0:
        kernel = np.ones((10, 10), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    debug.DebugImage(img) \
        .save("Simplified_image")
    return img


def canny_edge(img, sigma=0.33):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (7, 7), 2)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)

    # return the edged image

    # Otsu's thresholding
    _, otsu = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # debug.DebugImage(otsu).save("canny edge processed")

    debug.DebugImage(otsu) \
        .save("edge_map")

    return otsu


def get_hough_lines(img):
    """ gets ho
    #(image, rho, theta, threshold, lines, srn, stn, min_theta, max_theta)
    rho : The resolution of the parameter r in pixels. We use 1 pixel.
    theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    threshold: The minimum number of intersections to "*detect*" a line
    """
    lines = cv2.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)
    lines = np.reshape(lines, (-1, 2))
    return lines


def find_line_clusters():
    pass


def det_intervall(angle, centroid):
    """
    determines the interval where the input angle is
    """
    n = (angle * (180 / np.pi) + centroid) % 180
    # print("n = ", n)

    if n <= 15 or n > 165:
        return 0
    if 15 < n <= 45:
        return 1
    if 45 < n <= 75:
        return 2
    if 75 < n <= 105:
        return 3
    if 105 < n <= 135:
        return 4
    if 135 < n <= 165:
        return 5


def find_peak_angles(lines):
    """ finds two peak angles by finding biggest sum of orthogonal lines
    @:return: 2 lists containing the horizontal and vertical lines
    """
    # 0/180&90, 30&120, 60&150
    # how many intervals?
    line_array = [[] for _ in range(6)]
    angle_array = []
    angle_array = np.zeros(len(lines))
    # create array containing all angles in grad
    for i in range(len(lines)):
        distance, angle = lines[i]
        angle_array[i] = angle * (180 / np.pi)

    # cluster angles with db scan
    clustering = DBSCAN(eps=5, min_samples=8).fit(angle_array.reshape(-1, 1))
    labels = clustering.labels_

    # find 2 biggest clusters
    mc = Counter(labels).most_common(2)
    index_cluster_1 = mc[0][0]
    # index_cluster_2, = mc[1][0]
    # print(index_cluster_1, index_cluster_2)

    # biggest cluster centroid
    points_of_cluster_0 = angle_array[labels == index_cluster_1]
    centroid_of_cluster_0 = np.mean(points_of_cluster_0, axis=0)
    # print("centroid 0 ", centroid_of_cluster_0)

    # points_of_cluster_1 = angle_array[labels == index_cluster_2]
    # centroid_of_cluster_1 = np.mean(points_of_cluster_1, axis=0)
    # print("centroid 01 ", centroid_of_cluster_1)

    for distance, angle in lines:
        angle_index = det_intervall(angle, centroid_of_cluster_0)  # find interval of the angle
        line_array[angle_index].append((distance, angle))  # append line to line array

    # print(len(line_array[angle_index]))
    equi_angle = int(len(line_array) / 2)  # eg index 0 and 0 + equi_angle are orthogonal to each other \
    # -> see function above
    # print("equi_angle = ", equi_angle)

    # count list contains number of lines in the two orthogonal angles
    count_list = [0] * equi_angle
    for i in range(equi_angle):
        count_list[i] = len(line_array[i]) + len(line_array[i + equi_angle])

    # index of the angle interval with the most lines
    line_max = np.argmax(count_list)

    # contains the lines
    vertical, horizontal = line_array[line_max], line_array[line_max + equi_angle]
    return horizontal, vertical


def intersections(h, v, dims):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    height, width = dims
    points = []
    for rho1, theta1 in h:
        for rho2, theta2 in v:
            A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            point = np.linalg.solve(A, b)
            # koords (breite, höhe)
            if 0 <= point[0] < width and 0 <= point[1] <= height:
                points.append(point)

    return np.array(points)


def cluster(points, max_dist=80):
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


def closest_point(points, loc):
    """
    Returns the closest point, sorted by distance from loc.
    """
    p = copy(points)
    if loc in p: p.remove(loc)
    p.sort(key=lambda x: spatial.distance.euclidean(loc, x))
    return p[0]


def quad_area(a, b, c, d):
    """
    input 4 points
    triangulates 4 points into 4 triangles
    calculates area of them /2
    returns quad area
    """

    def _triangle_area(a, b, c):
        """
        input 3 points
        return triangle area via det
        """
        ax, ay = a
        bx, by = b
        cx, cy = c
        return abs(0.5 * (((bx - ax) * (cy - ay)) - ((cx - ax) * (by - ay))))

    ABD = _triangle_area(a, b, d)
    BCD = _triangle_area(b, c, d)
    ABC = _triangle_area(a, b, c)
    ACD = _triangle_area(a, c, d)
    return (ABD + BCD + ABC + ACD) / 2


def get_corners(points):
    """
    greedy corner search that maximizes quad area in the given point array.

    :param points: intersection points.
    :return: The four outer points of the detected chessboard.
    """

    tmp = np.array(points)

    hull = ConvexHull(tmp).vertices  # index of points
    hull = tmp[hull]

    # Rearrange Hull
    tmp = []
    for i in range(len(hull)):
        tmp.append((hull[i][0], hull[i][1]))
    hull = tmp

    # pop 4 random points
    corners = hull[:4]
    del hull[:4]

    # calculate quad area for them
    a, b, c, d = corners
    max_area = quad_area(a, b, c, d)

    # as long as there are points in the hull
    while hull:
        # get random
        new_node = hull.pop()
        tmp_max_area = -1
        tmp_max_list = []
        # test if new point creates bigger area than previous four
        for p in corners:
            # calculate area of new quad
            tmp_list = corners[:]  # fastest way to copy
            tmp_list.remove(p)
            tmp_list.append(new_node)

            a, b, c, d = tmp_list  #
            area = quad_area(a, b, c, d)

            # compare areas save maximum for this node of
            if area > tmp_max_area:
                tmp_max_list = tmp_list
                tmp_max_area = area

        # only remove point a that maximizes new area
        if tmp_max_area > max_area:
            corners = tmp_max_list
            max_area = tmp_max_area

    return corners


def four_point_transform(img, points, square_length=1816):
    """
    transforms image warps the perspective so that the 4 points are now the full img
    :param img:
    :param points:    points
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left

    :param square_length:  resulting size of cropped board
    :return:
    """
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))


def get_points(img=None, img_path=None):
    """
    Main Function to find points
    :param img:
    :param img_path:
    :return:
    """
    if (img is None) and not img_path:
        RuntimeError("No Image or Path provided to get points")

    if (img is None):
        img = cv2.imread(img_path, 1)

    # rows, columns, and channels (if the image is color):
    height, width, d = img.shape
    clahe_settings = [[3, (2, 6), 5],  # @1
                      [3, (6, 2), 5],  # @2
                      [5, (3, 3), 5],  # @3
                      [0, (0, 0), 0]]

    first_iteration = False
    all_points = []

    for key, arr in enumerate(clahe_settings):
        try:
            tmp = simplify_image(img, limit=arr[0], grid=arr[1], iters=arr[2])
            tmp = canny_edge(tmp)

            # get hough lines in right shape
            lines = get_hough_lines(tmp)
            debug.DebugImage(img) \
                .hough_lines(lines, (0, 0, 255)) \
                .save("all lines")

            debug.DebugImage(img).plot_lines_peaks(lines)
            # find maximum in angle peaks choose orthogonal lines
            h, v = find_peak_angles(lines)
            debug.DebugImage(img) \
                .hough_lines(v, (0, 0, 255)) \
                .hough_lines(h, (0, 255, 0)) \
                .save("horizontal and vertical lines")

            # find intersections
            points = intersections(h, v, [height, width])

            # Todo: Was wenn ich nicht davor clustere, wie wirkt sich das auf mein Endergebnis aus?
            points = cluster(points)
            # print_points(points, img)
            # all_points.append(points)

            # Problem Outlier points
            if not first_iteration:
                all_points = points
                first_iteration = True
            else:

                # if points.s == None: continue
                all_points = np.concatenate((all_points, points), axis=0)
        except:
            logging.error("Iteration Failed")
            continue
    # combine cluster
    # test = combine_points2(all_points, img)

    points = cluster(all_points)

    debug.DebugImage(img) \
        .points(points, color=(0, 0, 255), size=10) \
        .save("get_points_final_points")

    corners = get_corners(points)
    debug.DebugImage(img) \
        .points(corners, color=(0, 0, 255), size=10) \
        .save("get_points_final_corners")

    return corners


if __name__ == '__main__':
    """# Create the parser
    my_parser = argparse.ArgumentParser(prog='chess_recognition',
                                        description='Chessrecognition programm, evaluates a picture of a chess programm and ...')

    # Add the arguments
    my_parser.add_argument('Path',
                           metavar='image_path',
                           type=str,
                           help='the path to the image')

    # Execute parse_args()
    args = my_parser.parse_args()

    input_path = args.Path

    # print(vars(args))
    """
    for i in range(22, 23):
        input_path = "/home/joking/Projects/Chessrecognition/Data/chessboards/board_recog_2/{}.jpg".format(i)
        print("Loading board from ", input_path)

        img = cv2.imread(input_path, 1)

        corners = get_points(img=img)

        debug.DebugImage(img) \
            .points(corners, color=(0, 0, 255)) \
            .save("get_points_main_corners")

    # squares, board_img, corners = get_slid.get_board_slid(input_path)
