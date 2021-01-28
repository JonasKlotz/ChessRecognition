from collections import defaultdict

import cv2
import numpy as np
import scipy.cluster as clstr
import scipy.spatial as spatial
from scipy.spatial import ConvexHull

import debug


def simplify_image(img, limit, grid, iters):
    """Simplify image using CLAHE algorithm (adaptive histogram
    equalization)."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for _ in range(iters):
        img = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid).apply(img)
    if limit != 0:
        kernel = np.ones((10, 10), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
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
    return edged


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


def det_intervall(angle):
    """determines the intervall where the input angle is
    """
    n = angle * (180 / np.pi)
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
    """
    # 0/180&90, 30&120, 60&150
    line_array = [[] for _ in range(6)]

    for distance, angle in lines:
        angle_index = det_intervall(angle)  # find intervall of the angle
        line_array[angle_index].append((distance, angle))  # append line to line array
    # print(len(line_array[angle_index]))

    count_list = [0] * 3
    for i in range(3):
        count_list[i] = len(line_array[i]) + len(line_array[i + 3])

    line_max = np.argmax(count_list)
    vertical, horizontal = line_array[line_max], line_array[line_max + 3]
    return horizontal, vertical


def intersections(h, v, dims):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    height, width = dims
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
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

    p = points.copy()
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
        return triangle area with det
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

    # as long as there are point in the hull
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
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))


# centroid = np.mean(points, axis=0)
# print_points([centroid], img)


# cluster points in loop
# save in array


def combine_points(points_list):
    """ combines the given point lists
    only accepts points thst are in at least two of the point arrays
    :param points_list list of np point arrays
    returns clustered and reduced version with more problable points
    """
    _ANALYST_RADIUS = 80
    final_points = []

    # 4 mal points
    for i, points in enumerate(points_list):
        # für jeden point dadrin
        for point in points:
            found = False
            # für alle anderen point arrays
            for k, set_to_check in enumerate(points_list):
                # wenn das selbe set
                if k == i or len(set_to_check) < 1: continue
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


def get_points(img=None, img_path=None):
    if not img and not img_path:
        RuntimeError("No Image or Path provided to get points")
    if not img:
        img = cv2.imread(img_path, 1)
    # rows, columns, and channels (if the image is color):
    height, width, d = img.shape
    # print("Breite ", width, " Höhe ", height)
    clahe_settings = [[3, (2, 6), 5],  # @1
                      [3, (6, 2), 5],  # @2
                      [5, (3, 3), 5],  # @3
                      [0, (0, 0), 0]]

    first_iteration = False
    all_points = []

    for key, arr in enumerate(clahe_settings):
        tmp = simplify_image(img, limit=arr[0], grid=arr[1], iters=arr[2])
        tmp = canny_edge(tmp)

        # get hough lines in right shape
        lines = get_hough_lines(tmp)
        # print_hough_lines(lines, img.copy())

        # fing maximum in angle peaks choose orthogonal lines
        h, v = find_peak_angles(lines)
        # print_hough_lines(h+v, img.copy())
        # find intersections
        points = intersections(h, v, [height, width])
        points = cluster(points)
        # print_points(points, img)
        # all_points.append(points)
        debug.DebugImage(img) \
            .hough_lines(h + v, color=(0, 0, 255)) \
            .points(points, color=(0, 255, 0)) \
            .save("get_points debug")

        if not first_iteration:
            all_points = points
            first_iteration = True
        else:

            # if points.s == None: continue
            all_points = np.concatenate((all_points, points), axis=0)

    # cluster
    # points = combine_points(all_points)
    points = cluster(all_points)
    debug.DebugImage(img) \
        .points(points, color=(0, 255, 0)) \
        .save("get_points_debug_final")
    # corners = get_corners(points)
    return points


if __name__ == '__main__':
    points = get_points(img_path="data/chessboards/1.jpg")
