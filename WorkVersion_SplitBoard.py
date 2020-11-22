

import numpy as np

from IPython.display import Image, display
import PIL.Image
from matplotlib import pyplot as plt
# import Pymatplotlib.image as mpimg
import scipy.ndimage
import cv2  # For Sobel etc

np.set_printoptions(suppress=True)  # Better printing of arrays
plt.rcParams['image.cmap'] = 'jet'  # Default colormap is jet



import scipy.spatial as spatial
import scipy.cluster as clstr
from collections import defaultdict
import os




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



def print_points(plist, img):
    """
    Plots list of points as red circles on given image
    """
    circled = img.copy()
    for point in plist:
        cx, cy = point
        cx = int(cx)
        cy = int(cy)
        cv2.circle(circled, (cx, cy), 20, (255, 0, 0), -1)  # red (255,0,0), black 1
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(circled)



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
        print_points(points, img)
    return points



def get_board(path, show=False):
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
        print_points(points, img)
    # Find corners

    img_dim = np.shape(gray)
    img_dim = (img_dim[1], img_dim[0])
    board_corners = guess_corners(points, img_dim)
    # board_corners = find_corners(points, img_dim)

    if show:
        print_points(board_corners, img)

    new_img = four_point_transform(img, board_corners)
    if show:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(new_img)
        plt.show()
    return new_img



def split_board(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sq_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            arr.append(img[i * sq_len: (i + 1) * sq_len, j * sq_len: (j + 1) * sq_len])
    return arr




# get squares
# *    finde den übergang indem ich durch die liste durchgehe, immer einen weiter nach unten
# *    merke den normalen abstand, wenn der nächste punkt zu weit entfernt ist merke dir den übergang
# *    damit kästchen einfacher berechnet werden können
# *    kästchen ausschneiden
#     
# find square color
# *    binarize image of square
# *    look if bigger part is white or black
# 
# * at least 33 out of 64 squares are classified correctly -> then works. 
# * 
# To combat this, two chessboard data structures are created. One starts with
# a white square, the other with black. Then the classified colors of the real
# chessboard are compared with these two ideal chessboards. Whichever of the
# ideal chessboard more closely resembles the classified colors is considered the
# configuration of the chessboard in the picture. So now it is known that when a
# square was classified as black, even though it is supposed to be white, it is
# probably occupied by a black piece. This kind of approach aids determining the
# occupation and the color of the pieces.
# 
# * wenn zuviele squares und viele gleiche squares(am rand) gehören die warscheinlich nicht aufs brett
# 
# 



def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))



def get_squares(img, show=False):
    points = get_points(cropped, show=False)  #
    v_points = sorted(points)  # größer heißt höher , key=lambda x: x[0]
    print_points(v_points, img)
    # print_points(v_points[8:12], img)

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

    print(counter)
    # split in line array
    lists = chunks(v_points, counter)
    lines = []
    # sort lines after y coordinate
    for line in lists:
        lines.append(sorted(line, key=lambda x: x[1]))

    print()
    squares = []
    for i in range(len(lines) - 1):  # für jede line außer die letzte
        for k in range(len(lines[i]) - 1):  # für jedne punkt außer den letzten
            square_points = [lines[i][k], lines[i][k + 1], lines[i + 1][k + 1], lines[i + 1][k]]
            sq = four_point_transform(cropped, square_points, square_length=200)  # was ist input für ki 150²
            squares.append(sq)
            if show:
                # print_points(square_points, img)
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(sq)
                plt.show()
    return squares, counter


def test():
    for i in range(2, 3):
        path = "../Data/chessboards/" + str(i) + ".jpg"
        print(path)
        cropped = get_board(path, show=False)
        squares, counter = get_squares(cropped, show=False)  #
        print(len(squares))

# probleme counter -> winkel stimmt nicht auf dem brett zB 5
# wenn linien nicht ganz durchgezogen sind, warum sind die linien nicht ganz durchgezogen 8
# play with cluster distance


# good 1 4 5 8 12
# too many 2 3 9 10 11 13
# too few 6 7  14 16 17 18 19
# weird board 2 3  99  10 11 13
# [(242.5, 30.958654), (242.5, 282.63727), (435.54462, 1597.8215), (434.62378, 1808.9536)]



    gray_square = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_square, (5, 5), 0)
    _, img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove noise
    morph_kernel = np.ones((15, 15), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, morph_kernel)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_binary)
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cropped)
    plt.show()

# ## Problem ich will die randreihen finden
# 
# * wie mache ich das?
# * probleme es sind mehr als 8 
# * die extra reihe kann auf jeder seite sein
# * entweder mein step geht 8 runter oder 9
# * 8 nach rechts oder 9
# * wie stelle ich das fest welches
# * 
# * im worst case kann ich einfach immer nur 8 gehen das könnte zB in startstellung schwierig werden da erkennung mau sein kann
# * **aber wenn 8 mal hintereinander das selbe erkannt wird ist es schon unwarscheinlich**




# TODO OPTIMIERUNG

def find_field_colour(img, show=True):
    gray_square = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_square, (5, 5), 0)
    _, img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove noise
    morph_kernel = np.ones((15, 15), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, morph_kernel)

    rows, cols = img_binary.shape

    if show == True:
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.show()

    n_white_pix = cv2.countNonZero(img_binary)
    n_black_pix = rows * cols - n_white_pix

    if n_white_pix > n_black_pix:
        return 1
    return 0

for i in range(step):
    # erste spalte
    img = squares[(i)]
    first_col.append(find_field_colour(img, False))
   # f_c_index
    # erste zeile
    img = squares[(i) * step]
    first_row.append(find_field_colour(img, False))

    # letzte zeile
    img = squares[(i + 1) * step - 1]
    last_row.append(find_field_colour(img, False))

    # letzte spalte
    img = squares[(-i + 1)]
    last_col.append(find_field_colour(img, False))

    # thresh ab wann cut??

print("1.Spalte", first_col)
print("1.Reihe", first_row)
print("9.Spalte", last_col)
print("9. Reihe", last_row)

first_col_max = max(np.sum(first_col == zeros), np.sum(first_col == ones))
first_row_max = max(np.sum(first_row == zeros), np.sum(first_row == ones))
last_col_max = max(np.sum(last_col == zeros), np.sum(last_col == ones))
last_row_max = max(np.sum(last_row == zeros), np.sum(last_row == ones))

print(first_col_max, first_row_max, last_col_max, last_row_max)


# 0-step = 1-> erste spalte
# 0-step = i*step -> erste zeile

# step = (i+1)*step-1 -> letzte zeile
# end-step - end -> letzte spalte

# wenn alle gleiche farbe haben remove oder vlt wenn > 5 gleich da in einer r/c


def compare_board(board_to_compare):
    """
    #returns the board which is closer to given board
    """
    n_squares = len(board_to_compare)
    board_1 = np.zeros(n_squares, dtype=int)
    board_1[::2] = 1

    board_2 = np.zeros(n_squares, dtype=int)
    board_2[1::2] = 1

    board = 2
    if np.sum(board_to_compare == board_1) > np.sum(board_to_compare == board_2):
        return board_1
    return board_2




# path = "../Data/test.png"
# img = cv2.imread(path, cv2.IMREAD_COLOR)

def get_board_colors(squares):
    """
    für jedes feld einen listeneintrag
    das feld wird binarized 
    falls mehr weiße pixel im bild -> listeneintrag auf 1 gesetzt
    dann verglichen mit 2 brettern die die beiden möglichen startkonfigurationen haben
    anfangsfeld oben links schwarz, oder weiß dann abwechselnd

    """
    n_squares = len(squares)
    square_list = np.zeros(n_squares, dtype=int)  # für jedes square eine liste

    for i in range(n_squares):
        img = squares[i]
        square_list[i] = find_field_colour(img)

    return square_list


color = get_board_colors(squares)
# color = compare_board(color)
print(color)

# In[94]:


# ## Piece Color feststellen
# 
# * noise entfernen (blurring binary invertion, closing of picture(remove holes))
# * feldfarbe ist bekannt 
# * ob das feld leer ist ist bekannt



# ## Fill directories with squares
# 


def fill_dirs():
    for i in range(1, 20):
        path = "../Data/chessboards/" + str(i) + ".jpg"
        print(path)
        cropped = get_board(path, show=False)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(cropped)
        plt.show()
        squares = split_board(cropped)
        k = 0
        try:
            parent_dir = '../Data/chessboards/squares'
            directory = str(i)
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
            print("Directory '%s' created" % directory)
        except:
            print("Directory '%s' already exists" % directory)
        for square in squares:
            cv2.imwrite(path + "/" + str(k) + '.jpg', square)  # '../Data/chessboards/squares/' + str(i)
            k += 1

