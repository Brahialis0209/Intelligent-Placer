from math import cos, sin, pi
import cv2
from random import randint

from src.alg.structures import Contour, Point, Segment, distance


# converting grades to radians
def grad_ro_rad(grad: int):
    return grad * pi / 180


# transfer contour cnt to x_diff and y_diff
def transfer_contour(cnt: Contour, x_diff: int, y_diff: int):
    for s in cnt.contour:
        p1 = s.p1
        x = p1.x
        y = p1.y
        new_x = x + x_diff
        new_y = y + y_diff
        s.p1.x = new_x
        s.p1.y = new_y

        p2 = s.p2
        x = p2.x
        y = p2.y
        new_x = x + x_diff
        new_y = y + y_diff
        s.p2.x = new_x
        s.p2.y = new_y


# rotate contour cnt to angle alpha
def rotate_contour(cnt: Contour, alpha: int):
    cX, cY = cnt.get_moments()
    for s in cnt.contour:
        p1 = s.p1
        x = p1.x - cX
        y = p1.y - cY
        new_x = x * cos(alpha) - y * sin(alpha)
        new_y = x * sin(alpha) + y * cos(alpha)
        s.p1.x = int(new_x + cX)
        s.p1.y = int(new_y + cY)

        p2 = s.p2
        x = p2.x - cX
        y = p2.y - cY
        new_x = x * cos(alpha) - y * sin(alpha)
        new_y = x * sin(alpha) + y * cos(alpha)
        s.p2.x = int(new_x + cX)
        s.p2.y = int(new_y + cY)


# calculating orientation area
def area(p1: Point, p2: Point, p3: Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)


# if segment s1 intersect segment s2 (projections)
def intersect_axis(a, b, c, d):
    if a > b:
        k = a
        a = b
        b = k
    if c > d:
        k = c
        c = d
        d = k
    return max(a, c) <= min(b, d)


# if segment s1 intersect segment s2
def intersect_segments(s1: Segment, s2: Segment):
    a = s1.p1
    b = s1.p2
    c = s2.p1
    d = s2.p2
    return intersect_axis(a.x, b.x, c.x, d.x) and \
           intersect_axis(a.y, b.y, c.y, d.y) and \
           area(a, c, d) * area(b, c, d) <= 0 and \
           area(c, a, b) * area(d, a, b) <= 0


# if contour cnt1 intersect contour cnt2
def intersect_contours(cnt1: Contour, cnt2: Contour):
    for s1 in cnt1.contour:
        for s2 in cnt2.contour:
            if intersect_segments(s1, s2):
                return True
    return False


# if s2.node lay on s1
def intersect_node(s1: Segment, s2: Segment):
    if distance(s1.p1, s2.p1) + distance(s1.p2, s2.p1) == distance(s1.p1, s1.p2):
        return True, s2.p1.y < s2.p2.y
    elif distance(s1.p1, s2.p2) + distance(s1.p2, s2.p2) == distance(s1.p1, s1.p2):
        return True, s2.p2.y < s2.p1.y
    return False, False


#  Nested point p into contour cnt?
def point_in_polygon(p: Point, cnt: Contour):
    max_x = cnt.recalc_maxX()
    s = Segment(p, Point(max_x + 1, p.y))
    inter_count = 0
    for s_i in cnt.contour:
        node_flag, true_node_inter = intersect_node(s, s_i)
        if node_flag:
            if true_node_inter:
                inter_count += 1
            continue
        elif intersect_segments(s, s_i):
            inter_count += 1
    if inter_count % 2 == 0:
        return False
    else:
        return True


#  Nested contour cnt1 into contour cnt2?
def polygon_in_polygon(cnt1: Contour, cnts: list, cnt_num: int):
    for i, cnt in enumerate(cnts):
        if i == cnt_num:
            continue
        for s in cnt1.contour:
            p1 = s.p1
            p2 = s.p2
            if not point_in_polygon(p1, cnt) or not point_in_polygon(p2, cnt):
                return False
    return True


#  Nested rectangle cnt1 into rectangle сnt2?
def nested_poly(cnt1, cnt2):
    x1, y1, w1, h1 = cv2.boundingRect(cnt1)
    x2, y2, w2, h2 = cv2.boundingRect(cnt2)
    if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
        return True
    return False


# if one contour intersects other contours
def intersect_other_contours(cnt: Contour, cnts: list, contour_num: int):
    for i, other_cnt in enumerate(cnts):
        if i == contour_num:
            continue
        if intersect_contours(cnt, other_cnt):
            return True
    return False


#  generate new point (new center for contour), all nodes will be in polygon
def generate_point_in_poly(cnt: Contour, poly: Contour):
    x, y, w, h = cv2.boundingRect(poly.get_cv_contour())
    new_x = randint(x, x + w)
    new_y = randint(y, y + h)
    p = Point(new_x, new_y)
    xM, yM = cnt.get_moments()
    x_diff = p.x - xM
    y_diff = p.y - yM
    transfer_contour(cnt, x_diff, y_diff)
    while not point_in_polygon(p, poly) or intersect_contours(cnt, poly):
        new_x = randint(x, x + w)
        new_y = randint(y, y + h)
        p = Point(new_x, new_y)
        transfer_contour(cnt, -x_diff, -y_diff)
        x_diff = p.x - xM
        y_diff = p.y - yM
        transfer_contour(cnt, x_diff, y_diff)
    transfer_contour(cnt, -x_diff, -y_diff)
    return p


#  generate step (x and y range ) for contour (all nodes will be in polygon)
def generate_step_to_transfer(step_x, step_y, cnt: Contour, poly: Contour, direct: int):
    if direct == 0:
        transfer_contour(cnt, step_x, 0)
        if intersect_contours(cnt, poly):
            transfer_contour(cnt, -step_x, 0)
            direct = 1
        else:
            transfer_contour(cnt, -step_x, 0)
            return step_x, 0, direct
    if direct == 1:
        transfer_contour(cnt, 0, step_y)
        if intersect_contours(cnt, poly):
            transfer_contour(cnt, 0, -step_y)
            direct = 2
        else:
            transfer_contour(cnt, 0, -step_y)
            return 0, step_y, direct
    if direct == 2:
        transfer_contour(cnt, -step_x, 0)
        if intersect_contours(cnt, poly):
            transfer_contour(cnt, step_x, 0)
            direct = 3
        else:
            transfer_contour(cnt, step_x, 0)
            return -step_x, 0, direct
    if direct == 3:
        transfer_contour(cnt, 0, -step_y)
        if intersect_contours(cnt, poly):
            transfer_contour(cnt, 0, step_y)
            direct = 0
        else:
            transfer_contour(cnt, 0, step_y)
            return 0, -step_y, direct

    return 0, 0, direct
