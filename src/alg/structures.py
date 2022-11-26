import cv2
import numpy as np
from math import sqrt


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Segment:
    def __init__(self, _p1: Point, _p2: Point):
        self.p1 = _p1
        self.p2 = _p2


def convert_sm_to_px(x_old: int, y_old: int, size: tuple):
    size_x = size[1]
    size_y = size[0]
    x_a4 = 21
    y_a4 = 29.7
    x = 100 / x_a4 * x_old
    new_x = size_x * (x / 100)
    y = 100 / y_a4 * y_old
    new_y = size_y * (y / 100)
    return Point(int(new_x), int(new_y))


def update_maxX(max_old: int, size: tuple):
    size_x = size[1]
    x_a4 = 21
    x = 100 / x_a4 * max_old
    new_x = size_x * (x / 100)
    return new_x


def distance(p1: Point, p2: Point):
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


class Contour:
    def __init__(self, contour, polygon=False, img_size=(0, 0, 0)):
        if not polygon:
            self.cv_contour = contour
            self.contour = []
            self.max_x = 0
            list_contour = contour.tolist()
            for i in range(len(list_contour)):
                x1 = list_contour[i][0][0]
                y1 = list_contour[i][0][1]
                i_next = i + 1
                if i_next == len(list_contour):
                    i_next = 0
                x2 = list_contour[i_next][0][0]
                y2 = list_contour[i_next][0][1]
                x = max(x1, x2)
                if x > self.max_x:
                    self.max_x = x
                p1 = Point(x1, y1)
                p2 = Point(x2, y2)
                s = Segment(p1, p2)
                self.contour.append(s)
        else:
            self.cv_contour = np.array(contour)
            self.contour = []
            self.max_x = 0
            list_contour = contour[0]
            for i in range(len(list_contour)):
                x1 = list_contour[i][0]
                y1 = list_contour[i][1]
                i_next = i + 1
                if i_next == len(list_contour):
                    i_next = 0
                x2 = list_contour[i_next][0]
                y2 = list_contour[i_next][1]
                x = max(x1, x2)
                if x > self.max_x:
                    self.max_x = x
                p1 = convert_sm_to_px(x1, y1, img_size)
                p2 = convert_sm_to_px(x2, y2, img_size)
                s = Segment(p1, p2)
                self.contour.append(s)
            self.max_x = update_maxX(self.max_x, img_size)

    def recalc_maxX(self):
        for s in self.contour:
            p1 = s.p1
            p2 = s.p2
            x1 = p1.x
            x2 = p2.x
            x = max(x1, x2)
            if x > self.max_x:
                self.max_x = x
        return self.max_x

    def get_cv_contour(self):
        new_cv_contour = []
        for s in self.contour:
            p1 = s.p1
            p2 = s.p2
            new_cv_contour.append([p1.x, p1.y])
            new_cv_contour.append([p2.x, p2.y])
        self.cv_contour = np.array([new_cv_contour])
        return self.cv_contour

    def get_moments(self):
        # calculate moments for each contour
        M = cv2.moments(self.get_cv_contour())
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def get_min_dist_to_segments_from_point(self, p: Point):
        x = p.x
        y = p.y
        min = 1e9
        for s in self.contour:
            x_c = int((s.p1.x + s.p2.x) / 2)
            y_c = int((s.p1.y + s.p2.y) / 2)
            p1 = Point(x, y)
            p2 = Point(x_c, y_c)
            dst = distance(p1, p2)
            if dst < min:
                min = dst
        return min

    def get_average_dist_to_segments_from_center(self, p: Point):
        xM, yM = self.get_moments()
        sum = 0
        for s in self.contour:
            x_c = int((s.p1.x + s.p2.x) / 2)
            y_c = int((s.p1.y + s.p2.y) / 2)
            p1 = Point(xM, yM)
            p2 = Point(x_c, y_c)
            sum += distance(p1, p2)
        return int(sum / len(self.contour))