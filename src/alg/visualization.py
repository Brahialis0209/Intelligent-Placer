from matplotlib import pyplot as plt
import cv2

from src.alg.structures import Contour

red_bgr_color = (0, 0, 255)
contours_curve_dim = 10


def show_img(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def draw_contours_and_poly(img, cnts: list, poly: Contour):
    for cnt in cnts:
        cv2.drawContours(img, cnt.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
    cv2.drawContours(img, poly.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
    show_img(img)
