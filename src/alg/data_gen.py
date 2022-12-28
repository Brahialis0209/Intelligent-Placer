from glob import glob
import os
import cv2



def get_test_poly():
    poly_1 = [[3, 19], [17, 19], [17, 29], [3, 29]]  # sm
    poly_4 = [[2, 5], [19, 5], [20, 25], [3, 27]]  # sm
    poly_4 = [[2, 10], [17, 10], [18, 25], [3, 27]]  # sm
    test_poly = []
    test_poly.append(poly_4)
    return test_poly


def get_test_data(path):
    titles = []
    images = []
    format_length = 4

    for image_path in glob(os.path.join(path, "*.jpg")):
        tittle = os.path.basename(image_path)[:-format_length]
        if len(tittle) > 1 and tittle[-2] == '_':
            continue
        if tittle[0] != '4':
            continue
        titles.append(tittle)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images.append(image_bgr)
    return images, titles