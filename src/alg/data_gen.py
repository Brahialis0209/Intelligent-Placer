from glob import glob
import os
import cv2



def get_test_poly():
    poly_1_1 = [[3, 4], [17, 4], [17, 21], [3, 21]]  # sm
    poly_1_2 = [[1, 19], [19, 19], [19, 29], [1, 29]]  # sm
    poly_1_3 = [[5, 19], [15, 19], [15, 29], [5, 29]]  # sm
    poly_1_4 = [[3, 19], [17, 19], [17, 29], [3, 29]]  # sm
    test_poly = []
    test_poly.append(poly_1_4)
    return test_poly


def get_test_data(path):
    titles = []
    images = []
    format_length = 4

    for image_path in glob(os.path.join(path, "*.jpg")):
        tittle = os.path.basename(image_path)[:-format_length]
        if len(tittle) > 1 and tittle[-2] == '_':
            continue
        if tittle[0] != '1':
            continue
        titles.append(tittle)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images.append(image_bgr)
    return images, titles