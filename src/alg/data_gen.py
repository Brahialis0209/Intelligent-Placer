from glob import glob
import os
import cv2



def get_test_poly():
    # poly_1 = [[3, 19], [17, 19], [17, 29], [3, 29]]  # sm
    # # poly_2 = [[3, 1], [19, 1], [19, 25], [4, 25]]  # sm
    poly_3 = [[5, 1], [17, 1], [17, 22], [5, 20]]  # sm
    # poly_4 = [[2, 10], [17, 10], [18, 25], [3, 27]]  # sm
    # poly_5 = [[2, 9], [19, 9], [18, 25], [1, 27]]  # sm
    # poly_6 = [[9, 1], [15, 1], [15, 25], [5, 25]]  # sm
    # poly_7 = [[1, 3], [20, 2], [19, 28], [2, 29]]  # sm
    # test_poly = [poly_1, poly_2, poly_3, poly_4, poly_5, poly_6, poly_7]
    # test_poly = [poly_1, poly_3, poly_4, poly_5, poly_6, poly_7]
    test_poly = [poly_3]
    return test_poly


def get_test_data(path):
    titles = []
    images = []
    format_length = 4

    for image_path in glob(os.path.join(path, "*.jpg")):
        tittle = os.path.basename(image_path)[:-format_length]
        if len(tittle) > 1 and tittle[-2] == '_':
            continue
        # if tittle[0] != '4' or tittle[0] != '4':
        #     continue
        if tittle[0] == '8' or tittle[0] == '2' or tittle[0] != '3':
            continue
        titles.append(tittle)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images.append(image_bgr)
    return images, titles