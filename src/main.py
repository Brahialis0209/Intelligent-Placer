import os
import sys

from src.alg.data_gen import get_test_data, get_test_poly
from src.alg.geometric_operations import *
from src.alg.preprocessing import preprocess_data
from src.alg.visualization import draw_contours_and_poly, show_img, red_bgr_color, contours_curve_dim

sys.path.append('../')


# start main algorithm
def placer_start(contours_np, img, polygon_sm):
    # use our class for contours
    contours = []
    for cnt in contours_np:
        new_cnt = Contour(cnt)
        contours.append(new_cnt)
    polygon = Contour(polygon_sm, True, img.shape)

    img2 = img.copy()
    # to avoid hardcode, we declare variables before the start of the function
    rotate_angle = 30
    max_rot = int(360 / rotate_angle)
    max_transfers = 20
    max_step_x = 200
    max_step_y = 200
    # place the contours of objects in turn
    for contour_num, cnt in enumerate(contours):
        # replace out contour to new position in polygon, all nodes will be in polygon
        transfer_to_poly(cnt, polygon)
        # for debug
        cv2.drawContours(img2, cnt.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
        cv2.drawContours(img2, polygon.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
        show_img(img2)
        # for the first contour, it makes no sense to check its intersection with other contours
        if contour_num == 0:
            continue
        # for nested polygon it makes no sense to check rotations
        if not polygon_in_polygon(cnt, contours, contour_num):
            # try rotation until successful conformation
            rot_count = 0
            while intersect_contours(cnt, polygon) or intersect_other_contours(cnt, contours,
                                                                               contour_num) or polygon_in_polygon(cnt,
                                                                                                                  contours,
                                                                                                                  contour_num):
                if rot_count < max_rot:
                    rotate_contour(cnt, grad_ro_rad(rotate_angle))
                    rot_count += 1
                else:
                    break
        # if we could not find the correct conformation
        if intersect_contours(cnt, polygon) or intersect_other_contours(cnt, contours,
                                                                        contour_num) or polygon_in_polygon(cnt,
                                                                                                           contours,
                                                                                                           contour_num):
            transfer_count = 0
            # move one step and start to rotate, so a certain number of times
            direct = 0
            while transfer_count < max_transfers:
                #  generate step  (x and y range ) and transfer contour (all nodes will be in polygon)
                direct = generate_step_to_transfer(max_step_x, max_step_y, cnt, polygon, direct)
                # for debug
                cv2.drawContours(img2, cnt.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
                cv2.drawContours(img2, polygon.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
                show_img(img2)

                if intersect_other_contours(cnt, contours, contour_num):
                    rot_count = 0
                    # try rotation until successful conformation
                    while intersect_contours(cnt, polygon) or intersect_other_contours(cnt, contours, contour_num):
                        if rot_count < max_rot:
                            rotate_contour(cnt, grad_ro_rad(rotate_angle))
                            rot_count += 1
                        else:
                            break
                    # by rotations got the correct position
                    if rot_count < max_rot and not polygon_in_polygon(cnt, contours, contour_num):
                        break
                elif polygon_in_polygon(cnt, contours, contour_num):
                    transfer_count += 1
                    continue
                else:
                    break
                transfer_count += 1
            if transfer_count == max_transfers:
                return False

    # for debug
    img_copy = img.copy()
    draw_contours_and_poly(img_copy, contours, polygon)
    show_img(img_copy)
    return True


def main():
    # use relative paths
    date_path = os.path.join(os.getcwd(), '../data', 'dataset', 'mark')
    # read data
    images, titles = get_test_data(date_path)
    poly = get_test_poly()

    # we pre-process all images and start placer for every image
    obj_contours = []
    for image in images:
        cnts = preprocess_data(image)
        obj_contours.append(cnts)
        status = placer_start(cnts, image, poly)
        if status:
            print("True!")
        else:
            print("False!")


if __name__ == "__main__":
    main()
