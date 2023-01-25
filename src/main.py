import os
import sys

from src.alg.data_gen import get_test_data, get_test_poly
from src.alg.geometric_operations import *
from src.alg.preprocessing import preprocess_data
from src.alg.visualization import draw_contours_and_poly, show_img, red_bgr_color, contours_curve_dim, save_img

sys.path.append('../')


# start main algorithm
def placer_start(contours_np, img, polygon_sm, date_path, title, iter):
    # for debug
    back_path = os.path.join(os.getcwd(), '../data', 'dataset', 'one')
    image_bgr = cv2.imread(os.path.join(back_path, "back.jpg"), cv2.IMREAD_COLOR)
    # use our class for contours
    contours = []
    for cnt in contours_np:
        new_cnt = Contour(cnt)
        contours.append(new_cnt)
    polygon = Contour(polygon_sm, True, img.shape)

    # to avoid hardcode, we declare variables before the start of the function
    rotate_angle = 45
    max_rot = int(360 / rotate_angle)
    max_transfers = 20
    max_step_x = int(img.shape[1] / 20)
    max_step_y = int(img.shape[0] / 20)
    process_contours = []
    # place the contours of objects in turn
    for contour_num, cnt in enumerate(contours):
        process_contours.append(cnt)
        # draw_contours_and_poly(image_bgr.copy(), process_contours, polygon)
        # replace out contour to new position in polygon, all nodes will be in polygon
        if contour_num == 0:
            opt_transfer_to_poly(cnt, polygon)
        else:
            transfer_to_poly(cnt, polygon)
        # for debug
        # cv2.drawContours(img2, cnt.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
        # cv2.drawContours(img2, polygon.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
        # show_img(img2)

        # draw_contours_and_poly(image_bgr.copy(), process_contours, polygon)
        # for the first contour, it makes no sense to check its intersection with other contours
        if contour_num == 0:
            continue
        contour_in_contour_flag = False
        non_opt_conf_flag = False
        # for nested polygon it makes no sense to check rotations
        if not contour_in_contours(cnt, process_contours, contour_num):
            # try rotation until successful conformation
            rot_count = 0
            while intersect_contours(cnt, polygon) or intersect_other_contours(cnt, process_contours,
                                                                               contour_num):
                if rot_count < max_rot:
                    rotate_contour(cnt, grad_ro_rad(rotate_angle))
                    rot_count += 1
                else:
                    non_opt_conf_flag = True
                    break
        else:
            contour_in_contour_flag = True
        # if we could not find the correct conformation
        if contour_in_contour_flag or non_opt_conf_flag or contour_in_contours(cnt, process_contours, contour_num):
            transfer_count = 0
            # move one step and start to rotate, so a certain number of times
            direct = 0
            while transfer_count < max_transfers:
                #  generate step  (x and y range ) and transfer contour (all nodes will be in polygon)
                direct = generate_step_to_transfer(max_step_x, max_step_y, cnt, polygon, direct)
                # for debug
                # cv2.drawContours(img2, cnt.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
                # cv2.drawContours(img2, polygon.get_cv_contour(), -1, red_bgr_color, contours_curve_dim)
                # show_img(img2)


                # draw_contours_and_poly(image_bgr.copy(), process_contours, polygon)

                if intersect_other_contours(cnt, process_contours, contour_num):
                    rot_count = 0
                    # try rotation until successful conformation
                    while intersect_contours(cnt, polygon) or intersect_other_contours(cnt, process_contours, contour_num):
                        if rot_count < max_rot:
                            rotate_contour(cnt, grad_ro_rad(rotate_angle))
                            rot_count += 1
                        else:
                            break
                    # by rotations got the correct position
                    if rot_count < max_rot and not contour_in_contours(cnt, process_contours, contour_num):
                        break
                elif contour_in_contours(cnt, process_contours, contour_num):
                    transfer_count += 1
                    continue
                else:
                    break
                transfer_count += 1
            if transfer_count == max_transfers:
                return False

    # for debug
    # img_copy = img.copy()
    # draw_contours_and_poly(image_bgr.copy(), process_contours, polygon)
    save_img(image_bgr.copy(), process_contours, polygon, date_path + "/" + title + "_" + str(iter) + '_res.png')
    # show_img(img_copy)
    return True


def main():
    # use relative paths
    date_path = os.path.join(os.getcwd(), '../data', 'dataset', 'mark')
    # read data
    images, titles = get_test_data(date_path)
    poly = get_test_poly()
    fail_count = 0
    success_count = 0
    total_count = 0
    contouts = []
    for id, image in enumerate(images):
        cnts = preprocess_data(image)
        contouts.append(cnts)

    # we pre-process all images and start placer for every image
    for id, image in enumerate(images):
        cnts = contouts[id]
        for iter in range(5):
            print("Image name: " + titles[id] + " Iteration: " + str(iter))
            status = placer_start(cnts, image, [poly[id]], date_path, titles[id], iter)
            if status:
                success_count += 1
                print("True!")
            else:
                fail_count += 1
                print("False!")
            total_count += 1
    print("Success rate: " + str(success_count / total_count * 100) + "%")
    print("Failure rate: " + str(fail_count / total_count * 100) + "%")


if __name__ == "__main__":
    main()
