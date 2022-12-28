import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.morphology import binary_closing
from skimage.filters import gaussian
import cv2

from src.alg.geometric_operations import nested_poly


# we pre-process all images
def preprocess_data(img):
    # convert to gray gradations
    img_gray = rgb2gray(img)
    # gaussian blur to remove noise
    img_gray = gaussian(img_gray, sigma=3, channel_axis=True)
    # we perform binarization using the threshold to the otsu
    thresh_otsu = threshold_otsu(img_gray)
    otsu = img_gray <= thresh_otsu
    # apply binary closure to close gaps
    close_otsu = binary_closing(otsu, footprint=np.ones((20, 20)))
    # for reading opencv2
    cv_close_otsu = img_as_ubyte(close_otsu)
    # find contours in processed image
    contours, hierarchy = cv2.findContours(cv_close_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort our contours that find bigest area
    sort_contours = list(reversed(sorted(contours, key=cv2.contourArea)))
    obj_contours = []
    obj_areas = []
    # at the processing stage, cases were noticed when small outliers were present inside other objects,
    # they were removed by checking whether they lie strictly inside other object contours
    for check_contour in sort_contours:
        rubbish = False
        for contour in sort_contours:
            if nested_poly(check_contour, contour):
                rubbish = True
                break
        if not rubbish:
            obj_contours.append(check_contour)
            obj_areas.append(cv2.contourArea(check_contour))

    # due to possible noise in the images,
    # very small objects stood out somewhere at the edges,
    # they were removed through small area values
    average_obj_area = sum(obj_areas) / len(obj_areas)
    i = 0
    while i != len(obj_areas):
        if obj_areas[i] < average_obj_area / 10:
            del obj_areas[i]
            del obj_contours[i]
        else:
            i += 1

    print("Object contours count: {}\n".format(len(obj_contours)))
    return obj_contours
