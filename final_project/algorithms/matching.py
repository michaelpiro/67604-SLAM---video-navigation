import cv2
import numpy as np
from final_project.Inputs import read_images


def read_and_extract_matches(index=0):
    """
    read and extract the matches from 1 pair of stereo images
    :param index: index of the image
    :return: images, matching kp, and matches
    """
    left_img, right_img = read_images(index)
    left_kp, left_desc = FEATURE.detectAndCompute(left_img, None)
    right_kp, desc_right = FEATURE.detectAndCompute(right_img, None)
    matches = MATCHER.match(left_desc, desc_right)
    return left_img, right_img, left_kp, right_kp, matches


def get_akaze_matcher_lr_matcher():
    FEATURE = cv2.AKAZE_create(threshold=0.0008, nOctaves=4, nOctaveLayers=4)
    bf_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    MATCHER_LEFT_RIGHT = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    MATCHER = bf_matcher
    return FEATURE, MATCHER_LEFT_RIGHT, MATCHER


def get_sift_matcher_lr_matcher():
    """
    return a SIFT matcher
    """
    FEATURE = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6,
                              enable_precise_upscale=True)
    MATCHER = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    MATCHER_LEFT_RIGHT = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    return FEATURE, MATCHER_LEFT_RIGHT, MATCHER


def extract_kps_descs_matches(img_0, img1):
    """
    extract the keypoints and their descriptors from two images
    """
    kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    matches = MATCHER_LEFT_RIGHT.match(desc0, desc1)
    return kp0, kp1, desc0, desc1, matches


def extract_inliers_outliers(kp_left, kp_right, matches):
    """
    extract the inliers and outliers from the kp. version of ex4_v2
    """
    inliers = []
    outliers = []

    for match_index in range(len(matches)):
        ind_left = matches[match_index].queryIdx
        ind_right = matches[match_index].trainIdx
        point_left = kp_left[ind_left].pt
        point_right = kp_right[ind_right].pt

        # Use numpy arrays for comparisons
        good_map1 = abs(point_left[1] - point_right[1]) < 2
        good_map2 = point_left[0] > point_right[0] + 2
        if good_map1 and good_map2:
            inliers.append(match_index)
        else:
            outliers.append(match_index)

    return np.array(inliers), np.array(outliers)


FEATURE, MATCHER_LEFT_RIGHT, MATCHER = get_sift_matcher_lr_matcher()
