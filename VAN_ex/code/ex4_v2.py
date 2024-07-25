import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Sequence, Optional
from timeit import default_timer as timer
from tracking_database import TrackingDB, Link, MatchLocation
from tqdm import tqdm
# from ex4_2_to_6 import q_4_2, q_4_3, q_4_4, q_4_5, q_4_6, q_4_7, find_longest_track_frames

from ex3 import P, Q, K, M1, M2, read_extrinsic_matrices, rodriguez_to_mat
from ex5 import get_inverse, translate_later_t_to_older_t
import ex2
import random

NO_ID = -1

# FEATURE = cv2.SIFT_create()
FEATURE = cv2.SIFT_create()
# FEATURE.setNOctaves(2)
# FEATURE.setThreshold(0.008)
# print(FEATURE.getThreshold())
bf_matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
MATCHER = flann
# MATCHER = bf_matcher

# MATCHER2 = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
# DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'
DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))


# LEN_DATA_SET = 1000
# P = ex3.P
# Q = ex3.Q
# K = ex3.K
# M1 = ex3.M1
# M2 = ex3.M2


class Match:
    def __init__(self, queryIdx, trainIdx, kp1, kp2):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.kp1 = kp1
        self.kp2 = kp2

    def get_query_kp(self):
        return self.kp1

    def get_train_kp(self):
        return self.kp2

    def get_query_idx(self):
        return self.queryIdx

    def get_train_idx(self):
        return self.trainIdx

    @property
    def x1(self):
        return self.kp1.pt[0]

    @property
    def y1(self):
        return self.kp1.pt[1]

    @property
    def x2(self):
        return self.kp2.pt[0]

    @property
    def y2(self):
        return self.kp2.pt[1]


def extract_kps_descs_matches(img_0, img1):
    kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    matches = MATCHER.match(desc0, desc1)
    # kp0_new = []
    # kp1_new = []
    # desc0_new = []
    # desc1_new = []
    # for match in matches:
    #     ind0 = match.queryIdx
    #     ind1 = match.trainIdx
    #     kp0_new.append(kp0[ind0])
    #     kp1_new.append(kp1[ind1])
    #     desc0_new.append(desc0[ind0])
    #     desc1_new.append(desc1[ind1])
    #
    # return kp0_new, kp1_new, desc0_new, desc1_new, matches
    return kp0, kp1, desc0, desc1, matches


def extract_inliers_outliers(kp_left, kp_right, matches):
    kp_left_pts = np.array([kp.pt for kp in kp_left])
    kp_right_pts = np.array([kp.pt for kp in kp_right])

    inliers = []
    outliers = []

    for match_index in range(len(matches)):
        ind_left = matches[match_index].queryIdx
        ind_right = matches[match_index].trainIdx
        point_left = kp_left[ind_left].pt
        point_right = kp_right[ind_right].pt

        # Use numpy arrays for comparisons
        good_map1 = abs(point_left[1] - point_right[1]) < 2
        good_map2 = point_left[0] > point_right[0]
        # good_map2 = True
        if good_map1 and good_map2:
            inliers.append(match_index)
        else:
            outliers.append(match_index)

    return np.array(inliers), np.array(outliers)


def extract_agreements(matches_lr_0, in_lr_0, in_prev_idx, matches_l0_l1, matches_lr1, in_lr1, in_cur_idx):
    """
    return list of indexes (lr0, lr1, l0_l1) that agree with each other, and the indexes of the matches that are inliers
    :param matches_lr_0:
    :param in_lr_0:
    :param matches_l0_l1:
    :param matches_lr1:
    :param in_lr1:
    :return:
    """
    # if len(matches_lr_0) != len(in_lr_0):
    #     print("bad")
    # assert len(matches_lr_0) == len(in_lr_0)
    # assert len(matches_lr1) == len(in_lr1)
    # Create dictionaries for quick lookup
    # a dict that it's keys are the keypoint of the match in the left image,
    # and the value is the index of the match in the matches array
    # dict_lr_0 = {match.queryIdx: i for i, match in enumerate(matches_lr_0) if in_lr_0[i]}
    # dict_lr_1 = {match.queryIdx: i for i, match in enumerate(matches_lr1) if in_lr1[i]}

    agreement_idx = []
    matches_lr0_lr1_ll = []
    for i, match_l_l in enumerate(matches_l0_l1):
        feature_index_left = match_l_l.queryIdx
        feature_index_right = match_l_l.trainIdx

        real_featuer_idx_left0 = in_prev_idx[feature_index_left]
        real_featuer_idx_left1 = in_cur_idx[feature_index_right]
        # Check if the match exists in both dictionaries
        # if real_featuer_idx_left0 in dict_lr_0 and real_featuer_idx_left1 in dict_lr_1:
        # i0 = dict_lr_0[real_featuer_idx_left0]
        # i1 = dict_lr_1[real_featuer_idx_left1]
        agreement_idx.append((real_featuer_idx_left0, real_featuer_idx_left1, i))
        matches_lr0_lr1_ll.append(
            (matches_lr_0[real_featuer_idx_left0], matches_lr1[real_featuer_idx_left1], match_l_l))
        # else:
        #     print("bad_match")

        # dict_lr_1.pop(kp_left_1)
        # dict_lr_0.pop(kp_left_0)

    return agreement_idx, matches_lr0_lr1_ll


def find_concensus_points_and_idx(matches_lr_0, in_lr_0, in_prev_idx, matches_l0_l1, matches_lr1, in_lr1, in_cur_idx):
    """
    return list of indexes (lr0, lr1, l0_l1) that agree with each other, and the indexes of the matches that are inliers
    :param matches_lr_0:
    :param in_lr_0:
    :param matches_l0_l1:
    :param matches_lr1:
    :param in_lr1:
    :return:
    """
    # if len(matches_lr_0) != len(in_lr_0):
    #     print("bad")
    # assert len(matches_lr_0) == len(in_lr_0)
    # assert len(matches_lr1) == len(in_lr1)
    # Create dictionaries for quick lookup
    # a dict that it's keys are the keypoint of the match in the left image,
    # and the value is the index of the match in the matches array
    dict_lr_0 = {match.queryIdx: i for i, match in enumerate(matches_lr_0) if in_lr_0[i]}
    dict_lr_1 = {match.queryIdx: i for i, match in enumerate(matches_lr1) if in_lr1[i]}

    agreement_idx = []
    matches_lr0_lr1_ll = []
    for i, match_l_l in enumerate(matches_l0_l1):

        feature_index_left = match_l_l.queryIdx
        feature_index_right = match_l_l.trainIdx

        real_featuer_idx_left0 = in_prev_idx[feature_index_left]
        real_featuer_idx_left1 = in_cur_idx[feature_index_right]
        # Check if the match exists in both dictionaries
        if real_featuer_idx_left0 in dict_lr_0 and real_featuer_idx_left1 in dict_lr_1:
            i0 = dict_lr_0[real_featuer_idx_left0]
            i1 = dict_lr_1[real_featuer_idx_left1]
            agreement_idx.append((i0, i1, i))
            matches_lr0_lr1_ll.append((matches_lr_0[i0], matches_lr1[i1], match_l_l))
        else:
            print("bad_match")

            # dict_lr_1.pop(kp_left_1)
            # dict_lr_0.pop(kp_left_0)

    return agreement_idx, matches_lr0_lr1_ll


def transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
                             ordered_cur_left_pix_values, ordered_cur_right_pix_values, x_condition=True):
    T_4x4 = np.vstack((T, np.array([0, 0, 0, 1])))
    points_4d = np.hstack((traingulated_pts, np.ones((traingulated_pts.shape[0], 1)))).T
    l1_4d_points = (T_4x4 @ points_4d)
    to_the_right = (K @ M2)

    transform_to_l0_points = (P @ points_4d).T
    transform_to_l0_points = transform_to_l0_points / transform_to_l0_points[:, 2][:, np.newaxis]
    real_y = prev_left_pix_values[:, 1]
    agree_l0 = np.abs(transform_to_l0_points[:, 1] - real_y) < 2
    agree_x = np.abs(transform_to_l0_points[:, 0] - prev_left_pix_values[:, 0]) < 2
    agree_l0 = np.logical_and(agree_l0, agree_x)

    transform_to_r0_points = (to_the_right @ points_4d).T
    transform_to_r0_points = transform_to_r0_points / transform_to_r0_points[:, 2][:, np.newaxis]
    real_y = prev_right_pix_values[:, 1]
    agree_r0 = np.abs(transform_to_r0_points[:, 1] - real_y) < 2
    agree_x = np.abs(transform_to_r0_points[:, 0] - prev_right_pix_values[:, 0]) < 2
    agree_r0 = np.logical_and(agree_r0, agree_x)
    if x_condition:
        real_x_l = prev_left_pix_values[:, 0]
        real_x_r = prev_right_pix_values[:, 0]
        cond_x = real_x_l > real_x_r
    else:
        cond_x = np.ones_like(agree_r0)
    agree_0 = np.logical_and(agree_r0, cond_x, agree_l0)
    # agree_0 = np.array([True] * len(triangulate_points()))

    transformed_to_l1_points = (K @ l1_4d_points[:3, :]).T
    transformed_to_l1_points = transformed_to_l1_points / transformed_to_l1_points[:, 2][:, np.newaxis]
    real_y = ordered_cur_left_pix_values[:, 1]
    agree_l1 = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2
    agree_x = np.abs(transformed_to_l1_points[:, 0] - ordered_cur_left_pix_values[:, 0]) < 2
    agree_l1 = np.logical_and(agree_l1, agree_x)

    transformed_to_r1_points = (to_the_right @ l1_4d_points).T
    transformed_to_r1_points = transformed_to_r1_points / transformed_to_r1_points[:, 2][:, np.newaxis]
    real_y = ordered_cur_right_pix_values[:, 1]
    agree_r1 = np.abs(transformed_to_r1_points[:, 1] - real_y) < 2
    agree_x = np.abs(transformed_to_r1_points[:, 0] - ordered_cur_right_pix_values[:, 0]) < 2
    agree_r1 = np.logical_and(agree_r1, agree_x)
    if x_condition:
        real_x_l = ordered_cur_left_pix_values[:, 0]
        real_x_r = ordered_cur_right_pix_values[:, 0]
        cond_x = real_x_l > real_x_r
    else:
        cond_x = np.ones_like(agree_r1)
    agree_1 = np.logical_and(agree_r1, cond_x, agree_l1)
    return np.logical_and(agree_0, agree_1)


#
# def transformation_agreement(T, traingulated_pts, ordered_cur_left_pix_values):
#     T_4x4 = np.vstack((T, np.array([0, 0, 0, 1])))
#     points_4d = np.hstack((traingulated_pts, np.ones((traingulated_pts.shape[0], 1)))).T
#     l1_4d_points = (T_4x4 @ points_4d)
#     to_the_right = (K @ M2)
#
#
#
#     transformed_to_l1_points = (K @ l1_4d_points[:3, :]).T
#     transformed_to_l1_points = transformed_to_l1_points / transformed_to_l1_points[:, 2][:, np.newaxis]
#     real_y = ordered_cur_left_pix_values[:, 1]
#     agree_y = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2
#     agree_x = np.abs(transformed_to_l1_points[:, 0] - ordered_cur_left_pix_values[:, 0]) < 2
#     return np.logical_and(agree_y, agree_x)


def triangulate_points(matches, kp_l, kp_r, p, q):
    """
    Triangulate the matched points using OpenCV
    :param inliers:
    :return:
    """
    x = np.zeros((len(matches), 3))
    for i in range(len(matches)):
        left_idx = matches[i].queryIdx
        right_idx = matches[i].trainIdx
        left_point = kp_l[left_idx].pt
        right_point = kp_r[right_idx].pt
        x[i] = ex2.linear_least_squares_triangulation(p, q, left_point, right_point).T
        # x_left, x_right, y = links[i].x_left, links[i].x_right, links[i].y
        # p_left, p_right = (x_left, y), (x_right, y)
        # x[i] = ex2.linear_least_squares_triangulation(p, q, p_left, p_right)
    return x


def triangulate_links(links, p, q):
    """
    Triangulate the matched points using OpenCV
    :param inliers:
    :return:
    """
    x = np.zeros((len(links), 3))
    for i, link in enumerate(links):
        left_point = link.x_left, link.y
        right_point = link.x_right, link.y
        x[i] = ex2.linear_least_squares_triangulation(p, q, left_point, right_point).T
    return x


def get_pixels_from_matches(matches, kp_1, kp_2):
    pixels_first = []
    pixels_second = []
    for match in matches:
        pixels_first.append(kp_1[match.queryIdx].pt)
        pixels_second.append(kp_2[match.trainIdx].pt)
    return pixels_first, pixels_second


def get_pixels_from_links(links):
    pixels_first = []
    pixels_second = []
    for link in links:
        pixels_first.append((link.x_left, link.y))
        pixels_second.append((link.x_right, link.y))
    return pixels_first, pixels_second


def ransac_pnp_for_tracking_db(matches_l_l, prev_links, cur_links, inliers_percent):
    """ Perform RANSAC to find the best transformation"""
    # best_inliers = 0
    # best_T = None
    # best_matches_idx = None
    # diff_coeff = np.zeros((5, 1))
    #
    #
    # prev_left_pix_values = []
    # prev_right_pix_values = []
    # ordered_cur_left_pix_values = []
    # ordered_cur_right_pix_values = []
    # for link in links_prev:
    #     prev_left_pix_values.append((link.x_left, link.y))
    #     prev_right_pix_values.append((link.x_right, link.y))
    # for match in matches:
    #     link_index = match.trainIdx
    #     ordered_cur_left_pix_values.append((links_cur[link_index].x_left, links_cur[link_index].y))
    #     ordered_cur_right_pix_values.append((links_cur[link_index].x_right, links_cur[link_index].y))
    # prev_left_pix_values = np.array(prev_left_pix_values)
    # prev_right_pix_values = np.array(prev_right_pix_values)
    # ordered_cur_left_pix_values = np.array(ordered_cur_left_pix_values)
    # ordered_cur_right_pix_values = np.array(ordered_cur_right_pix_values)
    #
    # for i in range(ex3.RANSAC_ITERATIONS):
    #
    #     # Randomly select 4 points in the world coordinate system
    #     random_idx = np.random.choice(len(traingulated_pts), 4, replace=False)
    #     random_world_points = traingulated_pts[random_idx]
    #     random_cur_l_pixels = ordered_cur_left_pix_values[random_idx]
    #
    #     # solve PnP problem to get the transformation
    #     success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_cur_l_pixels, K,
    #                                                                 distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
    #     if success:
    #         T = ex3.rodriguez_to_mat(rotation_vector, translation_vector)
    #     else:
    #         continue
    #
    #     points_agreed = transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
    #                                              ordered_cur_left_pix_values, ordered_cur_right_pix_values,
    #                                              x_condition=True)
    #
    #     inliers_idx = np.where(points_agreed == True)
    #     if np.sum(points_agreed) > best_inliers:
    #         best_inliers = np.sum(points_agreed)
    #         best_T = T
    #         best_matches_idx = inliers_idx
    #
    # return best_T, best_matches_idx, ordered_cur_left_pix_values[best_matches_idx[0]]
    # concensus_idx = np.array(concensus_idx)
    # print(f"len concensus_idx: {len(concensus_idx)}")
    # concensus_matches_l_l_idx = concensus_idx[:, 2]
    # concensus_matches_prev_idx = concensus_idx[:, 0]
    # concensus_matches_cur_idx = concensus_idx[:, 1]
    #
    # matches_prev = np.array(matches_prev)
    # matches_cur = np.array(matches_cur)
    # matches_l_l = np.array(matches_l_l)

    # concensus_matches_l_l = matches_l_l[concensus_matches_l_l_idx]
    # concensus_matches_prev = matches_prev[concensus_matches_prev_idx]
    # concensus_matches_cur = matches_cur[concensus_matches_cur_idx]
    #
    # # points_3d = triangulate_points(concensus_matches_prev, kp_l_prev, kp_r_prev, P, Q)
    # points_to_triangulate = []
    # prev_left_pix_values = []
    # prev_right_pix_values = []
    # ordered_cur_left_pix_values = []
    # ordered_cur_right_pix_values = []

    ransac_iterations = calc_ransac_iteration(inliers_percent)

    filtered_links_cur = []
    filtered_links_prev = []
    for match in matches_l_l:
        link_index = match.trainIdx
        filtered_links_cur.append(cur_links[link_index])

        link_index = match.queryIdx
        filtered_links_prev.append(prev_links[link_index])

    points_3d = triangulate_links(filtered_links_prev, P, Q)
    prev_left_pix_values, prev_right_pix_values = get_pixels_from_links(filtered_links_prev)
    ordered_cur_left_pix_values, ordered_cur_right_pix_values = get_pixels_from_links(filtered_links_cur)

    prev_left_pix_values = np.array(prev_left_pix_values)
    prev_right_pix_values = np.array(prev_right_pix_values)
    ordered_cur_left_pix_values = np.array(ordered_cur_left_pix_values)
    ordered_cur_right_pix_values = np.array(ordered_cur_right_pix_values)

    #
    # for agreement in concensus_matches:
    #     match_lr_0, match_lr_1, match_l_l = agreement[0], agreement[1], agreement[2]
    #     points_to_triangulate.append(match_lr_0)

    # for match in matches:
    #     link_index = match.trainIdx
    #     ordered_cur_left_pix_values.append((links_cur[link_index].x_left, links_cur[link_index].y))
    #     ordered_cur_right_pix_values.append((links_cur[link_index].x_right, links_cur[link_index].y))

    diff_coeff = np.zeros((5, 1))
    best_inliers = 0
    best_T = None
    best_matches_idx = None

    for i in range(ransac_iterations):
        random_idx = np.random.choice(len(points_3d), 4, replace=False)
        random_world_points = points_3d[random_idx]
        # random_cur_l_pixels = np.array([(kp_l_cur[concensus_matches_cur_idx])[rand].pt for rand in random_idx])
        random_cur_l_pixels = ordered_cur_left_pix_values[random_idx]
        success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_cur_l_pixels, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)

        if success:
            T = rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue

        # points_agreed = transformation_agreement(T, points_3d, kp_l_prev, kp_r_prev, kp_l_cur, kp_r_cur,
        #                                          x_condition=True)

        points_agreed = transformation_agreement(T, points_3d, prev_left_pix_values, prev_right_pix_values,
                                                 ordered_cur_left_pix_values, ordered_cur_right_pix_values)

        num_inliers = np.sum(points_agreed)
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            # best_T = T
            best_matches_idx = np.where(points_agreed == True)[0]
    # print(best_inliers,len(concensus_idx))

    return best_matches_idx


def calc_ransac_iteration(inliers_percent):
    sec_prob = 0.9999999
    outliers_prob = 1 - (inliers_percent / 100)
    min_set_size = 4
    ransac_iterations = int(np.log(1 - sec_prob) / np.log(1 - np.power(1 - outliers_prob, min_set_size))) + 1
    return ransac_iterations


# def ransac_pnp_for_tracking_db(matches_l_l, matches_prev, matches_cur, concensus_idx, kp_l_prev, kp_r_prev, kp_l_cur,
#                                kp_r_cur, concensus_matches, inliers_percent):
#     """ Perform RANSAC to find the best transformation"""
#     # best_inliers = 0
#     # best_T = None
#     # best_matches_idx = None
#     # diff_coeff = np.zeros((5, 1))
#     #
#     #
#     # prev_left_pix_values = []
#     # prev_right_pix_values = []
#     # ordered_cur_left_pix_values = []
#     # ordered_cur_right_pix_values = []
#     # for link in links_prev:
#     #     prev_left_pix_values.append((link.x_left, link.y))
#     #     prev_right_pix_values.append((link.x_right, link.y))
#     # for match in matches:
#     #     link_index = match.trainIdx
#     #     ordered_cur_left_pix_values.append((links_cur[link_index].x_left, links_cur[link_index].y))
#     #     ordered_cur_right_pix_values.append((links_cur[link_index].x_right, links_cur[link_index].y))
#     # prev_left_pix_values = np.array(prev_left_pix_values)
#     # prev_right_pix_values = np.array(prev_right_pix_values)
#     # ordered_cur_left_pix_values = np.array(ordered_cur_left_pix_values)
#     # ordered_cur_right_pix_values = np.array(ordered_cur_right_pix_values)
#     #
#     # for i in range(ex3.RANSAC_ITERATIONS):
#     #
#     #     # Randomly select 4 points in the world coordinate system
#     #     random_idx = np.random.choice(len(traingulated_pts), 4, replace=False)
#     #     random_world_points = traingulated_pts[random_idx]
#     #     random_cur_l_pixels = ordered_cur_left_pix_values[random_idx]
#     #
#     #     # solve PnP problem to get the transformation
#     #     success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_cur_l_pixels, K,
#     #                                                                 distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
#     #     if success:
#     #         T = ex3.rodriguez_to_mat(rotation_vector, translation_vector)
#     #     else:
#     #         continue
#     #
#     #     points_agreed = transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
#     #                                              ordered_cur_left_pix_values, ordered_cur_right_pix_values,
#     #                                              x_condition=True)
#     #
#     #     inliers_idx = np.where(points_agreed == True)
#     #     if np.sum(points_agreed) > best_inliers:
#     #         best_inliers = np.sum(points_agreed)
#     #         best_T = T
#     #         best_matches_idx = inliers_idx
#     #
#     # return best_T, best_matches_idx, ordered_cur_left_pix_values[best_matches_idx[0]]
#     # concensus_idx = np.array(concensus_idx)
#     # print(f"len concensus_idx: {len(concensus_idx)}")
#     # concensus_matches_l_l_idx = concensus_idx[:, 2]
#     # concensus_matches_prev_idx = concensus_idx[:, 0]
#     # concensus_matches_cur_idx = concensus_idx[:, 1]
#     #
#     # matches_prev = np.array(matches_prev)
#     # matches_cur = np.array(matches_cur)
#     # matches_l_l = np.array(matches_l_l)
#
#     # concensus_matches_l_l = matches_l_l[concensus_matches_l_l_idx]
#     # concensus_matches_prev = matches_prev[concensus_matches_prev_idx]
#     # concensus_matches_cur = matches_cur[concensus_matches_cur_idx]
#     #
#     # # points_3d = triangulate_points(concensus_matches_prev, kp_l_prev, kp_r_prev, P, Q)
#     # points_to_triangulate = []
#     # prev_left_pix_values = []
#     # prev_right_pix_values = []
#     # ordered_cur_left_pix_values = []
#     # ordered_cur_right_pix_values = []
#
#     sec_prob = 0.9999999
#     outliers_prob = 1-(inliers_percent/100)
#     min_set_size = 4
#     ransac_iterations = int(np.log(1 - sec_prob) / np.log(1 - np.power(1 - outliers_prob, min_set_size))) + 1
#
#     concensus_matches = np.array(concensus_matches)
#     match_lr_0 = concensus_matches[:, 0]
#     match_lr_1 = concensus_matches[:, 1]
#     # match_left_left = concensus_matches[:, 2]
#
#     points_3d = triangulate_points(match_lr_0, kp_l_prev, kp_r_prev, P, Q)
#     prev_left_pix_values, prev_right_pix_values = get_pixels_from_matches(match_lr_0, kp_l_prev, kp_r_prev)
#     ordered_cur_left_pix_values, ordered_cur_right_pix_values = get_pixels_from_matches(match_lr_1, kp_l_cur, kp_r_cur)
#
#     prev_left_pix_values = np.array(prev_left_pix_values)
#     prev_right_pix_values = np.array(prev_right_pix_values)
#     ordered_cur_left_pix_values = np.array(ordered_cur_left_pix_values)
#     ordered_cur_right_pix_values = np.array(ordered_cur_right_pix_values)
#
#     #
#     # for agreement in concensus_matches:
#     #     match_lr_0, match_lr_1, match_l_l = agreement[0], agreement[1], agreement[2]
#     #     points_to_triangulate.append(match_lr_0)
#
#     # for match in matches:
#     #     link_index = match.trainIdx
#     #     ordered_cur_left_pix_values.append((links_cur[link_index].x_left, links_cur[link_index].y))
#     #     ordered_cur_right_pix_values.append((links_cur[link_index].x_right, links_cur[link_index].y))
#
#     diff_coeff = np.zeros((5, 1))
#     best_inliers = 0
#     best_T = None
#     best_matches_idx = None
#     kp_l_cur = np.array(kp_l_cur)
#
#     for i in range(ransac_iterations):
#         random_idx = np.random.choice(len(points_3d), 4, replace=False)
#         random_world_points = points_3d[random_idx]
#         # random_cur_l_pixels = np.array([(kp_l_cur[concensus_matches_cur_idx])[rand].pt for rand in random_idx])
#         random_cur_l_pixels = ordered_cur_left_pix_values[random_idx]
#         success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_cur_l_pixels, K,
#                                                                     distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
#
#         if success:
#             T = rodriguez_to_mat(rotation_vector, translation_vector)
#         else:
#             continue
#
#         # points_agreed = transformation_agreement(T, points_3d, kp_l_prev, kp_r_prev, kp_l_cur, kp_r_cur,
#         #                                          x_condition=True)
#
#         points_agreed = transformation_agreement(T, points_3d, prev_left_pix_values, prev_right_pix_values,
#                                                  ordered_cur_left_pix_values, ordered_cur_right_pix_values,
#                                                  x_condition=True)
#
#         num_inliers = np.sum(points_agreed)
#         if num_inliers > best_inliers:
#             best_inliers = num_inliers
#             best_T = T
#             best_matches_idx = np.where(points_agreed == True)[0]
#     # print(best_inliers,len(concensus_idx))
#     inliers_3d_points = points_3d[best_matches_idx]
#     # print(best_matches_idx)
#
#     # print(inliers_3d_points)
#     inliers_cur_points = ordered_cur_left_pix_values[best_matches_idx]
#     # inliers_cur_points = [(kp_l_cur[b])[0].pt for b in best_matches_idx]
#     # print(inliers_cur_points)
#
#     # assert len(inliers_3d_points) == len(inliers_cur_points)
#     assert len(inliers_3d_points) >= 4
#
#     success, rotation_vector, translation_vector = cv2.solvePnP(inliers_3d_points, inliers_cur_points, K,
#                                                                 distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
#     if success:
#         T = rodriguez_to_mat(rotation_vector, translation_vector)
#     else:
#         T = best_T
#
#     # global_cur_index = concensus_matches_cur_idx[best_matches_idx]
#     # global_prev_index = concensus_matches_prev_idx[best_matches_idx]
#     # global_l_l_index = concensus_matches_l_l_idx[best_matches_idx]
#     # return T, global_cur_index, global_prev_index, global_l_l_index, best_matches_idx
#     return best_matches_idx


def create_db(path_to_sequence=r"VAN_ex/code/VAN_ex/dataset/sequences/00", num_frames=200):
    db = TrackingDB()
    l_prev_img, r_prev_img = ex2.read_images(0)
    kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = extract_kps_descs_matches(l_prev_img, r_prev_img)

    in_prev = np.array([False] * len(matches_prev))
    in_prev_idx = extract_inliers_outliers(kp_l_prev, kp_r_prev, matches_prev)[0]
    in_prev[in_prev_idx] = True

    feature_prev, links_prev = db.create_links(desc_l_prev, kp_l_prev, kp_r_prev, matches_prev, in_prev)
    db.add_frame(links=links_prev, left_features=feature_prev, matches_to_previous_left=None, inliers=None)
    db.frameID_to_inliers_percent[0] = 100 * (len(in_prev_idx) / len(matches_prev))
    # print(f"Frame 0: {DB.frameID_to_inliers_percent[0]}% inliers")

    for i in tqdm(range(1, num_frames)):
        l_cur_img, r_cur_img = ex2.read_images(i)
        # kp_l_cur, kp_r_cur, desc_l_cur, desc_r_cur, matches_cur = extract_kps_descs_matches(l_prev_img, r_prev_img)
        kp_l_cur, kp_r_cur, desc_l_cur, desc_r_cur, matches_cur = extract_kps_descs_matches(l_cur_img, r_cur_img)

        # extract the inliers and outliers and triangulate the points
        in_cur_idx = extract_inliers_outliers(kp_l_cur, kp_r_cur, matches_cur)[0]

        db.frameID_to_inliers_percent[i] = 100 * (len(in_cur_idx) / len(matches_cur))
        if db.frameID_to_inliers_percent[i] < 35:
            print(i)

        # print(f"Frame {i}: {DB.frameID_to_inliers_percent[i]}% inliers, num of matches: {len(matches_cur)}")
        in_cur = np.array([False] * len(matches_cur))
        in_cur[in_cur_idx] = True

        # create the links for the curr frame:
        feature_cur, links_cur = db.create_links(desc_l_cur, kp_l_cur, kp_r_cur, matches_cur, in_cur)

        prev_features = db.features(i - 1)

        # extract matches of first left frame and the second left frame
        # matches_l_l = MATCHER.match(prev_features, feature_cur)
        # matches_l_l_backward = MATCHER.match(feature_cur, prev_features)
        matches_l_l = flann.knnMatch(prev_features, feature_cur, k=2)
        matches_l_l_backward = flann.knnMatch(feature_cur, prev_features, k=2)
        if type(matches_l_l[0]) is tuple:
            matches_l_l = np.array(matches_l_l)
            matches_l_l = (matches_l_l[:, 0]).reshape(-1)
        else:
            matches_l_l = np.array(matches_l_l)
        if type(matches_l_l_backward[0]) is tuple:
            matches_l_l_backward = np.array(matches_l_l_backward)
            matches_l_l_backward = (matches_l_l_backward[:, 0]).reshape(-1)
        else:
            matches_l_l_backward = np.array(matches_l_l_backward)

        good_idx = []
        for j, match in enumerate(matches_l_l):
            feature_in_prev = match.queryIdx
            matched_feature_idx = match.trainIdx
            if matches_l_l_backward[matched_feature_idx].trainIdx != feature_in_prev:
                continue
            else:
                good_idx.append(j)

        good_idx = np.array(good_idx)
        filtered_matches_l_l = matches_l_l[good_idx]


        prev_links = db.all_frame_links(i - 1)
        best_matches_idx = ransac_pnp_for_tracking_db(filtered_matches_l_l, prev_links, links_cur,
                                                      db.frameID_to_inliers_percent[i])
        best_matches_idx = good_idx[best_matches_idx]

        in_prev_cur = np.array([False] * len(matches_l_l))
        in_prev_cur[best_matches_idx] = True

        db.add_frame(links_cur, feature_cur, matches_l_l, in_prev_cur)

    return db


def q_4_2(tracking_db: TrackingDB):
    def track_length(tracking_db: TrackingDB, trackId) -> int:
        return len(tracking_db.frames(trackId))

    def total_number_of_tracks(tracking_db: TrackingDB) -> int:
        return len(
            [trackId for trackId in tracking_db.trackId_to_frames if track_length(tracking_db, trackId) > 1])

    def number_of_frames(tracking_db: TrackingDB) -> int:
        return len(tracking_db.frameId_to_trackIds_list)

    def mean_track_length(tracking_db: TrackingDB) -> float:
        lengths = [track_length(tracking_db, trackId) for trackId in tracking_db.trackId_to_frames if
                   track_length(tracking_db, trackId) > 1]
        return np.mean(lengths) if lengths else 0

    def max_track_length(tracking_db: TrackingDB) -> int:
        lengths = [track_length(tracking_db, trackId) for trackId in tracking_db.trackId_to_frames if
                   track_length(tracking_db, trackId) > 1]
        return max(lengths) if lengths else 0

    def min_track_length(tracking_db: TrackingDB) -> int:
        lengths = [track_length(tracking_db, trackId) for trackId in tracking_db.trackId_to_frames if
                   track_length(tracking_db, trackId) > 1]
        return min(lengths) if lengths else 0

    def mean_number_of_frame_links(tracking_db: TrackingDB) -> float:
        if not tracking_db.frameId_to_trackIds_list:
            return 0
        total_links = sum(len(trackIds) for trackIds in tracking_db.frameId_to_trackIds_list.values())
        return total_links / len(tracking_db.frameId_to_trackIds_list)

    total_tracks = total_number_of_tracks(tracking_db)
    num_frames = number_of_frames(tracking_db)
    mean_length = mean_track_length(tracking_db)
    max_length = max_track_length(tracking_db)
    min_length = min_track_length(tracking_db)
    mean_frame_links = mean_number_of_frame_links(tracking_db)

    print(f"Total number of tracks: {total_tracks}")
    print(f"Total number of frames: {num_frames}")
    print(f"Mean track length: {mean_length}, Max track length: {max_length}, Min track length: {min_length}")
    print(f"Mean number of frame links: {mean_frame_links}")


def q_4_3(tracking_db: TrackingDB):
    def get_feature_location(tracking_db: TrackingDB, frameId: int, trackId: int) -> Tuple[float, float]:
        link = tracking_db.linkId_to_link[(frameId, trackId)]
        return link.x_left, link.y

    def find_random_track_of_length(tracking_db: TrackingDB, length: int) -> Optional[int]:
        eligible_tracks = [trackId for trackId, frames in tracking_db.trackId_to_frames.items() if
                           len(frames) >= length]
        if not eligible_tracks:
            return None
        return random.choice(eligible_tracks)

    def visualize_track(tracking_db: TrackingDB, trackId: int):
        frames = tracking_db.frames(trackId)
        print(f"Track {trackId} has {len(frames)} frames")
        plt.figure()
        for i in range(0, 6, 1):
            # print(f"Frame {frames[i]}")
            frameId = frames[i]
            img, _ = ex2.read_images(frameId)
            x_left, y = get_feature_location(tracking_db, frameId, trackId)
            x_min = int(max(x_left - 10, 0))
            x_max = int(min(x_left + 10, img.shape[1]))
            y_min = int(max(y - 10, 0))
            y_max = int(min(y + 10, img.shape[0]))
            cutout = img[y_min:y_max, x_min:x_max]

            plt.subplot(6, 2, 2 * i + 1)
            plt.imshow(img, cmap='gray')
            plt.scatter(x_left, y, color='red')  # Center of the cutout

            plt.subplot(6, 2, 2 * i + 2)
            plt.imshow(cutout, cmap='gray')
            plt.scatter([10], [10], color='red', marker='x', linewidths=1)  # Center of the cutout
            if i == 0:
                plt.title(f"Frame {frameId}, Track {trackId}")
        # plt.show()

    minimal_length = 6
    trackId = find_random_track_of_length(tracking_db, minimal_length)
    if trackId is None:
        print(f"No track of length {minimal_length} found")
    else:
        print(f"Track of length {minimal_length} found: {trackId}")
        visualize_track(tracking_db, trackId)


def q_4_4(tracking_db: TrackingDB):
    def compute_outgoing_tracks(tracking_db: TrackingDB) -> Dict[int, int]:
        outgoing_tracks = {}
        for frameId in sorted(tracking_db.frameId_to_trackIds_list.keys()):
            next_frameId = frameId + 1
            if next_frameId in tracking_db.frameId_to_trackIds_list:
                current_tracks = set(tracking_db.frameId_to_trackIds_list[frameId])
                next_tracks = set(tracking_db.frameId_to_trackIds_list[next_frameId])
                outgoing_tracks[frameId] = len(current_tracks.intersection(next_tracks))
            else:
                outgoing_tracks[frameId] = 0
        return outgoing_tracks

    def plot_connectivity_graph(outgoing_tracks: Dict[int, int]):
        frames = sorted(outgoing_tracks.keys())
        counts = [outgoing_tracks[frame] for frame in frames]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, counts)
        plt.xlabel('Frame ID')
        plt.ylabel('Number of Outgoing Tracks')
        plt.title('Connectivity Graph: Outgoing Tracks per Frame')
        plt.grid(True)
        # plt.show()

    # Compute outgoing tracks
    outgoing_tracks = compute_outgoing_tracks(tracking_db)

    # Plot the connectivity graph
    plot_connectivity_graph(outgoing_tracks)


def q_4_5(tracking_db: TrackingDB):
    def plot_inliers_percentage_graph(inliers_percentage_dict: Dict[int, float]):
        frames = sorted(inliers_percentage_dict.keys())
        percentages = [inliers_percentage_dict[frame] for frame in frames]

        plt.figure(figsize=(20, 10))
        plt.plot(frames, percentages)
        plt.xlabel('Frame ID')
        plt.ylabel('Percentage of Inliers')
        plt.title('Percentage of Inliers per Frame')
        plt.grid(True)
        # plt.show()

    # inliers_percentage = {}
    # for frame_idx in range(LEN_DATA_SET):
    #     img_l, img_r = ex2.read_images(frame_idx)
    #     kp0, kp1, desc0, desc1, matches = ex3.extract_kps_descs_matches(img_l, img_r)
    #     inliers, outliers = ex3.extract_inliers_outliers(kp0, kp1, matches)
    #     inliers_percentage[frame_idx] = (len(inliers) / (len(inliers) + len(outliers))) * 100
    # Compute inliers percentage
    # inliers_percentage = compute_inliers_percentage(tracking_db)

    # Plot the inliers percentage graph
    plot_inliers_percentage_graph(tracking_db.frameID_to_inliers_percent)


def q_4_6(tracking_db: TrackingDB):
    def calculate_track_lengths(tracking_db: TrackingDB) -> List[int]:
        track_lengths = [len(frames) for trackId, frames in tracking_db.trackId_to_frames.items()]
        return track_lengths

    def plot_track_length_histogram(track_lengths: List[int]):
        plt.figure(figsize=(10, 6))
        plt.hist(track_lengths, bins=range(1, max(track_lengths) + 2), edgecolor='black')
        plt.xlabel('Track Length')
        plt.ylabel('Frequency')
        plt.title('Track Length Histogram')
        plt.grid(True)

        plt.yscale('log')
        plt.xlim((0, 150))
        # plt.show()

    # Calculate track lengths
    track_lengths = calculate_track_lengths(tracking_db)

    # Plot the track length histogram
    plot_track_length_histogram(track_lengths)


def q_4_7(tracking_db: TrackingDB):
    def find_random_track_of_length(tracking_db: TrackingDB, length: int) -> Optional[int]:
        eligible_tracks = [trackId for trackId, frames in tracking_db.trackId_to_frames.items() if
                           len(frames) >= length]
        if not eligible_tracks:
            return None
        return random.choice(eligible_tracks)

    def read_kth_camera(k):
        filename = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt'
        with open(filename, 'r') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == k:
                    camera = line.strip()
                    break
        numbers = list(map(float, camera.split()))
        matrix = np.array(numbers).reshape(3, 4)
        return matrix

    def plot_reprojection_errors(reprojection_errors: Dict[int, Tuple[float, float]]):
        frames = sorted(reprojection_errors.keys())
        left_errors = [reprojection_errors[frame][0] for frame in frames]
        right_errors = [reprojection_errors[frame][1] for frame in frames]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, left_errors, label='Left Camera')
        plt.plot(frames, right_errors, label='Right Camera')
        plt.xlabel('distance from reference frame')
        plt.ylabel('projection Error')
        plt.title('projection Error vs track length')
        plt.legend()
        plt.grid(True)
        # plt.show()

    trackId = find_random_track_of_length(tracking_db, 10)
    track_last_frame = tracking_db.last_frame_of_track(trackId)
    frames = tracking_db.frames(trackId)
    left_camera_mat = read_kth_camera(track_last_frame)
    link = tracking_db.link(track_last_frame, trackId)

    p = K @ left_camera_mat
    q = K @ M2 @ np.vstack((left_camera_mat, np.array([0, 0, 0, 1])))

    world_point = ex2.linear_least_squares_triangulation(p, q, (link.x_left, link.y), (link.x_right, link.y))
    world_point_4d = np.append(world_point, 1).reshape(4, 1)

    projections = {}
    reprojection_erros = {}
    for frameId in frames:
        left_camera_mat = read_kth_camera(frameId)
        p = K @ left_camera_mat
        q = K @ M2 @ np.vstack((left_camera_mat, np.array([0, 0, 0, 1])))
        projection_left, projection_right = ((p @ world_point_4d).T, (q @ world_point_4d).T)

        projection_left = projection_left / projection_left[:, 2][:, np.newaxis]
        projection_right = projection_right / projection_right[:, 2][:, np.newaxis]
        projections[frameId] = (projection_left, projection_right)
        link = tracking_db.link(frameId, trackId)
        points_vec_left = np.array([link.x_left, link.y, 1])
        points_vec_right = np.array([link.x_right, link.y, 1])
        reprojection_erros[int(track_last_frame - frameId)] = (np.linalg.norm(points_vec_left - projection_left[0:2]),
                                                               np.linalg.norm(points_vec_right - projection_right[0:2]))

    plot_reprojection_errors(reprojection_erros)


if __name__ == '__main__':
    # print(M1)
    # print(M2)
    # print(K)
    # all_frames_serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB3000"
    # serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB_all_after_changing the percent"
    # path = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00"
    # serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB_all_after_changing the percent"
    # serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/check50"
    serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/NEWEST_FLANN"
    path = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00"
    db = create_db(num_frames=LEN_DATA_SET)
    db.serialize(serialized_path)

    db = TrackingDB()
    db.load(serialized_path)
    # print(len(db.frames(7)))
    # print(db.all_tracks())
    # for track in db.all_tracks():
    #     if len(db.frames(track)) < 2:
    #         print(len(db.frames(track)))

    # q_4_1(db)
    q_4_2(db)
    q_4_3(db)
    q_4_4(db)
    q_4_5(db)
    q_4_6(db)
    q_4_7(db)
    plt.show()
