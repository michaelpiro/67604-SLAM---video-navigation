import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Sequence, Optional
from timeit import default_timer as timer
from tracking_database import TrackingDB, Link, MatchLocation
from tqdm import tqdm
from ex4_2_to_6 import q_4_2, q_4_3, q_4_4, q_4_5, q_4_6, q_4_7, find_longest_track_frames

import ex3
import ex2

NO_ID = -1

FEATURE = cv2.AKAZE_create()
FEATURE.setNOctaves(2)
FEATURE.setThreshold(0.001)
MATCHER = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
# DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'
DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))
# LEN_DATA_SET = 1000
P = ex3.P
Q = ex3.Q
K = ex3.K
M1 = ex3.M1
M2 = ex3.M2


def extract_kps_descs_matches(img_0, img1):
    kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    matches = MATCHER.match(desc0, desc1)
    return kp0, kp1, desc0, desc1, matches


# todo: change this function to least squares triangulation
def triangulate_last_frame(tracking_db: TrackingDB, p, q):
    """
    Triangulate the matched points using OpenCV
    :param inliers:
    :return:
    """
    links = tracking_db.all_last_frame_links()
    x = np.zeros((len(links), 3))
    for i in range(len(links)):
        x_left, x_right, y = links[i].x_left, links[i].x_right, links[i].y
        p_left, p_right = (x_left, y), (x_right, y)
        x[i] = ex2.linear_least_squares_triangulation(p, q, p_left, p_right)
    return x


def find_concensus_points_and_idx(matches_lr_0, in_lr_0, matches_l0_l1, matches_lr1, in_lr1):
    # Create dictionaries for quick lookup
    # a dict that it's keys are the keypoint of the match in the left image,
    # and the value is the index of the match in the matches array
    dict_lr_0 = {match.queryIdx: i for i, match in enumerate(matches_lr_0) if in_lr_0[i]}
    dict_lr_1 = {match.queryIdx: i for i, match in enumerate(matches_lr1) if in_lr1[i]}

    con = []
    matches = []
    matches_l0_l1_good_idx = []
    for i, match_l_l in enumerate(matches_l0_l1):
        kp_ll_0 = match_l_l.queryIdx
        kp_ll_1 = match_l_l.trainIdx

        # Check if the match exists in both dictionaries
        if kp_ll_0 in dict_lr_0 and kp_ll_1 in dict_lr_1:
            i0 = dict_lr_0[kp_ll_0]
            i1 = dict_lr_1[kp_ll_1]
            con.append((i0, i1))
            matches.append((matches_lr_0[i0], matches_lr1[i1]))
            matches_l0_l1_good_idx.append(i)
        else:
            print("bad match")

    return np.array(con), np.array(matches), matches_l0_l1_good_idx


def transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
                             ordered_cur_left_pix_values, ordered_cur_right_pix_values, x_condition=False):
    T_4x4 = np.vstack((T, np.array([0, 0, 0, 1])))
    points_4d = np.hstack((traingulated_pts, np.ones((traingulated_pts.shape[0], 1)))).T
    l1_T = K @ T
    l1_4d_points = (T_4x4 @ points_4d)
    to_the_right = (K @ M2)

    transform_to_l0_points = (P @ points_4d).T
    transform_to_l0_points = transform_to_l0_points / transform_to_l0_points[:, 2][:, np.newaxis]
    real_y = prev_left_pix_values[:, 1]
    agree_l0 = np.abs(transform_to_l0_points[:, 1] - real_y) < 2

    transform_to_r0_points = (to_the_right @ points_4d).T
    transform_to_r0_points = transform_to_r0_points / transform_to_r0_points[:, 2][:, np.newaxis]
    real_y = prev_right_pix_values[:, 1]
    agree_r0 = np.abs(transform_to_r0_points[:, 1] - real_y) < 2
    if x_condition:
        real_x_l = prev_left_pix_values[:, 0]
        real_x_r = prev_right_pix_values[:, 0]
        cond_x = real_x_l > real_x_r
    else:
        cond_x = np.ones_like(agree_r0)
    agree_0 = np.logical_and(agree_r0, cond_x, agree_l0)

    transformed_to_l1_points = (K @ l1_4d_points[:3, :]).T
    transformed_to_l1_points = transformed_to_l1_points / transformed_to_l1_points[:, 2][:, np.newaxis]
    real_y = ordered_cur_left_pix_values[:, 1]
    agree_l1 = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2

    transformed_to_r1_points = (to_the_right @ l1_4d_points).T
    transformed_to_r1_points = transformed_to_r1_points / transformed_to_r1_points[:, 2][:, np.newaxis]
    real_y = ordered_cur_right_pix_values[:, 1]
    agree_r1 = np.abs(transformed_to_r1_points[:, 1] - real_y) < 2
    if x_condition:
        real_x_l = ordered_cur_left_pix_values[:, 0]
        real_x_r = ordered_cur_right_pix_values[:, 0]
        cond_x = real_x_l > real_x_r
    else:
        cond_x = np.ones_like(agree_r1)
    agree_1 = np.logical_and(agree_r1, cond_x, agree_l1)
    return np.logical_and(agree_0, agree_1)


def ransac_pnp_for_tracking_db(traingulated_pts, matches, links_prev, links_cur):
    """ Perform RANSAC to find the best transformation"""
    best_inliers = 0
    best_T = None
    best_matches_idx = None
    diff_coeff = np.zeros((5, 1))
    prev_left_pix_values = []
    prev_right_pix_values = []
    ordered_cur_left_pix_values = []
    ordered_cur_right_pix_values = []
    for link in links_prev:
        prev_left_pix_values.append((link.x_left, link.y))
        prev_right_pix_values.append((link.x_right, link.y))
    for match in matches:
        link_index = match.trainIdx
        ordered_cur_left_pix_values.append((links_cur[link_index].x_left, links_cur[link_index].y))
        ordered_cur_right_pix_values.append((links_cur[link_index].x_right, links_cur[link_index].y))
    prev_left_pix_values = np.array(prev_left_pix_values)
    prev_right_pix_values = np.array(prev_right_pix_values)
    ordered_cur_left_pix_values = np.array(ordered_cur_left_pix_values)
    ordered_cur_right_pix_values = np.array(ordered_cur_right_pix_values)

    for i in range(ex3.RANSAC_ITERATIONS):

        # Randomly select 4 points in the world coordinate system
        random_idx = np.random.choice(len(traingulated_pts), 4, replace=False)
        random_world_points = traingulated_pts[random_idx]
        random_cur_l_pixels = ordered_cur_left_pix_values[random_idx]

        # solve PnP problem to get the transformation
        success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_cur_l_pixels, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
        if success:
            T = ex3.rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue

        points_agreed = transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
                                                 ordered_cur_left_pix_values, ordered_cur_right_pix_values,
                                                 x_condition=False)

        inliers_idx = np.where(points_agreed == True)
        if np.sum(points_agreed) > best_inliers:
            best_inliers = np.sum(points_agreed)
            best_T = T
            best_matches_idx = inliers_idx

    return best_T, best_matches_idx, ordered_cur_left_pix_values[best_matches_idx[0]]


def find_best_transformation_ex4(traingulated_pts, matches, links_prev, links_cur):
    """ Find the best transformation using RANSAC"""

    T, inliers_idx, agreed_img_points = ransac_pnp_for_tracking_db(traingulated_pts, matches, links_prev, links_cur)

    best_matches = matches[inliers_idx[0]]

    diff_coeff = np.zeros((5, 1))
    pt_3d = traingulated_pts[inliers_idx[0]]
    if len(pt_3d) < 4:
        raise ValueError("Not enough points to estimate the transformation")
    success, rotation_vector, translation_vector = cv2.solvePnP(pt_3d, agreed_img_points, K,
                                                                distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
    if success:
        return ex3.rodriguez_to_mat(rotation_vector, translation_vector), inliers_idx[0]
    return None


def create_DB(path_to_sequence=r"VAN_ex/code/VAN_ex/dataset/sequences/00", num_of_frames=50):
    l_prev_img, r_prev_img = ex2.read_images(0)
    kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = extract_kps_descs_matches(l_prev_img, r_prev_img)
    DB = TrackingDB()

    in_prev = np.array([False] * len(matches_prev))
    in_prev_idx = ex3.extract_inliers_outliers(kp_l_prev, kp_r_prev, matches_prev)[0]
    in_prev[in_prev_idx] = True
    feature_prev, links_prev = DB.create_links(desc_l_prev, kp_l_prev, kp_r_prev, matches_prev, in_prev)
    DB.add_frame(links=links_prev, left_features=feature_prev, matches_to_previous_left=None, inliers=None)
    DB.frameID_to_inliers_percent[0] = 100*(len(in_prev_idx)/len(matches_prev))

    for i in tqdm(range(1, num_of_frames)):
        # load the next frames and extract the keypoints and descriptors
        img_l_cur, img_r_cur = ex2.read_images(i)
        kp_l_cur, kp_r_cur, desc_l_cur, desc_r_cur, matches_cur = extract_kps_descs_matches(img_l_cur, img_r_cur)

        # extract the inliers and outliers and triangulate the points
        in_cur_idx = ex3.extract_inliers_outliers(kp_l_cur, kp_r_cur, matches_cur)[0]
        DB.frameID_to_inliers_percent[i] = 100 * (len(in_cur_idx) / len(matches_cur))
        in_cur = np.array([False] * len(matches_cur))
        in_cur[in_cur_idx] = True

        # create the links for the curr frame:
        feature_cur, links_cur = DB.create_links(desc_l_cur, kp_l_cur, kp_r_cur, matches_cur, in_cur)

        # extract matches of first left frame and the second left frame
        matches_l_l = MATCHER.match(feature_prev, feature_cur)
        # matches_l_l = np.array(MATCHER.knnMatch(desc_l_prev, desc_l_cur, k=2))
        if type(matches_l_l[0]) is tuple:
            matches_l_l = np.array(matches_l_l)
            matches_l_l = matches_l_l[:, 0]
        else:
            matches_l_l = np.array(matches_l_l)

        traingulated_pts = triangulate_last_frame(DB, P, Q)

        relative_transformation, idx = find_best_transformation_ex4(traingulated_pts, matches_l_l, links_prev,
                                                                    links_cur)

        # todo it needs to be so that the inliers would be a binary array for the matches of the l to l (matches l_l who are best
        # final_matches = (matches_pairs[:, 0])[idx]
        # final_matches = [match.queryIdx for match in final_matches]
        in_prev_cur = np.array([False] * len(matches_l_l))
        in_prev_cur[idx] = True
        DB.add_frame(links_cur, feature_cur, matches_l_l, in_prev_cur)

        # update the keypoints, descriptors and matches
        kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = kp_l_cur, kp_r_cur, desc_l_cur, \
            desc_r_prev, matches_cur
        feature_prev = feature_cur
        # needs to be only matches from l to prev l who are good
        links_prev = links_cur
    return DB


if __name__ == '__main__':
    all_frames_serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB3000"
    serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB_all_after_changing the percent"
    path = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00"
    db = create_DB(path, LEN_DATA_SET)
    db.serialize(serialized_db_path)
    # db = TrackingDB()
    # db.load(all_frames_serialized_db_path)
    q_4_2(db)
    q_4_3(db)
    q_4_4(db)
    q_4_5(db)
    q_4_6(db)
    q_4_7(db)
    plt.show()
