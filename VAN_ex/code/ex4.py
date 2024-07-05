import os

import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Sequence, Optional
from timeit import default_timer as timer
from tracking_database import TrackingDB, Link, MatchLocation
from tqdm import tqdm

import ex3
import ex2

NO_ID = -1

FEATURE = cv2.AKAZE_create()
FEATURE.setNOctaves(2)
FEATURE.setThreshold(0.0001)
MATCHER = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
# DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'
DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
P = ex3.P
Q = ex3.Q

def extract_kps_descs_matches(img_0, img1):
    kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    matches = MATCHER.match(desc0, desc1)
    return kp0, kp1, desc0, desc1, matches

def find_concensus_points_and_idx(matches_lr_0, in_lr_0, matches_l0_l1,matches_lr1, in_lr1):
    # Create dictionaries for quick lookup
    dict_lr_0 = {match.queryIdx: i for i, match in enumerate(matches_lr_0) if in_lr_0[i]}
    dict_lr_1 = {match.queryIdx: i for i, match in enumerate(matches_lr1)if in_lr1[i]}

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

    return np.array(con), np.array(matches), matches_l0_l1_good_idx

def create_DB(path_to_sequence=r"VAN_ex/code/VAN_ex/dataset/sequences/00", num_of_frames=50):
    l_prev_img, r_prev_img = ex2.read_images(0)
    kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = extract_kps_descs_matches(l_prev_img, r_prev_img)
    DB = TrackingDB()

    in_prev = np.zeros(len(matches_prev))
    in_prev[ex3.extract_inliers_outliers(kp_l_prev, kp_r_prev, matches_prev)[0]] = 1
    feature_prev, links_prev = DB.create_links(desc_l_prev, kp_l_prev, kp_r_prev,matches_prev, in_prev)
    DB.add_frame(links=links_prev, left_features=feature_prev,matches_to_previous_left=None, inliers=None)

    for i in tqdm(range(1, num_of_frames)):
        # load the next frames and extract the keypoints and descriptors
        img_l_cur, img_r_cur = ex2.read_images(i)
        kp_l_cur, kp_r_cur, desc_l_cur, desc_r_cur, matches_cur = extract_kps_descs_matches(img_l_cur, img_r_cur)
        matches_cur = np.array(matches_cur)

        # extract the inliers and outliers and triangulate the points
        in_cur_idx = ex3.extract_inliers_outliers(kp_l_cur, kp_r_cur, matches_cur)[0]
        in_cur = np.zeros(matches_cur.shape)
        in_cur[in_cur_idx] = 1
        # in_cur_instances = matches_cur[in_cur_idx]
        # in_cur_instances = np.array(in_cur_instances)

        #create the links for the curr frame:
        feature_cur, links_cur = DB.create_links(desc_l_cur, kp_l_cur, kp_r_cur, matches_cur, in_cur)

        # extract matches of first left frame and the second left frame
        matches_l_l = np.array(MATCHER.knnMatch(feature_prev, feature_cur, k=2))
        # matches_l_l = np.array(MATCHER.knnMatch(desc_l_prev, desc_l_cur, k=2))
        matches_l_l = matches_l_l[:,0]

        # # find the concensus matches
        # good_matches_idx, matches_pairs, matches_l_l_good_idx = find_concensus_points_and_idx(matches_prev, in_prev , matches_l_l,matches_cur, in_cur)
        # # triangulate the points only the good matches points from in_prev
        # prev_best_inliers = good_matches_idx[:, 0]
        # # prev_best_inliers = matches_prev[prev_best_inliers_idx]
        # traingulated_pts = ex2.cv_triangulate_matched_points(np.array(matches_prev)[prev_best_inliers], kp_l_prev, kp_r_prev, P, Q)


        good_matches_idx, matches_pairs, matches_l_l_good_idx = find_concensus_points_and_idx(matches_prev, in_prev,
                                                                                              matches_l_l, matches_cur,
                                                                                              in_cur)
        # triangulate the points only the good matches points from in_prev
        prev_best_inliers = matches_pairs[:, 0]
        print(f"prev_best_inliers: {len(prev_best_inliers)}")
        # prev_best_inliers = matches_prev[prev_best_inliers_idx]
        traingulated_pts = ex2.cv_triangulate_matched_points(prev_best_inliers, kp_l_prev,
                                                             kp_r_prev, P, Q)
        # find the best transformation
        relative_transformation, idx = ex3.find_best_transformation(traingulated_pts, matches_pairs, kp_l_prev,
                                                                    kp_r_prev, kp_l_cur, kp_r_cur)

        #todo it needs to be so that the inliers would be a binary array for the matches of the l to l (matches l_l who are best
        # final_matches = (matches_pairs[:, 0])[idx]
        # final_matches = [match.queryIdx for match in final_matches]
        in_prev_cur = np.zeros(len(matches_l_l))
        # in_prev_cur[matches_l_l_good_idx] = 1
        in_prev_cur[idx] = 1

        DB.add_frame(links_cur,feature_cur,matches_l_l, in_prev_cur)

        # update the keypoints, descriptors and matches
        kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = kp_l_cur, kp_r_cur, desc_l_cur, \
                                                                       desc_r_prev, matches_cur
        feature_prev = feature_cur
        #needs to be only matches from l to prev l who are good
        in_prev = in_cur

if __name__ == '__main__':
    path = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00"
    create_DB(path, 120)