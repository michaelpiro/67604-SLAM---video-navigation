import pickle

import gtsam

from VAN_ex.code import arguments
from VAN_ex.code.tracking_database import TrackingDB
from ex6 import load
import numpy as np
import cv2

MATCHER = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)

MAHALANOBIS_THRESHOLD = 20.0
INLIERS_THRESHOLD = 0.4
relative_covariance_dict = dict()


def get_relative_consecutive_covariance(c1, c2, marginals):
    c2_number = int(gtsam.DefaultKeyFormatter(c2)[1:])
    if c2_number in relative_covariance_dict:
        return relative_covariance_dict[c2_number]
    # return relative_covariance_dict[c2_number]
    keys = gtsam.KeyVector()
    keys.append(c1)
    keys.append(c2)
    marginal_information = marginals.jointMarginalInformation(keys)
    inf_c2_giving_c1 = marginal_information.at(c2, c2)
    cov_c2_giving_c1 = np.linalg.inv(inf_c2_giving_c1)
    return cov_c2_giving_c1


def get_relative_covariance(index_list, marginals):
    cov_sum = None
    for i in range(len(index_list) - 1):
        c1 = gtsam.symbol('c', index_list[i])
        c2 = gtsam.symbol('c', index_list[i + 1])
        c_n_giving_c_i = get_relative_consecutive_covariance(c1, c2, marginals)
        if cov_sum is None:
            cov_sum = c_n_giving_c_i
        else:
            cov_sum = cov_sum + c_n_giving_c_i
    return cov_sum

def get_relative_pose(pose):
    pass


def calculate_mahalanobis_distance(c_n, c_i, result, relative_information):
    pose_c_n = result.atPose3(c_n)
    pose_c_i = result.atPose3(c_i)
    relative_pose = pose_c_n.between(pose_c_i)
    delta = np.hstack([relative_pose.rotation().ypr(), relative_pose.translation()])
    mahalanobis_distance = delta @ relative_information @ delta
    # mahalanobis_distance = relative_pose @ relative_information @ relative_pose
    print(f"mahalanobis_distance:{mahalanobis_distance}, determinant: {np.linalg.det(relative_information)} delta norm: {np.linalg.norm(delta)}")
    return mahalanobis_distance

def get_symbol(index):
    return gtsam.symbol('c', index)

def check_candidate(c_n_idx, c_i_idx, marginals, result,index_list,sum_covariance):
    cur_index_list = index_list[c_i_idx:c_n_idx+1]
    c_n_giving_c_i = get_relative_covariance(cur_index_list, marginals)
    # c_n_giving_c_i = sum_covariance + get_relative_consecutive_covariance(gtsam.symbol('c', index_list[c_i_idx]),
    #                                                                       gtsam.symbol('c', index_list[c_n_idx]), marginals)
    relative_information = np.linalg.inv(c_n_giving_c_i)
    symbol_cn = get_symbol(index_list[c_n_idx])
    symbol_ci = get_symbol(index_list[c_i_idx])
    mahalanobis_distance = calculate_mahalanobis_distance(symbol_cn,symbol_ci, result, relative_information)

    return c_n_giving_c_i, mahalanobis_distance


def get_good_candidates(c_n_index, marginals, result,index_list):
    candidates = []
    relative_covatiance_sum = get_relative_consecutive_covariance(gtsam.symbol('c', index_list[0]),
                                                                  gtsam.symbol('c', index_list[c_n_index]), marginals)

    # camera_symbol = gtsam.symbol('c', camera_referance_index)
    for c_i_index in range(c_n_index):
        # camera_i_symbol = gtsam.symbol('c', i)
        relayive_covariance_sum, mahalanobis_distance = check_candidate(c_n_index, c_i_index, marginals, result,index_list, relative_covatiance_sum)
        if mahalanobis_distance < MAHALANOBIS_THRESHOLD:
            candidates.append(index_list[c_i_index])

    return candidates


def get_path(c_n, c_i,result):
    return [index for index in range(c_i, c_n + 1)]


def load(base_filename):
    """ load TrackingDB to base_filename+'.pkl' file. """
    filename = base_filename + '.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print('Bundles loaded from', filename)
    return data

def get_index_list(result):
    index_list = []
    for key in result.keys():
        cam_number = int(gtsam.DefaultKeyFormatter(key)[1:])
        index_list.append(cam_number)
    index_list.sort()
    return index_list



def q_7_1(data):
    pose_graph = data[0]
    result = data[1]
    marginals = gtsam.Marginals(pose_graph, result)
    index_list = get_index_list(result)
    for i in range(len(index_list) - 1):
        c1 = gtsam.symbol('c', index_list[i])
        c2 = gtsam.symbol('c', index_list[i + 1])
        c_n_giving_c_i = get_relative_consecutive_covariance(c1, c2, marginals)
        relative_covariance_dict[index_list[i + 1]] = c_n_giving_c_i

    relative_covariance_dict[index_list[0]] = marginals.marginalCovariance(gtsam.symbol('c', index_list[0]))
    print(f"relative_covariance_lst: {relative_covariance_dict}")
    for c_n_index in range(len(index_list)):
        good_candidates = get_good_candidates(c_n_index, marginals, result, index_list)
        print(f"good_candidates for camera {index_list[c_n_index]}: {good_candidates}")




def concensus_matching(keyframe1, keyframe2, db: TrackingDB):
    keyframe1_features = db.features(keyframe1)
    keyframe2_features = db.features(keyframe2)

    # get matches
    matches_l_l = MATCHER.match(keyframe1_features, keyframe2_features)


    good_idx = np.array(good_idx)
    filtered_matches_l_l = matches_l_l[good_idx]

    # img_points = np.array([kp_l_cur[match.trainIdx].pt for match in matches_l_l])
    # points_3d = triangulate_links(links_prev, P, Q)
    # camera_matrix = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]])
    # dist_
    # rvec,tvec,suc, inliers = cv2.solvePnPRansac(points_3d,img_points,camera_matrix,distCoeffs=None)

    prev_links = db.all_frame_links(i - 1)
    best_matches_idx = ransac_pnp_for_tracking_db(filtered_matches_l_l, prev_links, links_cur,
                                                  db.frameID_to_inliers_percent[i])



    matches =
if __name__ == '__main__':
    # load data
    path = arguments.DATA_HEAD + '/docs/pose_graph_result'
    data_list = load(path)
    q_7_1(data_list)
    print("pipi")
