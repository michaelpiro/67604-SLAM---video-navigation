import ex2
import cv2
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer



MAC = True
if MAC:
    DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
    # DATA_PATH = 'dataset/sequences/00'
else:
    DATA_PATH = r'...\VAN_ex\dataset\sequences\00\\'

SEC_PROB = 0.99
OUTLIER_PROB = 0.6
MIN_SET_SIZE = 4
# RANSAC_ITERATIONS = int(np.log(1 - SEC_PROB) / np.log(1 - np.power(1 - OUTLIER_PROB, MIN_SET_SIZE))) + 1
RANSAC_ITERATIONS = 50

LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))
# LEN_DATA_SET = 500

GROUND_TRUTH_PATH = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt"

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# MATCHER = cv2.FlannBasedMatcher(index_params, search_params)
MAX_FEATURES = 1000
hessianThreshold = 3000
# fast = cv2.FastFeatureDetector_create()
# brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
FEATURE = cv2.AKAZE_create()
# FEATURE = cv2.FastFeatureDetector_create(threshold = 80,
#                                      nonmaxSuppression = True)
FEATURE.setThreshold(0.001)
MATCHER = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)


def read_cameras():
    if MAC:
        data_path = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
    else:
        data_path = DATA_PATH
    with open(data_path + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


K, M1, M2 = read_cameras()
P, Q = K @ M1, K @ M2


def read_extrinsic_matrices(file_path=GROUND_TRUTH_PATH, n=LEN_DATA_SET):
    """
    Reads n lines from a file and returns a list of extrinsic camera matrices.

    Args:
    - file_path (str): Path to the file containing the extrinsic matrices.
    - n (int): Number of lines (matrices) to read from the file.

    Returns:
    - List[np.ndarray]: A list of n extrinsic camera matrices (each a 3x4 numpy array).
    """
    extrinsic_matrices = []

    with open(file_path, 'r') as file:
        for i in range(n):
            line = file.readline()
            if not line:
                break  # If less than n lines are available, stop reading
            numbers = list(map(float, line.strip().split()))
            if len(numbers) != 12:
                raise ValueError(f"Line {i + 1} does not contain 12 numbers.")
            matrix = np.array(numbers).reshape(3, 4)
            extrinsic_matrices.append(matrix)

    return extrinsic_matrices


def extract_kps_descs_matches(img_0, img1):
    kp0 = FEATURE.detect(img_0, None)
    kp1 = FEATURE.detect(img1, None)
    desc0 = FEATURE.compute(img_0, kp0)[1]
    desc1 = FEATURE.compute(img1, kp1)[1]
    matches = MATCHER.match(desc0, desc1)
    return kp0, kp1, desc0, desc1, matches
    # kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    # kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    # matches = MATCHER.match(desc0, desc1)
    # return kp0, kp1, desc0, desc1, matches


# def extract_inliers_outliers(kp_left, kp_right, matches):
#     inliers = []
#     outliers = []
#     for i in range(len(matches)):
#         ind_left = matches[i].queryIdx # index of kp_left
#         ind_right = matches[i].trainIdx # index of kp_right
#
#         good_map1 = abs(kp_left[ind_left].pt[1] - kp_right[ind_right].pt[1]) < 2
#         good_map2 = kp_left[ind_left].pt[0] > kp_right[ind_right].pt[0]
#         if good_map1 and good_map2:
#             inliers.append(matches[i])
#         else:
#             outliers.append(matches[i])
#     return inliers, outliers

def extract_inliers_outliers(kp_left, kp_right, matches):
    # Convert keypoints to numpy arrays for efficient computation
    kp_left_pts = np.array([kp.pt for kp in kp_left])
    kp_right_pts = np.array([kp.pt for kp in kp_right])

    inliers = []
    outliers = []

    for match in matches:
        ind_left = match.queryIdx
        ind_right = match.trainIdx

        # Use numpy arrays for comparisons
        good_map1 = abs(kp_left_pts[ind_left, 1] - kp_right_pts[ind_right, 1]) < 2
        good_map2 = kp_left_pts[ind_left, 0] > kp_right_pts[ind_right, 0]
        # good_map2 = good_map1
        if good_map1 and good_map2:
            inliers.append(match)
        else:
            outliers.append(match)

    return inliers, outliers



def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def choose_4_points(matches_lr_0, matches_l0_l1, matches_lr1):
    con = []
    for match_l_l in matches_l0_l1:
        kp_ll_0 = match_l_l.queryIdx
        kp_ll_1 = match_l_l.trainIdx
        for i0 in range(len(matches_lr_0)):
            if kp_ll_0 == matches_lr_0[i0].queryIdx:
                for i1 in range(len(matches_lr1)):
                    if kp_ll_1 == matches_lr1[i1].queryIdx:
                        con.append((i0, i1))
    return con


def find_concensus_points_and_idx2(good_matches_lr_0, matches_l0_l1, good_matches_lr1):
    con = []
    matches = []
    # good_idx_l0 = [good_matches_lr_0[i].queryIdx for i in range(len(good_matches_lr_0))]
    # good_idx_l1 = [good_matches_lr1[i].queryIdx for i in range(len(good_matches_lr1))]


    for i in range(len(good_matches_lr_0)):
        match = good_matches_lr_0[i]
        for j in range(len(matches_l0_l1)):
            if match.queryIdx == matches_l0_l1[j].queryIdx:
                for k in range(len(good_matches_lr1)):
                    if matches_l0_l1[j].trainIdx == good_matches_lr1[k].queryIdx:
                        con.append((i, k))
                        matches.append((good_matches_lr_0[i], good_matches_lr1[k]))
        # if match.queryIdx in good_idx_l0 and match.trainIdx in good_idx_l1:
        #     con.append((match.queryIdx, match.trainIdx))
        #     matches.append((good_matches_lr_0[match.queryIdx], good_matches_lr1[match.trainIdx]))
        # ind_in_kp_l_1 = match_l_l.trainIdx
        # is_a_good_match_0 = ind_in_kp_l_0 in good_idx_l0
        # is_a_good_match_1 = ind_in_kp_l_1 in good_idx_l1
        # if is_a_good_match_0 and is_a_good_match_1:
        #     con.append((ind_in_kp_l_0, ind_in_kp_l_1))
        #     matches.append((good_matches_lr_0[ind_in_kp_l_0], good_matches_lr1[ind_in_kp_l_1]))
    return np.array(con), np.array(matches)


def extract_y_values(matches, kp_l, kp_r):
    # Convert keypoints to numpy arrays
    kp1_array = np.array([kp.pt for kp in kp_l])
    kp2_array = np.array([kp.pt for kp in kp_r])

    # Extract indices from matches
    query_indices = np.array([match.queryIdx for match in matches])
    train_indices = np.array([match.trainIdx for match in matches])

    # Extract y-values using the indices
    y_values_1 = kp1_array[query_indices, 1]
    y_values_2 = kp2_array[train_indices, 1]

    return y_values_1, y_values_2


def extract_inliers_outliers_triangulate(P, Q, kp_l_first, kp_r_first, matches_first):
    inliers_0, outliers_0 = ex2.extract_inliers_outliers(kp_l_first, kp_r_first, matches_first)
    triangulated_0 = ex2.triangulate_matched_points(P, Q, inliers_0, kp_l_first, kp_r_first)
    return triangulated_0, inliers_0, outliers_0


def ransac_pnp2(matches_idx, inliers_matches, inliers_traingulated_pts, kp_l_first, kp_r_first, kp_l_second,
                kp_r_second, old_T):
    """ Perform RANSAC to find the best transformation"""
    best_inliers = 0
    best_T = None
    best_matches_idx = None
    # relevant_3d_pts = traingulated_pts[matches_idx[:, 0]]
    for i in range(RANSAC_ITERATIONS):
        random_idx = np.random.choice(len(inliers_matches), 4, replace=False)
        random_matches = inliers_matches[random_idx]
        # img_pts = np.array([kp_l_second[m[1].queryIdx].pt for m in random_matches]).reshape(-1, 2, 1)
        img_pts = np.array([kp_l_second[m[1].queryIdx].pt for m in random_matches])
        # q = np.vstack((old_T, np.array([0, 0, 0, 1])))
        # q = M2 @ q
        random_traingulated_pts = ex2.triangulate_matched_points(P, Q, random_matches[:, 0], kp_l_first,
                                                                 kp_r_first)
        # random_traingulated_pts = random_traingulated_pts[random_idx]
        diff_coeff = np.zeros((5, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(random_traingulated_pts, img_pts, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
        if success:
            T = rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue
        points_agreed = check_transform_agreed(T, old_T, inliers_traingulated_pts, inliers_matches, kp_l_first,
                                               kp_r_first,
                                               kp_l_second,
                                               kp_r_second)
        inliers_idx = np.where(points_agreed == True)
        # inliers = relevant_3d_pts[inliers_idx]
        if np.sum(points_agreed) > best_inliers:
            best_inliers = np.sum(points_agreed)
            best_T = T
            best_matches_idx = inliers_idx
    return best_T, best_matches_idx


def perform_tracking2(first_indx):
    transformations = []
    img_l_first, img_r_first = ex2.read_images(first_indx)
    kp_l_first, kp_r_first, desc_l_first, desc_r_first, matches_first = extract_kps_descs_matches(img_l_first,
                                                                                                  img_r_first)
    left_cam_matrix = M1
    right_cam_matrix = M2

    for i in range(first_indx + 1, first_indx + LEN_DATA_SET):
        triangulated_first, in_first, out_first = extract_inliers_outliers_triangulate(left_cam_matrix,
                                                                                       right_cam_matrix, kp_l_first,
                                                                                       kp_r_first,
                                                                                       matches_first)

        img_l_second, img_r_second = ex2.read_images(i)
        kp_l_second, kp_r_second, desc_l_second, desc_r_second, matches_second = extract_kps_descs_matches(img_l_second,
                                                                                                           img_r_second)

        in_second, out_second = ex2.extract_inliers_outliers(kp_l_second, kp_r_second, matches_second)

        matches_l_l = ex2.MATCHER.match(desc_l_first, desc_l_second)

        # idx, match = find_concensus_points_and_idx(in_first, matches_l_l, in_second)
        idx, match = find_concensus_points_and_idx(in_first, matches_l_l, in_second)
        l_first_inliers_from_concensus = [m[0] for m in match]
        triangulated_from_concensus = ex2.cv_triangulate_matched_points(l_first_inliers_from_concensus, kp_l_first,
                                                                        kp_r_first,
                                                                        # K @ left_cam_matrix,
                                                                        # K @ right_cam_matrix)
                                                                        P, Q)
        old_t = transformations[-1] if transformations else M1
        T, inliers_idx = find_best_transformation(match, idx, triangulated_from_concensus, kp_l_first, kp_r_first,
                                                  kp_l_second,
                                                  kp_r_second, old_t)
        # last_T = transformations[-1] if transformations else M1
        # last_row = np.array([0, 0, 0, 1])
        # last_T = np.vstack((last_T, last_row))
        # T =T @ last_T
        global_trasnform = T @ np.vstack((old_t, np.array([0, 0, 0, 1])))
        transformations.append(global_trasnform)
        left_cam_matrix = M1
        # right_cam_matrix = np.vstack((T, np.array([0, 0, 0, 1])))
        right_cam_matrix = M2
        kp_l_first, kp_r_first = kp_l_second, kp_r_second
        desc_l_first, desc_r_first, matches_first = desc_l_second, desc_r_second, matches_second
    return transformations


def extract_matches_from_images(img_0, img1):
    kp0 = FEATURE.detect(img_0)
    kp1 = FEATURE.detect(img1)
    desc0 = FEATURE.compute(img_0, kp0)[1]
    desc1 = FEATURE.compute(img1, kp1)[1]
    matches = MATCHER.match(desc0, desc1)
    return kp0,kp1,matches
    # kp0, desc0 = FEATURE.detectAndCompute(img_0)
    # kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    # matches = MATCHER.match(desc0, desc1)
    # return kp0, kp1, matches


def calculate_4_camera_locations(T: np.array):
    """
    claculate the 4 diffferent cameras locations in regards to the first camera
    using the formula of -RT(t) for each R t Extrinsic matrix
    :return: the 4 camera locations as a 4 array of 3d points
    """
    left_camera_0 = np.zeros(3)  # the first camera is always at the center
    # first camera, the right, as cordination in the global where left0 is center
    # M2 is our extrinsic Matrix
    RM2 = np.array(M2[:, :3])
    tM2 = np.array(M2[:, 3:])
    right_camera_0 = -(RM2.transpose() @ tM2).reshape((3))
    # left1 camera place, with T our transformation
    Rleft1 = np.array(T[:, :3])
    tleft1 = np.array(T[:, 3:])
    left_camera_1 = -(Rleft1.transpose() @ tleft1).reshape((3))

    # finally, the right camera would neet to use the T transformation and M2 together as we difined in q3.3
    # the extrinsic matrix is [M2T|M2(tLeft1)+tM2] so the camera location would be
    right_camera_1 = -((RM2 @ Rleft1).transpose() @ ((RM2 @ tleft1) + tM2)).reshape((3))
    return left_camera_0, right_camera_0, left_camera_1, right_camera_1


def plot_points_on_img1_img2(idx, not_idx, matches, img1, img2, kp_l0, kp_l1):
    # plot the points that agree with the transformation
    idx = np.array(idx[0])
    not_idx = np.array(not_idx[0])
    kp_l0 = np.array(kp_l0)
    kp_l1 = np.array(kp_l1)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter([m.pt[0] for m in kp_l0[idx]], [m.pt[1] for m in kp_l0[idx]], c='r', s=8)
    plt.scatter([m.pt[0] for m in kp_l0[not_idx]], [m.pt[1] for m in kp_l0[not_idx]], c='b', s=5)
    plt.title('left0: Points that agree with the transformation in red, disagree in blue')
    plt.subplot(2, 1, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter([m.pt[0] for m in kp_l1[idx]], [m.pt[1] for m in kp_l1[idx]], c='r', s=8)
    plt.scatter([m.pt[0] for m in kp_l1[not_idx]], [m.pt[1] for m in kp_l1[not_idx]], c='b', s=5)
    plt.title('left1: Points that agree with the transformation in red, disagree in blue')

    # plt.show()


# def find_concensus_points_and_idx(good_matches_lr_0, matches_l0_l1, good_matches_lr1):
#     con = []
#     matches = []
#     for match_l_l in matches_l0_l1:
#         kp_ll_0 = match_l_l.queryIdx
#         kp_ll_1 = match_l_l.trainIdx
#         for i0 in range(len(good_matches_lr_0)):
#             if kp_ll_0 == good_matches_lr_0[i0].queryIdx:
#                 for i1 in range(len(good_matches_lr1)):
#                     if kp_ll_1 == good_matches_lr1[i1].queryIdx:
#                         con.append((i0, i1))
#                         matches.append((good_matches_lr_0[i0], good_matches_lr1[i1]))
#     return np.array(con), np.array(matches)

def find_concensus_points_and_idx(good_matches_lr_0, matches_l0_l1, good_matches_lr1):
    # Create dictionaries for quick lookup
    dict_lr_0 = {match.queryIdx: i for i, match in enumerate(good_matches_lr_0)}
    dict_lr_1 = {match.queryIdx: i for i, match in enumerate(good_matches_lr1)}

    con = []
    matches = []

    for match_l_l in matches_l0_l1:
        kp_ll_0 = match_l_l.queryIdx
        kp_ll_1 = match_l_l.trainIdx

        # Check if the match exists in both dictionaries
        if kp_ll_0 in dict_lr_0 and kp_ll_1 in dict_lr_1:
            i0 = dict_lr_0[kp_ll_0]
            i1 = dict_lr_1[kp_ll_1]
            con.append((i0, i1))
            matches.append((good_matches_lr_0[i0], good_matches_lr1[i1]))

    return np.array(con), np.array(matches)

# def find_concensus_points_and_idx(good_matches_lr_0, matches_l0_l1, good_matches_lr1):
#     con = []
#     matches = []
#     for match_l_l in matches_l0_l1:
#         kp_ll_0 = match_l_l.queryIdx
#         kp_ll_1 = match_l_l.trainIdx
#         for i0 in range(len(good_matches_lr_0)):
#             if kp_ll_0 == good_matches_lr_0[i0].queryIdx:
#                 for i1 in range(len(good_matches_lr1)):
#                     if kp_ll_1 == good_matches_lr1[i1].queryIdx:
#                         con.append((i0, i1))
#                         matches.append((good_matches_lr_0[i0], good_matches_lr1[i1]))
#                         break
#     return np.array(con), np.array(matches)




def check_transform_agreed(T, matches_3d_l0, consensus_matches, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    # todo: to change the calculation of r1 to the M2@(L1 POINTS)

    # extract y values from the matches
    real_l_0_pix_val, real_r_0_pix_val = extract_y_values(consensus_matches[:, 0], kp_l_first, kp_r_first)
    real_l_1_pix_val, real_r_1_pix_val = extract_y_values(consensus_matches[:, 1], kp_l_second, kp_r_second)
    ones = np.ones((matches_3d_l0.shape[0], 1))
    matches_in_4d = np.hstack((matches_3d_l0, ones)).T


    # 3d points to first camera coordinate system
    pts3d_to_l0 = (M1 @ matches_in_4d)
    # pts3d_to_l0 = matches_in_4d
    # 3d points to second camera coordinate system
    pts3d_to_r0 = (M2 @ matches_in_4d)

    pix_values_l_0 = (K @ pts3d_to_l0).T
    pix_values_l_0 = pix_values_l_0 / pix_values_l_0[:, 2][:, np.newaxis]
    pix_values_l_0_y = pix_values_l_0[:, 1]
    agrees_l0 = np.abs(real_l_0_pix_val - pix_values_l_0_y) < 2


    pix_values_r_0 = (K @ pts3d_to_r0).T
    pix_values_r_0 = pix_values_r_0 / pix_values_r_0[:, 2][:, np.newaxis]
    pix_values_r_0_y = pix_values_r_0[:, 1]
    agrees_r0 = np.abs(real_r_0_pix_val - pix_values_r_0_y) < 2


    # to_4d = np.hstack((pts3d_to_l0.T, ones)).T
    l1_pts = (T @ matches_in_4d)
    pix_values_l_1 = (K @ l1_pts).T
    pix_values_l_1 = pix_values_l_1 / pix_values_l_1[:, 2][:, np.newaxis]
    pix_values_l_1_y = pix_values_l_1[:, 1]
    agrees_l1 = np.abs(real_l_1_pix_val - pix_values_l_1_y) < 2


    to_4d = np.hstack((l1_pts.T, ones)).T
    pix_values_r_1 = (K @ M2 @ to_4d).T
    pix_values_r_1 = pix_values_r_1 / pix_values_r_1[:, 2][:, np.newaxis]
    pix_values_r_1_y = pix_values_r_1[:, 1]
    agrees_r1 = np.abs(real_r_1_pix_val - pix_values_r_1_y) < 2

    # agree_all = np.logical_and(agrees_l0 , agrees_r0 , agrees_l1 , agrees_r1)
    agree_all = agrees_l0 & agrees_l1 & agrees_r0 & agrees_r1
    # points = np.where(agree_all)
    # return points
    return agree_all


# def ransac_pnp(matches_idx, matches, traingulated_pts, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
#     """ Perform RANSAC to find the best transformation"""
#     best_inliers = 0
#     best_T = None
#     best_matches_idx = None
#     relevant_3d_pts = traingulated_pts[matches_idx[:, 0]]
#     for i in range(RANSAC_ITERATIONS):
#         random_idx = np.random.choice(matches_idx.shape[0], 4, replace=False)
#         random_matches = matches[random_idx]
#         img_pts = np.array([kp_l_second[m[1].queryIdx].pt for m in random_matches]).reshape(-1, 2, 1)
#         random_traingulated_pts = relevant_3d_pts[random_idx].reshape(-1, 3, 1)
#         diff_coeff = np.zeros((5, 1))
#         success, rotation_vector, translation_vector = cv2.solvePnP(random_traingulated_pts, img_pts, K,
#                                                                     distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
#         if success:
#             T = rodriguez_to_mat(rotation_vector, translation_vector)
#         else:
#             continue
#         # T = cv2.solvePnPRansac(random_traingulated_pts, random_matches, K, None, flags=cv2.SOLVEPNP_EPNP)

#         inliers_mat = check_transform_agreed(T, relevant_3d_pts, matches, kp_l_first, kp_r_first, kp_l_second,
#                                              kp_r_second)

#         inliers_idx = np.where(inliers_mat)
#         # inliers = relevant_3d_pts[inliers_idx]
#         if np.sum(inliers_mat) > best_inliers:
#             best_inliers = np.sum(inliers_mat)
#             best_T = T
#             best_matches_idx = inliers_idx

#     return best_T, best_matches_idx


# def check_transform_agreed(T_new, matches_3d_l0, consensus_matches, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
#     """
#     Checks if the transformation agrees with the given matches and keypoints.
#
#     Args:
#     - T (np.ndarray): Transformation matrix.
#     - matches_3d_l0 (np.ndarray): 3D points of the matches from the first left camera.
#     - consensus_matches (np.ndarray): Array of consensus matches between images.
#     - kp_l_first (list of cv2.KeyPoint): Keypoints from the first left image.
#     - kp_r_first (list of cv2.KeyPoint): Keypoints from the first right image.
#     - kp_l_second (list of cv2.KeyPoint): Keypoints from the second left image.
#     - kp_r_second (list of cv2.KeyPoint): Keypoints from the second right image.
#
#     Returns:
#     - agree_all (np.ndarray): Boolean array indicating agreement for each match.
#     """
#     # Extract y values from the matches
#     agreed_points_idx = []
#     for i in range(len(matches_3d_l0)):
#         p = matches_3d_l0[i]
#         p4d = np.array([p[0], p[1], p[2], 1])
#         pl0 = (P @ p4d)
#         pl0 = pl0 / pl0[2]
#         pl0_y = pl0[1]
#         p1 = np.array([pl0[0], pl0[1], pl0[2], 1])
#         q = np.vstack((M1, np.array([0, 0, 0, 1])))
#         pr0 = (K @ M2 @ q @ p4d)
#         pr0 = pr0 / pr0[2]
#         pr0_y = pr0[1]
#         pl1 = (K @ T_new @ p4d)
#         pl1 = pl1 / pl1[2]
#         pl1_y = pl1[1]
#         p1 = np.array([pl1[0], pl1[1], pl1[2], 1])
#         q = np.vstack((T_new, np.array([0, 0, 0, 1])))
#         pr1 = (K @ M2 @ q @ p4d)
#         pr1_y = pr1[1]
#
#         real_pl0_y = kp_l_first[consensus_matches[i, 0].queryIdx].pt[1]
#         real_pr0_y = kp_r_first[consensus_matches[i, 0].trainIdx].pt[1]
#         real_pl1_y = kp_l_second[consensus_matches[i, 1].queryIdx].pt[1]
#         real_pr1_y = kp_r_second[consensus_matches[i, 1].trainIdx].pt[1]

#         #
#         # if abs(pl0_y - real_pl0_y) > 2 or abs(pr0_y - real_pr0_y) > 2 or abs(pl1_y - real_pl1_y) > 2 or abs(
#         #         pr1_y - real_pr1_y) > 2:
#         #     agreed_points_idx.append(False)
#         # else:
#         #     agreed_points_idx.append(True)
#         if abs(pl0_y - real_pl0_y) < 2 and abs(pl1_y - real_pl1_y) < 2 :
#             agreed_points_idx.append(True)
#         else:
#             agreed_points_idx.append(False)
#     return np.array(agreed_points_idx)
#     real_l_0_pix_val, real_r_0_pix_val = extract_y_values(consensus_matches[:, 0], kp_l_first, kp_r_first)
#     real_l_1_pix_val, real_r_1_pix_val = extract_y_values(consensus_matches[:, 1], kp_l_second, kp_r_second)
#
#     # Append ones to the 3D matches
#     ones = np.ones((matches_3d_l0.shape[0], 1))
#     matches_in_4d = np.hstack((matches_3d_l0, ones)).T
#
#     # 3D points to first camera coordinate system
#     pts3d_to_l0 = (M1 @ matches_in_4d)
#
#     # 3D points to second camera coordinate system
#     pts3d_to_r0 = (M2 @ matches_in_4d)
#
#     # Project points to image coordinates for the first left camera
#     pix_values_l_0 = (K @ pts3d_to_l0).T
#     pix_values_l_0 = (pix_values_l_0 / pix_values_l_0[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_l_0_y = pix_values_l_0[:, 1]
#     agrees_l0 = np.abs(real_l_0_pix_val - pix_values_l_0_y) < 2
#
#     # Project points to image coordinates for the first right camera
#     pix_values_r_0 = (K @ pts3d_to_r0).T
#     pix_values_r_0 = (pix_values_r_0 / pix_values_r_0[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_r_0_y = pix_values_r_0[:, 1]
#     agrees_r0 = np.abs(real_r_0_pix_val - pix_values_r_0_y) < 2
#
#     # Apply transformation and project to second left camera
#     # to_4d = np.hstack((pts3d_to_l0.T, ones)).T
#     pix_values_l_1 = (K @ T @ matches_in_4d).T
#     pix_values_l_1 = (pix_values_l_1 / pix_values_l_1[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_l_1_y = pix_values_l_1[:, 1]
#     agrees_l1 = np.abs(real_l_1_pix_val - pix_values_l_1_y) < 2
#
#     # Apply transformation and project to second right camera
#     # to_4d = np.hstack((pts3d_to_r0.T, ones)).T
#     pix_values_r_1 = (T @ matches_in_4d).T
#     pix_values_r_1 = np.hstack((pix_values_r_1, ones)).T
#     pix_values_r_1 = (K @ M2 @ pix_values_r_1).T
#     pix_values_r_1 = (pix_values_r_1 / pix_values_r_1[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_r_1_y = pix_values_r_1[:, 1]
#     agrees_r1 = np.abs(real_r_1_pix_val - pix_values_r_1_y) < 2
#
#     # Check agreement for all matches
#     agree_all1 = np.logical_and(agrees_l0, agrees_r0)
#     agree_all2 = np.logical_and(agrees_l1, agrees_r1)
#     agree_all = np.logical_and(agree_all1, agree_all2)
#     idx = agree_all == True
#
#     return np.array(idx)


def transformation_agreement(T, consensus_matches, points_3d, kp_l0, kp_r0, kp_l1, kp_r1):


    T_4x4 = np.vstack((T, np.array([0, 0, 0, 1])))
    points_4d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    l1_T = K @ T
    l1_4d_points = (T_4x4 @ points_4d)
    to_the_right = (K @ M2)

    real_y_values = np.array([[kp_l0[m[0].queryIdx].pt[1],
                               kp_r0[m[0].trainIdx].pt[1],
                               kp_l1[m[1].queryIdx].pt[1],
                               kp_r1[m[1].trainIdx].pt[1],
                               ] for m in consensus_matches])

    # transform_to_l0_points = (P @ points_4d).T
    # transform_to_l0_points = transform_to_l0_points / transform_to_l0_points[:, 2][:, np.newaxis]
    # # transform_y_values = transform_to_l0_points[:, 1]
    # # real_y_values = np.array([kpr0[m[0].trainIdx].pt[1] for m in consensus_matches])
    # real_y = real_y_values[:, 0]
    # agree_l0 = np.abs(transform_to_l0_points[:, 1] - real_y) < 2
    # if np.sum(agree_l0) != len(agree_l0):


    transform_to_r0_points = (to_the_right @ points_4d).T
    transform_to_r0_points = transform_to_r0_points / transform_to_r0_points[:, 2][:, np.newaxis]
    # transform_y_values = transform_to_r0_points[:, 1]
    # real_y_values = np.array([kpr0[m[0].trainIdx].pt[1] for m in consensus_matches])
    real_y = real_y_values[:, 1]
    agree_r0 = np.abs(transform_to_r0_points[:, 1] - real_y) < 2

    # transformed_to_l1_points = (l1_T @ points_4d).T
    transformed_to_l1_points = (K @ l1_4d_points[:3, :]).T
    transformed_to_l1_points = transformed_to_l1_points / transformed_to_l1_points[:, 2][:, np.newaxis]
    # transform_y_values = transformed_to_l1_points[:, 1]
    # real_y_values = np.array([kpl1[m[1].queryIdx].pt[1] for m in consensus_matches])
    real_y = real_y_values[:, 2]
    agree_l1 = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2

    # r1_T = K @ M2 @ (np.vstack((T, np.array([0, 0, 0, 1]))))
    # transformed_to_r1_points = (r1_T @ points_4d).T
    transformed_to_r1_points = (to_the_right @ l1_4d_points).T
    transformed_to_r1_points = transformed_to_r1_points / transformed_to_r1_points[:, 2][:, np.newaxis]
    # transform_y_values = transformed_to_r1_points[:, 1]
    # real_y_values = np.array([kpr1[m[1].trainIdx].pt[1] for m in consensus_matches])
    real_y = real_y_values[:, 3]
    agree_r1 = np.abs(transformed_to_r1_points[:, 1] - real_y) < 2

    return np.logical_and(agree_l1, agree_r1, agree_r0)


def ransac_pnp(inliers_traingulated_pts, best_matches_pairs, kp_l_0, kp_r_0, kp_l_1, kp_r_1):
    """ Perform RANSAC to find the best transformation"""
    best_inliers = 0
    best_T = None
    best_matches_idx = None
    diff_coeff = np.zeros((5, 1))
    points_pixel_values = np.array([kp_l_1[m[1].queryIdx].pt for m in best_matches_pairs])
    for i in range(RANSAC_ITERATIONS):
        # Randomly select 4 points in the world coordinate system
        random_idx = np.random.choice(len(inliers_traingulated_pts), 4, replace=False)
        random_world_points = inliers_traingulated_pts[random_idx]
        # random_matches = best_matches_pairs[random_idx]
        # Get the corresponding pixel values in l1
        # random_points_pixel_values = np.array([kp_l_1[m[1].queryIdx].pt for m in random_matches])
        random_points_pixel_values = points_pixel_values[random_idx]


        # solve PnP problem to get the transformation
        success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_points_pixel_values, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
        if success:
            T = rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue
        # points_agreed = check_transform_agreed(T, inliers_traingulated_pts, best_matches_pairs, kp_l_0,
        #                                        kp_r_0,
        #                                        kp_l_1,
        #                                        kp_r_1)
        points_agreed = transformation_agreement(T, best_matches_pairs, inliers_traingulated_pts, kp_l_0, kp_r_0,
                                                 kp_l_1, kp_r_1)

        inliers_idx = np.where(points_agreed == True)
        if np.sum(points_agreed) > best_inliers:
            best_inliers = np.sum(points_agreed)
            best_T = T
            best_matches_idx = inliers_idx

    return best_T, best_matches_idx


def find_best_transformation(traingulated_pts, matches, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    """ Find the best transformation using RANSAC"""
    T, inliers_idx = ransac_pnp(traingulated_pts, matches, kp_l_first, kp_r_first, kp_l_second,
                                kp_r_second)

    # return T, inliers_idx


    # return T,inliers_idx
    # relevant_3d_pts = traingulated_pts[matches_idx[:, 0]][inliers_idx]
    # relevant_3d_pts = relevant_3d_pts[inliers_idx]
    best_matches = matches[inliers_idx[0]]
    img_pts = np.array([kp_l_second[m[1].queryIdx].pt for m in best_matches])
    diff_coeff = np.zeros((5, 1))
    pt_3d = traingulated_pts[inliers_idx[0]]
    success, rotation_vector, translation_vector = cv2.solvePnP(pt_3d, img_pts, K,
                                                                distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
    if success:
        return rodriguez_to_mat(rotation_vector, translation_vector), inliers_idx[0]
    return None


def perform_tracking(first_indx):
    global_transformations = []

    img_l_0, img_r_0 = ex2.read_images(first_indx)
    kp_l_0, kp_r_0, desc_l_0, desc_r_0, matches_f_0 = extract_kps_descs_matches(img_l_0, img_r_0)
    # in_f_0, out_f_0 = ex2.extract_inliers_outliers(kp_l_0, kp_r_0, matches_f_0)
    in_f_0, out_f_0 = extract_inliers_outliers(kp_l_0, kp_r_0, matches_f_0)
    in_f_0 = np.array(in_f_0)

    for i in range(first_indx + 1, LEN_DATA_SET):
        if i%10 == 0:
            print(f" frame {i} is being processed")
        # load the next frames and extract the keypoints and descriptors
        img_l_1, img_r_1 = ex2.read_images(i)
        kp_l_1, kp_r_1, desc_l_1, desc_r_1, matches_f_1 = extract_kps_descs_matches(img_l_1, img_r_1)

        # extract the inliers and outliers and triangulate the points

        in_f_1, out_f_1 = extract_inliers_outliers(kp_l_1, kp_r_1, matches_f_1)
        in_f_1 = np.array(in_f_1)

        # extract matches of first left frame and the second left frame
        matches_l_l = MATCHER.match(desc_l_0, desc_l_1)
        matches_l_l = sorted(matches_l_l, key=lambda x: x.distance)[:300]
        # find the concensus matches
        # good matches idx is np.array. each element is a tuple (i0,i1)
        # where i0 is the index of the match in in_f_0 and i1 is the index of the match in in_f_1
        good_matches_idx, matches_pairs = find_concensus_points_and_idx(in_f_0, matches_l_l, in_f_1)

        # triangulate the points only the good matches points from in_f_0
        # start = timer()
        l_0_best_inliers_idx = good_matches_idx[:, 0]
        l_0_best_inliers = in_f_0[l_0_best_inliers_idx]
        traingulated_pts = ex2.cv_triangulate_matched_points(l_0_best_inliers, kp_l_0, kp_r_0, P, Q)

        # find the best transformation
        relative_transformation, idx = find_best_transformation(traingulated_pts, matches_pairs, kp_l_0, kp_r_0, kp_l_1,
                                                                kp_r_1)
        # end = timer()
        # print(f"Triangulation time: {end - start}")
        # calculate the global transformation
        # global_trasnform = relative_transformation
        last_transformation = global_transformations[-1] if len(global_transformations) > 0 else M1
        global_trasnform = relative_transformation @ np.vstack((last_transformation, np.array([0, 0, 0, 1])))
        global_transformations.append(global_trasnform)

        # update the keypoints, descriptors and matches
        kp_l_0, kp_r_0 = kp_l_1, kp_r_1
        desc_l_0 = desc_l_1
        matches_f_0 = matches_f_1
        in_f_0 = in_f_1
    return global_transformations


def q3_1(kp_l_first, kp_r_first, matches_first, kp_l_second, kp_r_second, matches_second, P, Q):
    tr_0, inliers_0, out_0 = extract_inliers_outliers_triangulate(P, Q, kp_l_first, kp_r_first, matches_first)
    tr_1, inliers_1, out_1 = extract_inliers_outliers_triangulate(P, Q, kp_l_second, kp_r_second, matches_second)
    return tr_0, tr_1, inliers_0, inliers_1


def q3_2():
    return extract_matches_from_images(l_0_img, l_1_img)[2]


def q3_3(inl_lr_0, matches_l, inl_lr_1, triangulated_pts_0):
    con = choose_4_points(inl_lr_0, matches_l, inl_lr_1)
    # choose random 4 points
    samps = random.sample(con, 4)
    image_pts = [l_kp_1[inl_lr_1[i[1]].queryIdx].pt for i in samps]
    tri_samps = []
    for match in samps:
        tri_samps.append(triangulated_pts_0[match[0]])
    tri_samps = np.array(tri_samps)
    image_pts = np.array(image_pts)
    diff_coeff = np.zeros((5, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(tri_samps, image_pts, K, distCoeffs=diff_coeff,
                                                                flags=cv2.SOLVEPNP_EPNP)
    if success:

        T = rodriguez_to_mat(rotation_vector, translation_vector)
        camera_positions = calculate_4_camera_locations(T)
        camera_to_mat = np.array([camera_positions[i] for i in range(4)])

        # Camera labels in specified order
        camera_labels = ['left0', 'right0', 'left1', 'right1']

        # Extract X and Z coordinates for top-down view (ignore Y)
        x_coords = camera_to_mat[:, 0]
        z_coords = camera_to_mat[:, 2]

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, z_coords, color='blue', marker='o')

        # Annotate the points with their labels for clarity
        for i, (x, z) in enumerate(zip(x_coords, z_coords)):
            plt.annotate(camera_labels[i], (x, z), textcoords="offset points", xytext=(0, 10), ha='center')

        # Adding labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')
        plt.title('Relative Position of the Four Cameras (Top-Down View)')
        plt.grid(True)
        plt.axis('equal')  # Ensure the aspect ratio is equal
        # plt.show()

        # Show the plot
        return T
    else:
        return None


def q3_4(relevant_idx, match, traingulated_pts, T, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    """ check which points are consistent with the transformation T"""
    relevent_3d_pts = traingulated_pts[relevant_idx[:, 0]]
    agreed_matrix = transformation_agreement(T, match,relevent_3d_pts, kp_l_first, kp_r_first, kp_l_second,
                                           kp_r_second)
    idx = np.where(agreed_matrix)
    not_idx = np.where(~agreed_matrix)
    agreed_points = relevent_3d_pts[idx]
    disagreed_points = relevent_3d_pts[~agreed_matrix]
    print(f"Number of points that agree with the transformation: {len(agreed_points)}")
    print(f"Number of points that disagree with the transformation: {len(disagreed_points)}")
    return idx, not_idx


def q3_5(matches_idx, matches, traingulated_pts, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    T, best_idx = find_best_transformation(matches, matches_idx, traingulated_pts, kp_l_first, kp_r_first, kp_l_second,
                                           kp_r_second, M1)

    transformed_pair0 = T @ (np.hstack((triangulated_pts_0, np.ones((triangulated_pts_0.shape[0], 1))))).T
    transformed_pair0 = transformed_pair0.T
    pair1 = triangulated_pts_1
    # Plotting
    plt.figure(figsize=(8, 6))

    # Extract X and Z coordinates for pair 0 after transformation
    x_coords_transformed = transformed_pair0[:, 0]
    z_coords_transformed = transformed_pair0[:, 2]

    # Extract X and Z coordinates for pair 1
    x_coords_pair1 = pair1[:, 0]
    z_coords_pair1 = pair1[:, 2]

    # Scatter plot for pair 0 after transformation (in blue)
    plt.scatter(x_coords_transformed, z_coords_transformed, color='blue', label='Pair 0 (transformed)')

    # Scatter plot for pair 1 (in orange)
    plt.scatter(x_coords_pair1, z_coords_pair1, color='orange', label='Pair 1')

    # Adding labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Point Clouds from Above')
    plt.grid(True)
    plt.legend()

    # Set appropriate limits to crop unnecessary areas
    plt.xlim(min(np.min(x_coords_transformed), np.min(x_coords_pair1)) - 1,
             max(np.max(x_coords_transformed), np.max(x_coords_pair1)) + 1)
    plt.ylim(min(np.min(z_coords_transformed), np.min(z_coords_pair1)) - 1,
             max(np.max(z_coords_transformed), np.max(z_coords_pair1)) + 1)

    # Show the plot
    # plt.show()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(l_0_img, cmap='gray')
    plt.scatter([l_kp_0[m.queryIdx].pt[0] for m in inl_lr_0], [l_kp_0[m.queryIdx].pt[1] for m in inl_lr_0], c='r', s=4)

    # plt.scatter([m.pt[0] for m in kp_l0[idx]],[m.pt[1] for m in kp_l0[idx]], c='r', s=8)
    # plt.scatter([m.pt[0] for m in kp_l0[not_idx]],[m.pt[1] for m in kp_l0[not_idx]], c='b', s=5)
    plt.title('left0: inliers')
    plt.subplot(2, 1, 2)
    plt.imshow(l_1_img, cmap='gray')
    plt.scatter([l_kp_1[m.queryIdx].pt[0] for m in inl_lr_1], [l_kp_1[m.queryIdx].pt[1] for m in inl_lr_1], c='r', s=4)
    # plt.scatter([m.pt[0] for m in kp_l1[idx]], [m.pt[1] for m in kp_l1[idx]], c='r', s=8)
    # plt.scatter([m.pt[0] for m in kp_l1[not_idx]], [m.pt[1] for m in kp_l1[not_idx]], c='b', s=5)
    plt.title('left1: inliers')
    # plt.show()
    return T, best_idx

    # return T, best_idx


def calculate_camera_locations(camera_transformations):
    loc = np.array([0, 0, 0])
    for T in camera_transformations:
        R = T[:, :3]
        t = T[:, 3:]
        loc = np.vstack((loc, (-R.transpose() @ t).reshape((3))))
    return loc


def q3_6():
    start = timer()
    transformations = perform_tracking(0)
    end = timer()
    minutes = (end - start) // 60
    seconds = (end - start) % 60
    print(f"Time taken to perform tracking: {minutes} minutes and {seconds} seconds")
    transformations2 = read_extrinsic_matrices(n=LEN_DATA_SET)
    camera_location = calculate_camera_locations(transformations)
    ground_truth_location = calculate_camera_locations(transformations2)

    x_coords = camera_location[:, 0]
    realx = ground_truth_location[:, 0]
    z_coords = camera_location[:, 2]
    realz = ground_truth_location[:, 2]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, z_coords, color='blue', marker='o')
    plt.scatter(realx, realz, color='red', marker='o')

    # Annotate the points with their labels for clarity
    # for i, (x, z) in enumerate(zip(x_coords, z_coords)):
    #     plt.annotate(camera_labels[i], (x, z), textcoords="offset points", xytext=(0, 10), ha='center')

    # Adding labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Relative Position of the Four Cameras (Top-Down View)')
    plt.grid(True)
    plt.axis('equal')  # Ensure the aspect ratio is equal
    plt.show()

    return camera_location, transformations


if __name__ == '__main__':
    l_0_img, r_0_img = ex2.read_images(0)
    l_kp_0, r_kp_0, matches_0 = extract_matches_from_images(l_0_img, r_0_img)

    l_1_img, r_1_img = ex2.read_images(1)
    l_kp_1, r_kp_1, matches_1 = extract_matches_from_images(l_1_img, r_1_img)
    #
    triangulated_pts_0, triangulated_pts_1, inl_lr_0, inl_lr_1 = q3_1(l_kp_0, r_kp_0, matches_0, l_kp_1, r_kp_1,
                                                                      matches_1, P, Q)
    # matches_l = q3_2()
    # t = q3_3(inl_lr_0, matches_l, inl_lr_1, triangulated_pts_0)
    # idx, match = find_concensus_points_and_idx(inl_lr_0, matches_l, inl_lr_1)
    # relevant_triangulated = triangulated_pts_0[idx[:, 0]]
    # ind, not_ind = q3_4(idx, match, triangulated_pts_0, t, kp_l_0, kp_r_0, kp_l_1, kp_r_1)
    # plot_points_on_img1_img2(ind, not_ind, matches_0, img_l_0, img_l_1, kp_l_0, kp_l_1)
    # T, inliers_idx = q3_5(idx, match, relevant_triangulated, kp_l_0, kp_r_0, kp_l_1, kp_r_1)
    # for i in tqdm(range(1,LEN_DATA_SET - 1)):

    loc, _ = q3_6()
