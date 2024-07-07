import ex2
import cv2
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
MAC = False
if MAC:
    DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
    # DATA_PATH = 'dataset/sequences/00'
else:
    DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'

SEC_PROB = 0.999999
OUTLIER_PROB = 0.45
MIN_SET_SIZE = 4
RANSAC_ITERATIONS = int(np.log(1 - SEC_PROB) / np.log(1 - np.power(1 - OUTLIER_PROB, MIN_SET_SIZE))) + 1

# RANSAC_ITERATIONS = 36
print(f"RANSAC iterations: {RANSAC_ITERATIONS}")
LEN_DATA_SET = len(os.listdir(DATA_PATH + '/image_0'))
# LEN_DATA_SET = 1500

GROUND_TRUTH_PATH = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\poses\00.txt"

FEATURE = cv2.AKAZE_create()
FEATURE.setThreshold(0.003)
MATCHER = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)


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
    kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    matches = MATCHER.match(desc0, desc1)
    return kp0, kp1, desc0, desc1, matches

def extract_inliers_outliers(kp_left, kp_right, matches):
    kp_left_pts = np.array([kp.pt for kp in kp_left])
    kp_right_pts = np.array([kp.pt for kp in kp_right])

    match_indices = np.array([(match.queryIdx, match.trainIdx) for match in matches])
    left_indices = match_indices[:, 0]
    right_indices = match_indices[:, 1]

    good_map1 = np.abs(kp_left_pts[left_indices, 1] - kp_right_pts[right_indices, 1]) < 2
    good_map2 = kp_left_pts[left_indices, 0] > kp_right_pts[right_indices, 0]

    inliers = np.where(good_map1 & good_map2)[0]
    outliers = np.where(~(good_map1 & good_map2))[0]

    return inliers, outliers

def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))

def choose_4_points(matches_lr_0, matches_l0_l1, matches_lr1):
    lr_0_dict = {match.queryIdx: i for i, match in enumerate(matches_lr_0)}
    lr1_dict = {match.queryIdx: i for i, match in enumerate(matches_lr1)}

    con = [(lr_0_dict[match.queryIdx], lr1_dict[match.trainIdx])
           for match in matches_l0_l1 if match.queryIdx in lr_0_dict and match.trainIdx in lr1_dict]
    return con

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


def extract_matches_from_images(img_0, img1):
    kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    matches = MATCHER.match(desc0, desc1)
    return kp0, kp1, matches


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
    points_l0 = np.array([kp_l0[m.queryIdx].pt for m in matches])
    points_l1 = np.array([kp_l1[m.trainIdx].pt for m in matches])
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
    dict_lr_0 = {match.queryIdx: i for i, match in enumerate(good_matches_lr_0)}
    dict_lr_1 = {match.queryIdx: i for i, match in enumerate(good_matches_lr1)}

    con = [(dict_lr_0[match.queryIdx], dict_lr_1[match.trainIdx])
           for match in matches_l0_l1 if match.queryIdx in dict_lr_0 and match.trainIdx in dict_lr_1]
    matches = [(good_matches_lr_0[i0], good_matches_lr1[i1]) for i0, i1 in con]

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


def transformation_agreement(T, consensus_matches, points_3d, kp_l0, kp_r0, kp_l1, kp_r1, x_condition=True):
    T_4x4 = np.vstack((T, np.array([0, 0, 0, 1])))
    points_4d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    l1_T = K @ T
    l1_4d_points = (T_4x4 @ points_4d)
    to_the_right = (K @ M2)
    x_l0, x_r0, x_l1, x_r1 = 0, 1, 2, 3
    y_l0, y_r0, y_l1, y_r1 = 4, 5, 6, 7

    real_x_y_values = np.array([[kp_l0[m[0].queryIdx].pt[0],
                                 kp_r0[m[0].trainIdx].pt[0],
                                 kp_l1[m[1].queryIdx].pt[0],
                                 kp_r1[m[1].trainIdx].pt[0],
                                 kp_l0[m[0].queryIdx].pt[1],
                                 kp_r0[m[0].trainIdx].pt[1],
                                 kp_l1[m[1].queryIdx].pt[1],
                                 kp_r1[m[1].trainIdx].pt[1],
                                 ] for m in consensus_matches])

    transform_to_l0_points = (P @ points_4d).T
    transform_to_l0_points = transform_to_l0_points / transform_to_l0_points[:, 2][:, np.newaxis]
    # transform_y_values = transform_to_l0_points[:, 1]
    # real_y_values = np.array([kpr0[m[0].trainIdx].pt[1] for m in consensus_matches])
    real_y = real_x_y_values[:, y_l0]
    agree_l0 = np.abs(transform_to_l0_points[:, 1] - real_y) < 2

    transform_to_r0_points = (to_the_right @ points_4d).T
    transform_to_r0_points = transform_to_r0_points / transform_to_r0_points[:, 2][:, np.newaxis]
    # transform_y_values = transform_to_r0_points[:, 1]
    # real_y_values = np.array([kpr0[m[0].trainIdx].pt[1] for m in consensus_matches])
    real_y = real_x_y_values[:, y_r0]
    agree_r0 = np.abs(transform_to_r0_points[:, 1] - real_y) < 2
    if x_condition:
        real_x_l = real_x_y_values[:, x_l0]
        real_x_r = real_x_y_values[:, x_r0]
        cond_x = real_x_l > real_x_r
    else:
        cond_x = np.ones_like(agree_r0)
    agree_0 = np.logical_and(agree_r0, cond_x, agree_l0)

    # transformed_to_l1_points = (l1_T @ points_4d).T
    transformed_to_l1_points = (K @ l1_4d_points[:3, :]).T
    transformed_to_l1_points = transformed_to_l1_points / transformed_to_l1_points[:, 2][:, np.newaxis]
    # transform_y_values = transformed_to_l1_points[:, 1]
    # real_y_values = np.array([kpl1[m[1].queryIdx].pt[1] for m in consensus_matches])
    real_y = real_x_y_values[:, y_l1]
    agree_l1 = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2

    # r1_T = K @ M2 @ (np.vstack((T, np.array([0, 0, 0, 1]))))
    # transformed_to_r1_points = (r1_T @ points_4d).T
    transformed_to_r1_points = (to_the_right @ l1_4d_points).T
    transformed_to_r1_points = transformed_to_r1_points / transformed_to_r1_points[:, 2][:, np.newaxis]
    # transform_y_values = transformed_to_r1_points[:, 1]
    # real_y_values = np.array([kpr1[m[1].trainIdx].pt[1] for m in consensus_matches])
    real_y = real_x_y_values[:, y_r1]
    agree_r1 = np.abs(transformed_to_r1_points[:, 1] - real_y) < 2
    if x_condition:
        real_x_l = real_x_y_values[:, x_l1]
        real_x_r = real_x_y_values[:, x_r1]
        cond_x = real_x_l > real_x_r
    else:
        cond_x = np.ones_like(agree_r1)
    agree_1 = np.logical_and(agree_r1, cond_x, agree_l1)
    return np.logical_and(agree_0, agree_1)

def ransac_pnp(inliers_triangulated_pts, best_matches_pairs, kp_l_0, kp_r_0, kp_l_1, kp_r_1):
    """ Perform RANSAC to find the best transformation"""
    best_inliers = 0
    best_T = None
    best_matches_idx = None
    diff_coeff = np.zeros((5, 1))
    points_pixel_values = np.array([kp_l_1[m[1].queryIdx].pt for m in best_matches_pairs])

    # Precompute matrix of homogeneous coordinates for triangulated points
    ones = np.ones((inliers_triangulated_pts.shape[0], 1))
    triangulated_points_4d = np.hstack((inliers_triangulated_pts, ones)).T

    # Precompute pixel values of all keypoints in consensus_matches
    real_x_y_values = np.array([[kp_l_0[m[0].queryIdx].pt[0], kp_r_0[m[0].trainIdx].pt[0],
                                 kp_l_1[m[1].queryIdx].pt[0], kp_r_1[m[1].trainIdx].pt[0],
                                 kp_l_0[m[0].queryIdx].pt[1], kp_r_0[m[0].trainIdx].pt[1],
                                 kp_l_1[m[1].queryIdx].pt[1], kp_r_1[m[1].trainIdx].pt[1]]
                                for m in best_matches_pairs])

    for _ in range(RANSAC_ITERATIONS):
        # Randomly select 4 points in the world coordinate system
        random_idx = np.random.choice(len(inliers_triangulated_pts), 4, replace=False)
        random_world_points = inliers_triangulated_pts[random_idx]
        random_points_pixel_values = points_pixel_values[random_idx]

        # Solve PnP problem to get the transformation
        success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_points_pixel_values, K, distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
        if success:
            T = rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue

        points_agreed = transformation_agreement_matrix(T, triangulated_points_4d, real_x_y_values)

        inliers_idx = np.where(points_agreed == True)
        if np.sum(points_agreed) > best_inliers:
            best_inliers = np.sum(points_agreed)
            best_T = T
            best_matches_idx = inliers_idx

    return best_T, best_matches_idx

def transformation_agreement_matrix(T, points_4d, real_x_y_values):
    T_4x4 = np.vstack((T, np.array([0, 0, 0, 1])))
    l1_T = K @ T
    l1_4d_points = (T_4x4 @ points_4d)
    to_the_right = K @ M2

    # Transform points to left and right images
    transform_to_l0_points = (P @ points_4d).T
    transform_to_l0_points /= transform_to_l0_points[:, 2][:, np.newaxis]
    real_y = real_x_y_values[:, 4]  # y_l0
    agree_l0 = np.abs(transform_to_l0_points[:, 1] - real_y) < 2

    transform_to_r0_points = (to_the_right @ points_4d).T
    transform_to_r0_points /= transform_to_r0_points[:, 2][:, np.newaxis]
    real_y = real_x_y_values[:, 5]  # y_r0
    agree_r0 = np.abs(transform_to_r0_points[:, 1] - real_y) < 2

    # Ensure x condition
    real_x_l = real_x_y_values[:, 0]  # x_l0
    real_x_r = real_x_y_values[:, 1]  # x_r0
    cond_x = real_x_l > real_x_r
    agree_0 = agree_r0 & cond_x & agree_l0

    transformed_to_l1_points = (K @ l1_4d_points[:3, :]).T
    transformed_to_l1_points /= transformed_to_l1_points[:, 2][:, np.newaxis]
    real_y = real_x_y_values[:, 6]  # y_l1
    agree_l1 = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2

    transformed_to_r1_points = (to_the_right @ l1_4d_points).T
    transformed_to_r1_points /= transformed_to_r1_points[:, 2][:, np.newaxis]
    real_y = real_x_y_values[:, 7]  # y_r1
    agree_r1 = np.abs(transformed_to_r1_points[:, 1] - real_y) < 2

    real_x_l = real_x_y_values[:, 2]  # x_l1
    real_x_r = real_x_y_values[:, 3]  # x_r1
    cond_x = real_x_l > real_x_r
    agree_1 = agree_r1 & cond_x & agree_l1

    return agree_0 & agree_1


def find_best_transformation(triangulated_pts, matches, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    T, inliers_idx = ransac_pnp(triangulated_pts, matches, kp_l_first, kp_r_first, kp_l_second, kp_r_second)

    if T is not None:
        best_matches = matches[inliers_idx]
        img_pts = np.array([kp_l_second[m[1].queryIdx].pt for m in best_matches])
        diff_coeff = np.zeros((5, 1))
        pt_3d = triangulated_pts[inliers_idx]

        if len(pt_3d) >= 4:
            success, rotation_vector, translation_vector = cv2.solvePnP(pt_3d, img_pts, K,
                                                                        distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
            if success:
                return rodriguez_to_mat(rotation_vector, translation_vector), inliers_idx
    return None



# def find_negative_z_indx(points):
# bad_pts_idx = np.where([points[:, 2] < 0])


def perform_tracking(first_indx):
    global_transformations = []

    img_l_0, img_r_0 = ex2.read_images(first_indx)
    kp_l_0, kp_r_0, desc_l_0, desc_r_0, matches_f_0 = extract_kps_descs_matches(img_l_0, img_r_0)
    matches_f_0 = np.array(matches_f_0)
    in_f_0 = matches_f_0[extract_inliers_outliers(kp_l_0, kp_r_0, matches_f_0)[0]]
    in_f_0 = np.array(in_f_0)

    for i in tqdm(range(first_indx + 1, LEN_DATA_SET)):
        # load the next frames and extract the keypoints and descriptors
        img_l_1, img_r_1 = ex2.read_images(i)
        # start = timer()
        kp_l_1, kp_r_1, desc_l_1, desc_r_1, matches_f_1 = extract_kps_descs_matches(img_l_1, img_r_1)
        matches_f_1 = np.array(matches_f_1)
        # end = timer()
        # print(f"Time to extract keypoints and descriptors: {end - start}")
        # extract the inliers and outliers and triangulate the points

        in_f_1 = matches_f_1[extract_inliers_outliers(kp_l_1, kp_r_1, matches_f_1)[0]]
        in_f_1 = np.array(in_f_1)

        # extract matches of first left frame and the second left frame
        # start = timer()
        matches_l_l = MATCHER.match(desc_l_0, desc_l_1)
        # l = len(matches_l_l)
        # matches_l_l = np.array(sorted(matches_l_l, key=lambda x: x.distance)[:500])
        # end = timer()
        # print(f"Time to match: {end - start}, number of matches: {l}")

        # find the concensus matches
        # good matches idx is np.array. each element is a tuple (i0,i1)
        # where i0 is the index of the match in in_f_0 and i1 is the index of the match in in_f_1
        # start = timer()
        good_matches_idx, matches_pairs = find_concensus_points_and_idx(in_f_0, matches_l_l, in_f_1)
        # triangulate the points only the good matches points from in_f_0
        l_0_best_inliers_idx = good_matches_idx[:, 0]
        l_0_best_inliers = in_f_0[l_0_best_inliers_idx]
        traingulated_pts = ex2.cv_triangulate_matched_points(l_0_best_inliers, kp_l_0, kp_r_0, P, Q)

        # find the best transformation
        relative_transformation, idx = find_best_transformation(traingulated_pts, matches_pairs, kp_l_0, kp_r_0, kp_l_1,
                                                                kp_r_1)
        # end = timer()
        # print(f"Time to find best transformation: {end - start}")

        # calculate the global transformation
        # global_trasnform = relative_transformation
        last_transformation = global_transformations[-1] if len(global_transformations) > 0 else M1
        global_trasnform = relative_transformation @ np.vstack((last_transformation, np.array([0, 0, 0, 1])))
        global_transformations.append(global_trasnform)

        # update the keypoints, descriptors and matches
        kp_l_0, kp_r_0 = kp_l_1, kp_r_1
        desc_l_0 = desc_l_1
        in_f_0 = in_f_1
    return global_transformations


def q3_1(kp_l_first, kp_r_first, matches_first, kp_l_second, kp_r_second, matches_second, P, Q):
    tr_0, inliers_0, out_0 = extract_inliers_outliers_triangulate(P, Q, kp_l_first, kp_r_first, matches_first)
    tr_1, inliers_1, out_1 = extract_inliers_outliers_triangulate(P, Q, kp_l_second, kp_r_second, matches_second)
    return tr_0, tr_1, inliers_0, inliers_1


def q3_2(img_l_0, img_l_1):
    return extract_matches_from_images(img_l_0, img_l_1)[2]


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


def q3_4(T, match, inliers_traingulated_pts, kp_l_first, kp_r_first, kp_l_second, kp_r_second, img_l0, img_l1):
    """ check which points are consistent with the transformation T"""
    supporters_idx = transformation_agreement(T, match, inliers_traingulated_pts, kp_l_first, kp_r_first, kp_l_second,
                                             kp_r_second, x_condition=False)
    supporters_idx = np.where(supporters_idx)[0]
    not_supporters_idx = np.where(~supporters_idx)[0]
    # agreed_points = inliers_traingulated_pts[idx]
    # disagreed_points = inliers_traingulated_pts[~agreed_matrix]
    print(f"Number of points that agree with the transformation: {len(supporters_idx)}")
    print(f"Number of points that disagree with the transformation: {len(not_supporters_idx)}")



    # plot on images left0 and left1 the matches, with supporters in different color.
    l0_supporters = [kp_l_first[match[i,0].queryIdx].pt for i in supporters_idx]
    l0_not_supporters = [kp_l_first[match[i,0].queryIdx].pt for i in not_supporters_idx]

    l1_supporters = [kp_l_second[match[i,1].queryIdx].pt for i in supporters_idx]
    l1_not_supporters = [kp_l_second[match[i,0].queryIdx].pt for i in not_supporters_idx]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img_l0, cmap='gray')
    plt.scatter([m[0] for m in l0_supporters], [m[1] for m in l0_supporters], c='r', s=2)
    plt.scatter([m[0] for m in l0_not_supporters], [m[1] for m in l0_not_supporters], c='b', s=2)
    plt.title('Q3.4: left_0, transformation supporters in red, other in blue')
    plt.subplot(2, 1, 2)
    plt.imshow(img_l1, cmap='gray')
    plt.scatter([m[0] for m in l1_supporters], [m[1] for m in l1_supporters], c='r', s=2)
    plt.scatter([m[0] for m in l1_not_supporters], [m[1] for m in l1_not_supporters], c='b', s=2)
    plt.title('Q3.4: left_1, transformation supporters in red, other in blue')
    # plt.show()

    return supporters_idx, not_supporters_idx


def q3_5(matches, traingulated_pts, kp_l_first, kp_r_first, kp_l_second, kp_r_second):

    T, best_idx = find_best_transformation(traingulated_pts, matches, kp_l_first, kp_r_first, kp_l_second,
                                           kp_r_second)

    transformed_pair0 = T @ (np.hstack((triangulated_pts_0, np.ones((triangulated_pts_0.shape[0], 1))))).T
    transformed_pair0 = transformed_pair0.T
    pair1 = triangulated_pts_1


    # Plotting the 3d point cloud
    x1, y1, z1 = transformed_pair0[:, 0], transformed_pair0[:, 1], transformed_pair0[:, 2]
    x2, y2, z2 = pair1[:, 0], pair1[:, 1], pair1[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    s1 = ax.scatter3D(x1, y1, z1, color='r', s=2)
    s2 = ax.scatter3D(x2, y2, z2, color='b', s=1.5)
    title1 = 'Q3.5: 3D Points from Pair 0 After Transformation (red) Pair 1 (Blue)'
    ax.set_title(title1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-50, 300)

    inlier_idx = np.where(best_idx)[0]

    outlayer_idx = np.where(best_idx == 0)[0]

    inliers_0 = [kp_l_first[matches[i, 0].queryIdx].pt for i in inlier_idx]
    outlayers_0 = [kp_l_first[matches[i, 0].queryIdx].pt for i in outlayer_idx]

    inliers_1 = [kp_l_second[matches[i, 1].queryIdx].pt for i in inlier_idx]
    outlayers_1 = [kp_l_second[matches[i, 1].queryIdx].pt for i in outlayer_idx]

    #plot on images left0 and left1 the inliers and outliers in different colors.

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(l_0_img, cmap='gray')
    plt.scatter([m[0] for m in inliers_0], [m[1] for m in inliers_0], c='r', s=2)
    plt.scatter([m[0] for m in outlayers_0], [m[1] for m in outlayers_0], c='b', s=2)
    plt.title('Q3.5: left_0, inliers in red, outliers in blue')
    plt.subplot(2, 1, 2)
    plt.imshow(l_1_img, cmap='gray')
    plt.scatter([m[0] for m in inliers_1], [m[1] for m in inliers_1], c='r', s=2)
    plt.scatter([m[0] for m in outlayers_1], [m[1] for m in outlayers_1], c='b', s=2)
    # plt.scatter([m.pt[0] for m in kp_l1[not_idx]], [m.pt[1] for m in kp_l1[not_idx]], c='b', s=5)
    plt.title('Q3.5: left_1, inliers in red, outliers in blue')

    #
    # # Extract X and Z coordinates for pair 0 after transformation
    # x_coords_transformed = transformed_pair0[:, 0]
    # z_coords_transformed = transformed_pair0[:, 2]
    #
    # # Extract X and Z coordinates for pair 1
    # x_coords_pair1 = pair1[:, 0]
    # z_coords_pair1 = pair1[:, 2]
    #
    # # Scatter plot for pair 0 after transformation (in blue)
    # plt.scatter(x_coords_transformed, z_coords_transformed, color='blue', label='Pair 0 (transformed)')
    #
    # # Scatter plot for pair 1 (in orange)
    # plt.scatter(x_coords_pair1, z_coords_pair1, color='orange', label='Pair 1')
    #
    # # Adding labels and title
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Z Coordinate')
    # plt.title('Point Clouds from Above')
    # plt.grid(True)
    # plt.legend()
    #
    # # Set appropriate limits to crop unnecessary areas
    # plt.xlim(min(np.min(x_coords_transformed), np.min(x_coords_pair1)) - 1,
    #          max(np.max(x_coords_transformed), np.max(x_coords_pair1)) + 1)
    # plt.ylim(min(np.min(z_coords_transformed), np.min(z_coords_pair1)) - 1,
    #          max(np.max(z_coords_transformed), np.max(z_coords_pair1)) + 1)
    #
    # # Show the plot
    # # plt.show()
    # plt.figure()

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
    matches_l = q3_2(img_l_0=l_0_img, img_l_1=l_1_img)
    t = q3_3(inl_lr_0, matches_l, inl_lr_1, triangulated_pts_0)
    concensus_idx, concensus_matches = find_concensus_points_and_idx(inl_lr_0, matches_l, inl_lr_1)
    triangulated_pts_0 = triangulated_pts_0[concensus_idx[:, 0]]
    ind, not_ind = q3_4(t,concensus_matches, triangulated_pts_0, l_kp_0, r_kp_0, l_kp_1, r_kp_1, l_0_img, l_1_img)
    T, inliers_idx = q3_5(concensus_matches, triangulated_pts_0,  l_kp_0, r_kp_0,l_kp_1, r_kp_1)
    # plt.show()
    loc, _ = q3_6()
