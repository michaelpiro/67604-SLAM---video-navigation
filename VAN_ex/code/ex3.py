import ex2
import cv2
import random
import matplotlib.pyplot as plt
import os
import numpy as np

MAC = True
if MAC:
    DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
    # DATA_PATH = 'dataset/sequences/00'
else:
    DATA_PATH = r'...\VAN_ex\dataset\sequences\00\\'
RANSAC_ITERATIONS = 100

LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))
LEN_DATA_SET = 100

GROUND_TRUTH_PATH = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt"


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
def read_extrinsic_matrices(file_path = GROUND_TRUTH_PATH, n = LEN_DATA_SET):
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
                raise ValueError(f"Line {i+1} does not contain 12 numbers.")
            matrix = np.array(numbers).reshape(3, 4)
            extrinsic_matrices.append(matrix)

    return extrinsic_matrices

def extract_kps_descs_matches(img_0, img1):
    kp0, desc0 = ex2.FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = ex2.FEATURE.detectAndCompute(img1, None)
    matches = ex2.MATCHER.match(desc0, desc1)
    return kp0, kp1, desc0, desc1, matches


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
    good_idx_l0 = [good_matches_lr_0[i].queryIdx for i in range(len(good_matches_lr_0))]
    good_idx_l1 = [good_matches_lr1[i].queryIdx for i in range(len(good_matches_lr1))]
    for match_l_l in matches_l0_l1:
        ind_in_kp_l_0 = match_l_l.queryIdx
        ind_in_kp_l_1 = match_l_l.trainIdx
        is_a_good_match_0 = ind_in_kp_l_0 in good_idx_l0


        for i0 in range(len(good_matches_lr_0)):
            if kp_ll_0 == good_matches_lr_0[i0].queryIdx:
                for i1 in range(len(good_matches_lr1)):
                    if kp_ll_1 == good_matches_lr1[i1].queryIdx:
                        con.append((i0, i1))
                        matches.append((good_matches_lr_0[i0], good_matches_lr1[i1]))
    return np.array(con), np.array(matches)


def find_concensus_points_and_idx(good_matches_lr_0, matches_l0_l1, good_matches_lr1):
    con = []
    matches = []
    for match_l_l in matches_l0_l1:
        kp_ll_0 = match_l_l.queryIdx
        kp_ll_1 = match_l_l.trainIdx
        for i0 in range(len(good_matches_lr_0)):
            if kp_ll_0 == good_matches_lr_0[i0].queryIdx:
                for i1 in range(len(good_matches_lr1)):
                    if kp_ll_1 == good_matches_lr1[i1].queryIdx:
                        con.append((i0, i1))
                        matches.append((good_matches_lr_0[i0], good_matches_lr1[i1]))
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


# def check_transform_agreed(T, matches_3d_l0, consensus_matches, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
#     # todo: to change the calculation of r1 to the M2@(L1 POINTS)
#
#     # extract y values from the matches
#     real_l_0_pix_val, real_r_0_pix_val = extract_y_values(consensus_matches[:, 0], kp_l_first, kp_r_first)
#     real_l_1_pix_val, real_r_1_pix_val = extract_y_values(consensus_matches[:, 1], kp_l_second, kp_r_second)
#     ones = np.ones((matches_3d_l0.shape[0], 1))
#     matches_in_4d = np.hstack((matches_3d_l0, ones)).T
#     # print(M1)
#
#     # 3d points to first camera coordinate system
#     pts3d_to_l0 = (M1 @ matches_in_4d)
#     # pts3d_to_l0 = matches_in_4d
#     # 3d points to second camera coordinate system
#     pts3d_to_r0 = (M2 @ matches_in_4d)
#
#     pix_values_l_0 = (K @ pts3d_to_l0).T
#     pix_values_l_0 = (pix_values_l_0 / pix_values_l_0[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_l_0_y = pix_values_l_0[:, 1]
#     agrees_l0 = np.abs(real_l_0_pix_val - pix_values_l_0_y) < 2
#
#     pix_values_r_0 = (K @ pts3d_to_r0).T
#     pix_values_r_0 = (pix_values_r_0 / pix_values_r_0[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_r_0_y = pix_values_r_0[:, 1]
#     agrees_r0 = np.abs(real_r_0_pix_val - pix_values_r_0_y) < 2
#
#     to_4d = np.hstack((pts3d_to_l0.T, ones)).T
#     pix_values_l_1 = (K @ T @ to_4d).T
#     pix_values_l_1 = (pix_values_l_1 / pix_values_l_1[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_l_1_y = pix_values_l_1[:, 1]
#     agrees_l1 = np.abs(real_l_1_pix_val - pix_values_l_1_y) < 2
#
#     to_4d = np.hstack((pts3d_to_r0.T, ones)).T
#     pix_values_r_1 = (K @ T @ to_4d).T
#     pix_values_r_1 = (pix_values_r_1 / pix_values_r_1[:, 2].reshape(-1, 1))[:, 0:2]
#     pix_values_r_1_y = pix_values_r_1[:, 1]
#     agrees_r1 = np.abs(real_r_1_pix_val - pix_values_r_1_y) < 2
#
#     agree_all = agrees_l0 & agrees_r0 & agrees_l1 & agrees_r1
#     # print(agree_all.shape)
#     # points = np.where(agree_all)
#     # return points
#     return agree_all


def check_transform_agreed(T, matches_3d_l0, consensus_matches, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    """
    Checks if the transformation agrees with the given matches and keypoints.

    Args:
    - T (np.ndarray): Transformation matrix.
    - matches_3d_l0 (np.ndarray): 3D points of the matches from the first left camera.
    - consensus_matches (np.ndarray): Array of consensus matches between images.
    - kp_l_first (list of cv2.KeyPoint): Keypoints from the first left image.
    - kp_r_first (list of cv2.KeyPoint): Keypoints from the first right image.
    - kp_l_second (list of cv2.KeyPoint): Keypoints from the second left image.
    - kp_r_second (list of cv2.KeyPoint): Keypoints from the second right image.

    Returns:
    - agree_all (np.ndarray): Boolean array indicating agreement for each match.
    """
    # Extract y values from the matches
    real_l_0_pix_val, real_r_0_pix_val = extract_y_values(consensus_matches[:, 0], kp_l_first, kp_r_first)
    real_l_1_pix_val, real_r_1_pix_val = extract_y_values(consensus_matches[:, 1], kp_l_second, kp_r_second)

    # Append ones to the 3D matches
    ones = np.ones((matches_3d_l0.shape[0], 1))
    matches_in_4d = np.hstack((matches_3d_l0, ones)).T

    # 3D points to first camera coordinate system
    pts3d_to_l0 = (M1 @ matches_in_4d)

    # 3D points to second camera coordinate system
    pts3d_to_r0 = (M2 @ matches_in_4d)

    # Project points to image coordinates for the first left camera
    pix_values_l_0 = (K @ pts3d_to_l0).T
    pix_values_l_0 = (pix_values_l_0 / pix_values_l_0[:, 2].reshape(-1, 1))[:, 0:2]
    pix_values_l_0_y = pix_values_l_0[:, 1]
    agrees_l0 = np.abs(real_l_0_pix_val - pix_values_l_0_y) < 2

    # Project points to image coordinates for the first right camera
    pix_values_r_0 = (K @ pts3d_to_r0).T
    pix_values_r_0 = (pix_values_r_0 / pix_values_r_0[:, 2].reshape(-1, 1))[:, 0:2]
    pix_values_r_0_y = pix_values_r_0[:, 1]
    agrees_r0 = np.abs(real_r_0_pix_val - pix_values_r_0_y) < 2

    # Apply transformation and project to second left camera
    to_4d = np.hstack((pts3d_to_l0.T, ones)).T
    pix_values_l_1 = (K @ T @ to_4d).T
    pix_values_l_1 = (pix_values_l_1 / pix_values_l_1[:, 2].reshape(-1, 1))[:, 0:2]
    pix_values_l_1_y = pix_values_l_1[:, 1]
    agrees_l1 = np.abs(real_l_1_pix_val - pix_values_l_1_y) < 2

    # Apply transformation and project to second right camera
    to_4d = np.hstack((pts3d_to_r0.T, ones)).T
    pix_values_r_1 = (K @ T @ to_4d).T
    pix_values_r_1 = (pix_values_r_1 / pix_values_r_1[:, 2].reshape(-1, 1))[:, 0:2]
    pix_values_r_1_y = pix_values_r_1[:, 1]
    agrees_r1 = np.abs(real_r_1_pix_val - pix_values_r_1_y) < 2

    # Check agreement for all matches
    agree_all = agrees_l0 & agrees_r0 & agrees_l1 & agrees_r1

    return agree_all

def ransac_pnp(matches_idx, matches, traingulated_pts, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    """ Perform RANSAC to find the best transformation"""
    best_inliers = 0
    best_T = None
    best_matches_idx = None
    relevant_3d_pts = traingulated_pts[matches_idx[:, 0]]
    for i in range(RANSAC_ITERATIONS):
        random_idx = np.random.choice(matches_idx.shape[0], 4, replace=False)
        random_matches = matches[random_idx]
        img_pts = np.array([kp_l_second[m[1].queryIdx].pt for m in random_matches]).reshape(-1, 2, 1)
        random_traingulated_pts = relevant_3d_pts[random_idx].reshape(-1, 3, 1)
        diff_coeff = np.zeros((5, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(random_traingulated_pts, img_pts, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
        if success:
            T = rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue
        # T = cv2.solvePnPRansac(random_traingulated_pts, random_matches, K, None, flags=cv2.SOLVEPNP_EPNP)
        # print(matches.shape)
        inliers_mat = check_transform_agreed(T, relevant_3d_pts, matches, kp_l_first, kp_r_first, kp_l_second,
                                             kp_r_second)
        # print(np.sum(inliers_mat))
        inliers_idx = np.where(inliers_mat)
        # inliers = relevant_3d_pts[inliers_idx]
        if np.sum(inliers_mat) > best_inliers:
            best_inliers = np.sum(inliers_mat)
            best_T = T
            best_matches_idx = inliers_idx
    print(f"Best number of inliers: {best_inliers}")
    return best_T, best_matches_idx


def find_best_transformation(matches, matches_idx, traingulated_pts, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    T, inliers_idx = ransac_pnp(matches_idx, matches, traingulated_pts, kp_l_first, kp_r_first, kp_l_second,
                                kp_r_second)
    # return T,inliers_idx
    relevant_3d_pts = traingulated_pts[matches_idx[:, 0]][inliers_idx]
    # relevant_3d_pts = relevant_3d_pts[inliers_idx]
    best_matches = matches[inliers_idx]
    img_pts = np.array([kp_l_second[m[1].queryIdx].pt for m in best_matches])
    diff_coeff = np.zeros((5, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(relevant_3d_pts, img_pts, K,
                                                                distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
    if success:
        return rodriguez_to_mat(rotation_vector, translation_vector), inliers_idx
    return None


def q3_1(kp_l_first, kp_r_first, matches_first, kp_l_second, kp_r_second, matches_second, P, Q):
    tr_0, inliers_0, out_0 = extract_inliers_outliers_triangulate(P, Q, kp_l_first, kp_r_first, matches_first)
    tr_1, inliers_1, out_1 = extract_inliers_outliers_triangulate(P, Q, kp_l_second, kp_r_second, matches_second)
    return tr_0, tr_1, inliers_0, inliers_1


def extract_inliers_outliers_triangulate(P, Q, kp_l_first, kp_r_first, matches_first):
    inliers_0, outliers_0 = ex2.extract_inliers_outliers(kp_l_first, kp_r_first, matches_first)
    triangulated_0 = ex2.triangulate_matched_points(P, Q, inliers_0, kp_l_first, kp_r_first)
    return triangulated_0, inliers_0, outliers_0

def extract_matches_from_images(img_0, img1):
    kp0, desc0 = ex2.FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = ex2.FEATURE.detectAndCompute(img1, None)
    matches = ex2.MATCHER.match(desc0, desc1)
    return kp0, kp1, matches


def q3_2():
    return extract_matches_from_images(img_l_0, img_l_1)[2]


def q3_3(inl_lr_0, matches_l, inl_lr_1, triangulated_pts_0):
    con,_ = find_concensus_points_and_idx(inl_lr_0, matches_l, inl_lr_1)
    # choose random 4 points
    samps = random.sample(con, 4)
    image_pts = [kp_l_1[inl_lr_1[i[1]].queryIdx].pt for i in samps]
    tri_samps = []
    for match in samps:
        tri_samps.append(triangulated_pts_0[match[0]])
    tri_samps = np.array(tri_samps)
    image_pts = np.array(image_pts)
    diff_coeff = np.zeros((5, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(tri_samps, image_pts, K, distCoeffs=diff_coeff,
                                                                flags=cv2.SOLVEPNP_EPNP)
    if success:
        return rodriguez_to_mat(rotation_vector, translation_vector)
    else:
        return None


def q3_4(relevant_idx, match, traingulated_pts, T, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    """ check which points are consistent with the transformation T"""
    relevent_3d_pts = traingulated_pts[relevant_idx[:, 0]]
    agreed_matrix = check_transform_agreed(T, relevent_3d_pts, match, kp_l_first, kp_r_first, kp_l_second, kp_r_second)
    idx = np.where(agreed_matrix)
    agreed_points = relevent_3d_pts[idx]
    disagreed_points = relevent_3d_pts[~agreed_matrix]
    print(f"Number of points that agree with the transformation: {len(agreed_points)}")
    print(f"Number of points that disagree with the transformation: {len(disagreed_points)}")

    # print(point_cloud_3d)
    # print(f"Number of points that agree with the transformation: {len(point_cloud_3d[0])}")


def q3_5(matches_idx, matches, traingulated_pts, kp_l_first, kp_r_first, kp_l_second, kp_r_second):
    T, best_idx = find_best_transformation(matches, matches_idx, traingulated_pts, kp_l_first, kp_r_first, kp_l_second,
                                           kp_r_second)
    return T, best_idx


def perform_tracking(first_indx):
    first_cam_mat = M1
    intrinsics = K
    transformations = [M1]
    img_l_first, img_r_first = ex2.read_images(first_indx)
    kp_l_first, kp_r_first, desc_l_first, desc_r_first, matches_first = extract_kps_descs_matches(img_l_first,
                                                                                                  img_r_first)

    for i in range(first_indx + 1, LEN_DATA_SET):

        triangulated_first, in_first, out_first = extract_inliers_outliers_triangulate(P, Q, kp_l_first, kp_r_first,
                                                                                       matches_first)
        img_l_second, img_r_second = ex2.read_images(i)
        kp_l_second, kp_r_second, desc_l_second, desc_r_second, matches_second = extract_kps_descs_matches(img_l_second,
                                                                                                           img_r_second)
        in_second, out_second = ex2.extract_inliers_outliers(kp_l_second, kp_r_second, matches_second)
        matches_l_l = ex2.MATCHER.match(desc_l_first, desc_l_second)
        idx, match = find_concensus_points_and_idx(in_first, matches_l_l, in_second)
        T, inliers_idx = find_best_transformation(match, idx, triangulated_first, kp_l_first, kp_r_first, kp_l_second,
                                                  kp_r_second)
        # old_cam_mat = transformations[-1]
        # last_line = np.array([0, 0, 0, 1])
        # old_cam_mat = np.vstack((old_cam_mat, last_line))
        # print(T)
        # print(old_cam_mat)
        # new_cam_mat = T @ old_cam_mat
        # print(new_cam_mat)
        transformations.append(T)
        kp_l_first, kp_r_first = kp_l_second, kp_r_second
        desc_l_first, desc_r_first, matches_first = desc_l_second, desc_r_second, matches_second
    return transformations




if __name__ == '__main__':
    img_l_0, img_r_0 = ex2.read_images(0)
    kp_l_0, kp_r_0, matches_0 = extract_matches_from_images(img_l_0, img_r_0)

    img_l_1, img_r_1 = ex2.read_images(1)
    kp_l_1, kp_r_1, matches_1 = extract_matches_from_images(img_l_1, img_r_1)

    triangulated_pts_0, triangulated_pts_1, inl_lr_0, inl_lr_1 = q3_1(kp_l_0, kp_r_0, matches_0, kp_l_1, kp_r_1,
                                                                      matches_1, P, Q)
    matches_l = q3_2()
    T = q3_3(inl_lr_0, matches_l, inl_lr_1, triangulated_pts_0)
    idx, match = find_concensus_points_and_idx(inl_lr_0, matches_l, inl_lr_1)

    print(q3_4(idx, match, triangulated_pts_0, T, kp_l_0, kp_r_0, kp_l_1, kp_r_1))
    # print(ransac_pnp(idx, match, triangulated_pts_0))
    T, inliers_idx = q3_5(idx, match, triangulated_pts_0, kp_l_0, kp_r_0, kp_l_1, kp_r_1)

    #
    transformations = perform_tracking(0)
    extrinsic_matrices = read_extrinsic_matrices()
    diffs = []
    max = 0
    for i in range(1, LEN_DATA_SET):
        x = np.round(np.abs(transformations[i] - extrinsic_matrices[i]))
        # if x > max:
        #     max = x
        diffs.append(x)
    for i in range(len(diffs)):
        print(diffs[i])
        print(i)
        print("\n")

