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
RANSAC_ITERATIONS = 1000
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



def q3_1():

    inliers_1, outliers_1 = ex2.extract_inliers_outliers(kp_l_1,kp_r_1,matches_1)
    inliers_0, outliers_0 = ex2.extract_inliers_outliers(kp_l_0,kp_r_0,matches_0)
    triangulated_0 = ex2.triangulate_matched_points(P,Q,inliers_0,kp_l_0,kp_r_0)
    triangulated_1 = ex2.triangulate_matched_points(P,Q,inliers_1,kp_l_1,kp_r_1)
    return triangulated_0, triangulated_1,inliers_0,inliers_1

def q3_2():
    return ex2.extract_matches_from_images(img_l_0,img_l_1)[2]

def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def choose_4_points(matches_lr_0,matches_l0_l1,matches_lr1):
    con = []
    for match_l_l in matches_l0_l1:
        kp_ll_0 = match_l_l.queryIdx
        kp_ll_1 = match_l_l.trainIdx
        for i0 in range(len(matches_lr_0)):
            if kp_ll_0 == matches_lr_0[i0].queryIdx:
                for i1 in range(len(matches_lr1)):
                    if kp_ll_1 == matches_lr1[i1].queryIdx:
                        con.append((i0,i1))
    return con


def q3_3():
    con = choose_4_points(inl_lr_0,matches_l,inl_lr_1)
    #choose random 4 points
    samps = random.sample(con,4)
    image_pts = [kp_l_1[inl_lr_1[i[1]].queryIdx].pt for i in samps]
    tri_samps = []
    for match in samps:
        tri_samps.append(triangulated_pts_0[match[0]])
    tri_samps = np.array(tri_samps)
    image_pts = np.array(image_pts)
    success, rotation_vector, translation_vector = cv2.solvePnP(tri_samps, image_pts, K, distCoeffs=None,
                                                                flags=cv2.SOLVEPNP_EPNP)
    if success:
        return rodriguez_to_mat(rotation_vector, translation_vector)
        # rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # return rotation_matrix, translation_vector
    else:
        return None

def choose_4_points_that_returns_matches(matches_lr_0,matches_l0_l1,matches_lr1):
    con = []
    matches = []
    for match_l_l in matches_l0_l1:
        kp_ll_0 = match_l_l.queryIdx
        kp_ll_1 = match_l_l.trainIdx
        for i0 in range(len(matches_lr_0)):
            if kp_ll_0 == matches_lr_0[i0].queryIdx:
                for i1 in range(len(matches_lr1)):
                    if kp_ll_1 == matches_lr1[i1].queryIdx:
                        con.append((i0,i1))
                        matches.append((matches_lr_0[i0], matches_lr1[i1]))
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

def check_transform_agreed(T, matches_3d_l0, consensus_matches):

    real_l_0_pix_val,real_r_0_pix_val = extract_y_values(consensus_matches[:, 0], kp_l_0, kp_r_0)
    real_l_1_pix_val,real_r_1_pix_val = extract_y_values(consensus_matches[:, 1], kp_l_1, kp_r_1)
    ones = np.ones(( matches_3d_l0.shape[0],1))
    matches_in_4d = np.hstack((matches_3d_l0, ones)).T

    pts3d_to_l0 = (M1 @ matches_in_4d)
    pts3d_to_r0 = (M2 @ matches_in_4d)

    pix_values_l_0 = (K@pts3d_to_l0).T
    pix_values_l_0 = (pix_values_l_0 / pix_values_l_0[:,2].reshape(-1,1))[:,0:2]
    pix_values_l_0_y = pix_values_l_0[:,1]
    agrees_l0 = np.abs(real_l_0_pix_val - pix_values_l_0_y) < 2

    pix_values_r_0 = (K @ pts3d_to_r0).T
    pix_values_r_0 = (pix_values_r_0 / pix_values_r_0[:,2].reshape(-1,1))[:,0:2]
    pix_values_r_0_y = pix_values_r_0[:,1]
    agrees_r0 = np.abs(real_r_0_pix_val - pix_values_r_0_y) < 2

    to_3d = np.hstack((pts3d_to_l0.T, ones)).T
    pix_values_l_1 = (K @ T @ to_3d).T
    pix_values_l_1 = (pix_values_l_1 / pix_values_l_1[:,2].reshape(-1,1))[:,0:2]
    pix_values_l_1_y = pix_values_l_1[:,1]
    agrees_l1 = np.abs(real_l_1_pix_val - pix_values_l_1_y) < 2


    to_3d = np.hstack((pts3d_to_r0.T, ones)).T
    pix_values_r_1 = (K @ T @ to_3d).T
    pix_values_r_1 = (pix_values_r_1 / pix_values_r_1[:,2].reshape(-1,1))[:,0:2]
    pix_values_r_1_y = pix_values_r_1[:,1]
    agrees_r1 = np.abs(real_r_1_pix_val - pix_values_r_1_y) < 2

    agree_all = agrees_l0 & agrees_r0 & agrees_l1 & agrees_r1
    # points = np.where(agree_all)
    # return points
    return agree_all







def q3_4(relevant_idx,match,traingulated_pts ,T):
    """ check which points are consistent with the transformation T"""
    relevent_3d_pts = traingulated_pts[relevant_idx[:, 0]]
    agreed_matrix = check_transform_agreed(T, relevent_3d_pts, match)
    idx = np.where(agreed_matrix)
    agreed_points = relevent_3d_pts[idx]
    disagreed_points = relevent_3d_pts[~agreed_matrix]
    print(f"Number of points that agree with the transformation: {len(agreed_points)}")
    print(f"Number of points that disagree with the transformation: {len(disagreed_points)}")

    # print(point_cloud_3d)
    # print(f"Number of points that agree with the transformation: {len(point_cloud_3d[0])}")







def ransac_pnp(matches_idx, matches, traingulated_pts):
    """ Perform RANSAC to find the best transformation"""
    best_inliers = 0
    best_T = None
    best_matches_idx = None
    for i in range(RANSAC_ITERATIONS):
        random_idx = np.random.choice(matches_idx.shape[0], 4, replace=False)
        random_matches = matches[random_idx]
        random_traingulated_pts = traingulated_pts[random_idx]
        T = cv2.solvePnPRansac(random_traingulated_pts, random_matches, K, None, flags=cv2.SOLVEPNP_EPNP)
        inliers = check_transform_agreed(T, traingulated_pts, matches)
        if np.sum(inliers) > best_inliers:
            best_inliers = np.sum(inliers)
            best_T = T
            best_matches_idx = inliers
    return best_T, best_matches_idx




if __name__ == '__main__':
    img_l_0,img_r_0 = ex2.read_images(0)
    img_l_1,img_r_1 = ex2.read_images(1)
    kp_l_1,kp_r_1,matches_1 = ex2.extract_matches_from_images(img_l_1,img_r_1)
    kp_l_0, kp_r_0, matches_0 = ex2.extract_matches_from_images(img_l_0,img_r_0)

    triangulated_pts_0, triangulated_pts_1,inl_lr_0,inl_lr_1 = q3_1()
    matches_l = q3_2()
    T = q3_3()
    idx,match = choose_4_points_that_returns_matches(inl_lr_0, matches_l ,inl_lr_1)
    print(q3_4(idx,match,triangulated_pts_0,T))


