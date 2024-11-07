import gtsam
import numpy as np
import cv2
from final_project.algorithms.triangulation import triangulate_links
from final_project.backend.GTSam.gtsam_utils import get_inverse
from final_project.utils import rodriguez_to_mat
from final_project.Inputs import read_cameras

K, M1, M2 = read_cameras()
P, Q = K @ M1, K @ M2


def get_pixels_from_links(links):
    """
    get the pixel locations as pixels from links
    :param links: the links to get the pixels from
    """
    pixels_first = []
    pixels_second = []
    for link in links:
        pixels_first.append((link.x_left, link.y))
        pixels_second.append((link.x_right, link.y))
    return pixels_first, pixels_second


def transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
                             ordered_cur_left_pix_values, ordered_cur_right_pix_values, x_condition=True):
    points_4d = np.hstack((traingulated_pts, np.ones((traingulated_pts.shape[0], 1)))).T
    l1_4d_projected = ((K @ T @ np.vstack((M1, np.array([0, 0, 0, 1])))) @ points_4d)[:3, :].T
    r1_4d_projected = ((K @ T @ np.vstack((M2, np.array([0, 0, 0, 1])))) @ points_4d)[:3, :].T

    transformed_to_l1_points = l1_4d_projected / l1_4d_projected[:, 2][:, np.newaxis]
    real_y = ordered_cur_left_pix_values[:, 1]
    agree_l1 = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2
    agree_x = np.abs(transformed_to_l1_points[:, 0] - ordered_cur_left_pix_values[:, 0]) < 2
    agree_l1 = np.logical_and(agree_l1, agree_x)

    transformed_to_r1_points = r1_4d_projected / r1_4d_projected[:, 2][:, np.newaxis]
    real_y = ordered_cur_right_pix_values[:, 1]
    agree_r1 = np.abs(transformed_to_r1_points[:, 1] - real_y) < 2
    agree_x = np.abs(transformed_to_r1_points[:, 0] - ordered_cur_right_pix_values[:, 0]) < 2
    agree_r1 = np.logical_and(agree_r1, agree_x)

    return np.logical_and(agree_l1, agree_r1)


def calc_ransac_iteration(inliers_percent):
    """
    calculate the number of iterations needed to solve the RANSAC problem
    """
    suc_prob = 0.9999999999
    outliers_prob = 1 - (inliers_percent / 100) + 0.0000000001
    min_set_size = 4
    ransac_iterations = int(np.log(1 - suc_prob) / np.log(1 - np.power(1 - outliers_prob, min_set_size))) + 1
    return ransac_iterations


def ransac_pnp_for_tracking_db(matches_l_l, prev_links, cur_links, inliers_percent):
    """ Perform RANSAC to find the best transformation"""
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

    diff_coeff = np.zeros((5, 1))
    best_inliers = 0
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

        points_agreed = transformation_agreement(T, points_3d, prev_left_pix_values, prev_right_pix_values,
                                                 ordered_cur_left_pix_values, ordered_cur_right_pix_values,
                                                 x_condition=False)

        num_inliers = np.sum(points_agreed)
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_matches_idx = np.where(points_agreed == True)[0]
    return best_matches_idx


def ransac_pnp(matches_l_l, prev_links, cur_links):
    """
    Perform RANSAC to find the best pose transformation between two frames.

    :param matches_l_l: List of feature matches between frames.
    :param prev_links: Feature links from the previous frame.
    :param cur_links: Feature links from the current frame.
    :return: Relative pose, indices of best matches, and number of inliers.
    """
    #TODO: CHANGE THE RANSAC ITERETAION
    ransac_iterations = 10000  # Number of RANSAC iterations

    filtered_links_cur = []
    filtered_links_prev = []

    # Filter the links based on matches
    for match in matches_l_l:
        link_index = match.trainIdx
        filtered_links_cur.append(cur_links[link_index])

        link_index = match.queryIdx
        filtered_links_prev.append(prev_links[link_index])

    # Triangulate 3D points from previous links
    points_3d = triangulate_links(filtered_links_prev, P, Q)

    # Extract pixel coordinates from links
    prev_left_pix_values, prev_right_pix_values = get_pixels_from_links(filtered_links_prev)
    ordered_cur_left_pix_values, ordered_cur_right_pix_values = get_pixels_from_links(filtered_links_cur)

    # Convert to NumPy arrays for processing
    prev_left_pix_values = np.array(prev_left_pix_values)
    prev_right_pix_values = np.array(prev_right_pix_values)
    ordered_cur_left_pix_values = np.array(ordered_cur_left_pix_values)
    ordered_cur_right_pix_values = np.array(ordered_cur_right_pix_values)

    diff_coeff = np.zeros((5, 1))  # Distortion coefficients (assumed zero)
    best_inliers = 0
    best_T = None
    best_matches_idx = []

    # RANSAC loop to find the best transformation
    for i in range(ransac_iterations):
        # Randomly select 4 points for PnP
        random_idx = np.random.choice(len(points_3d), 4, replace=False)
        random_world_points = points_3d[random_idx]
        random_cur_l_pixels = ordered_cur_left_pix_values[random_idx]

        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            random_world_points, random_cur_l_pixels, K,
            distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP
        )

        if success:
            # Convert rotation vector to rotation matrix and form transformation matrix
            T = rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue

        # Check agreement of transformation with all points
        points_agreed = transformation_agreement(
            T, points_3d, prev_left_pix_values, prev_right_pix_values,
            ordered_cur_left_pix_values, ordered_cur_right_pix_values,
            x_condition=False
        )

        # Count the number of inliers
        num_inliers = np.sum(points_agreed)
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_T = T
            best_matches_idx = np.where(points_agreed == True)[0]

    # Recompute transformation using all inliers
    world_points = points_3d[best_matches_idx]
    pixels = ordered_cur_left_pix_values[best_matches_idx]
    if len(best_matches_idx) < 4:
        return None, [], []

    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points, pixels, K,
        distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP
    )

    if success:
        # Convert to transformation matrix and compute relative pose
        T = rodriguez_to_mat(rotation_vector, translation_vector)
        inv_t = get_inverse(T)
        relative_pose = gtsam.Pose3(gtsam.Rot3(inv_t[:3, :3]), gtsam.Point3(inv_t[:3, 3]))
        return relative_pose, best_matches_idx, best_inliers
    else:
        return None, None, None
