import math
import pickle
from tqdm import tqdm

import gtsam
import matplotlib.pyplot as plt
from gtsam.utils.plot import plot_trajectory

# from VAN_ex.code import arguments
# from VAN_ex.code.ex5 import K_OBJECT
# from VAN_ex.code.graph import Graph
# from VAN_ex.code.tracking_database import TrackingDB
# from ex6 import load
import numpy as np
import cv2
# from ex3 import read_extrinsic_matrices
# from ex4_v2 import rodriguez_to_mat, transformation_agreement, K
# from ex5 import get_inverse
# from ex1 import read_images
# from ex6 import save

# Initialize the BFMatcher with Hamming distance and no cross-checking
from final_project import arguments
from final_project.Inputs import read_extrinsic_matrices, read_images
from final_project.algorithms.ransac import get_pixels_from_links, transformation_agreement
from final_project.algorithms.triangulation import triangulate_links
from final_project.arguments import SIFT_DB_PATH
from final_project.backend.GTSam import bundle
from final_project.backend.GTSam.bundle import K_OBJECT, optimize_graph
from final_project.backend.GTSam.gtsam_utils import get_inverse, save
from final_project.backend.GTSam.pose_graph import PoseGraph, calculate_relative_pose_cov
from final_project.backend.database.tracking_database import TrackingDB
from final_project.backend.loop.graph import Graph
from final_project.utils import P, Q, K, rodriguez_to_mat
from final_project.algorithms.matching import MATCHER, MATCHER_LEFT_RIGHT


# MATCHER = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)

# Threshold constants for Mahalanobis distance and inliers
MAHALANOBIS_THRESHOLD = 2700
INLIERS_THRESHOLD = 130
FAR_MAHALANOBIS_THRESHOLD = MAHALANOBIS_THRESHOLD * 7

# Symbols used for locations and cameras in the pose graph
LOCATION_SYMBOL = 'l'
CAMERA_SYMBOL = 'c'

# Dictionary to store relative covariances and initialize a Dijkstra graph for covariance paths
relative_covariance_dict = dict()
cov_dijkstra_graph = Graph()

# Path to the serialized tracking database
PATH_TO_DB = serialized_path = arguments.DATA_HEAD + "/docs/AKAZE/db/db_3359"

# Initial pose (identity rotation and zero translation)
P0 = gtsam.Pose3(gtsam.Rot3(np.eye(3)), gtsam.Point3(np.zeros(3)))


def get_tracking_database(path_do_db_file=PATH_TO_DB):
    """
    Load the tracking database from a serialized file.

    :param path_do_db_file: Path to the serialized tracking database file.
    :return: Loaded TrackingDB object.
    """
    tracking_db = TrackingDB()
    tracking_db.load(path_do_db_file)
    return tracking_db


# Load the tracking database
db = get_tracking_database(PATH_TO_DB)


# from ex4_v2 import calc_ransac_iteration, triangulate_links, get_pixels_from_links, P, Q
# from ex6 import calculate_relative_pose_cov


def update_pose_graph(pose_grpah, result, relative_pose, relative_cov, start_frame, end_frame):
    """
    Add a relative pose between two frames to the pose graph and optimize the graph.

    :param pose_grpah: The current pose graph.
    :param result: The current optimized pose estimates.
    :param relative_pose: The relative pose between start_frame and end_frame.
    :param relative_cov: The covariance associated with the relative pose.
    :param start_frame: The index of the start frame.
    :param end_frame: The index of the end frame.
    :return: Updated pose graph and optimized result.
    """
    # Define a Gaussian noise model based on the relative covariance
    noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov)

    # Create symbols for the start and end frames
    start_frame_symbol = gtsam.symbol(CAMERA_SYMBOL, start_frame)
    end_frame_symbol = gtsam.symbol(CAMERA_SYMBOL, end_frame)

    # Create a BetweenFactorPose3 and add it to the pose graph
    fac = gtsam.BetweenFactorPose3(start_frame_symbol, end_frame_symbol, relative_pose, noise_model)
    pose_grpah.add(fac)

    # Calculate the global pose for the end frame based on the relative pose
    first_global_pose = result.atPose3(start_frame_symbol)
    global_pose = first_global_pose.transformPoseFrom(relative_pose)

    # Update the result with the new global pose
    result.erase(end_frame_symbol)
    result.insert(end_frame_symbol, global_pose)

    # Optimize the pose graph using Levenberg-Marquardt optimizer
    optimizer = gtsam.LevenbergMarquardtOptimizer(pose_grpah, result)
    new_result = optimizer.optimize()

    # Add the edge to the Dijkstra graph for covariance paths
    cov_dijkstra_graph.add_edge(start_frame, end_frame, relative_cov)

    return pose_grpah, new_result


def get_relative_consecutive_covariance(c1, c2, marginals):
    """
    Retrieve the relative covariance between two consecutive camera frames.

    :param c1: Symbol of the first camera frame.
    :param c2: Symbol of the second camera frame.
    :param marginals: Marginals object containing covariance information.
    :return: Relative covariance matrix.
    """
    if c2 == c1:
        # If both symbols are the same, return the marginal covariance of c1
        keys = gtsam.KeyVector()
        keys.append(c1)
        marginal_information = marginals.jointMarginalInformation(keys)
        inf_c2_giving_c1 = marginal_information.at(c1, c1)
        cov_c2_giving_c1 = np.linalg.inv(inf_c2_giving_c1)
        return cov_c2_giving_c1

    # Extract the numerical part of the symbol
    c2_number = int(gtsam.DefaultKeyFormatter(c2)[1:])

    # Check if the covariance is already computed
    if c2_number in relative_covariance_dict:
        return relative_covariance_dict[c2_number]

    # Compute the relative covariance using joint marginal information
    keys = gtsam.KeyVector()
    keys.append(c1)
    keys.append(c2)
    marginal_information = marginals.jointMarginalInformation(keys)
    inf_c2_giving_c1 = marginal_information.at(c2, c2)
    cov_c2_giving_c1 = np.linalg.inv(inf_c2_giving_c1)
    return cov_c2_giving_c1


def get_relative_covariance(index_list, marginals):
    """
    Compute the cumulative relative covariance along a path of camera frames.

    :param index_list: List of camera frame indices representing the path.
    :param marginals: Marginals object containing covariance information.
    :return: Cumulative covariance matrix.
    """
    cov_sum = None
    if len(index_list) == 1:
        # Single camera frame: return its marginal covariance
        c1 = gtsam.symbol('c', index_list[0])
        c_n_giving_c_i = get_relative_consecutive_covariance(c1, c1, marginals)
        return c_n_giving_c_i

    # Iterate through consecutive pairs to accumulate covariance
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
    """
    Placeholder function to compute relative pose. (Function not implemented)

    :param pose: Pose object.
    """
    pass


def calculate_mahalanobis_distance(c_n, c_i, result, relative_information):
    """
    Calculate the Mahalanobis distance between two camera poses.

    :param c_n: Symbol of the first camera frame.
    :param c_i: Symbol of the second camera frame.
    :param result: Current optimized pose estimates.
    :param relative_information: Inverse of the relative covariance matrix.
    :return: Mahalanobis distance.
    """
    # Retrieve poses from the result
    pose_c_n = result.atPose3(c_n)
    pose_c_i = result.atPose3(c_i)

    # Compute the relative pose between c_n and c_i
    relative_pose = pose_c_n.between(pose_c_i)

    # Concatenate rotation (yaw, pitch, roll) and translation into a single delta vector
    delta = np.hstack([relative_pose.rotation().ypr(), relative_pose.translation()])
    # delta = np.hstack([relative_pose.rotation().xyz(), relative_pose.translation()])

    # Calculate Mahalanobis distance using the relative information matrix
    mahalanobis_distance = delta @ relative_information @ delta
    return math.sqrt(mahalanobis_distance)


def get_symbol(index):
    """
    Generate a GTSAM symbol for a given camera index.

    :param index: Camera frame index.
    :return: GTSAM symbol.
    """
    return gtsam.symbol('c', index)


def check_candidate(c_n_idx, c_i_idx, marginals, result, index_list):
    """
    Check if a candidate camera frame is a good loop closure candidate based on Mahalanobis distance.

    :param c_n_idx: Index of the current camera frame.
    :param c_i_idx: Index of the candidate camera frame.
    :param marginals: Marginals object containing covariance information.
    :param result: Current optimized pose estimates.
    :param index_list: List of camera frame indices.
    :return: Mahalanobis distance between the two camera frames.
    """
    # Get the shortest path between the two camera frames in the covariance graph
    cur_index_list = cov_dijkstra_graph.get_shortest_path(index_list[c_i_idx], index_list[c_n_idx])

    # Compute the cumulative relative covariance along the path
    c_n_giving_c_i = get_relative_covariance(cur_index_list, marginals)

    # Compute the relative information matrix
    relative_information = np.linalg.inv(c_n_giving_c_i)

    # Generate symbols for the camera frames
    symbol_cn = get_symbol(index_list[c_n_idx])
    symbol_ci = get_symbol(index_list[c_i_idx])

    # Calculate the Mahalanobis distance between the two camera poses
    mahalanobis_distance = calculate_mahalanobis_distance(symbol_cn, symbol_ci, result, relative_information)

    return mahalanobis_distance


KEY_FRAME_GAP = 10  # Minimum gap between key frames to consider for loop closure


def get_good_candidates(c_n_index, marginals, result, index_list):
    """
    Identify good loop closure candidates for a given camera frame based on Mahalanobis distance.

    :param c_n_index: Index of the current camera frame.
    :param marginals: Marginals object containing covariance information.
    :param result: Current optimized pose estimates.
    :param index_list: List of camera frame indices.
    :return: List of candidate camera frame indices that are good loop closures.
    """
    candidates = []
    last_index_to_check = c_n_index - KEY_FRAME_GAP

    # Iterate through potential candidates ensuring a minimum frame gap
    for c_i_index in range(0, last_index_to_check, 1):
        mahalanobis_distance = check_candidate(c_n_index, c_i_index, marginals, result, index_list)
        # print(f'Mahalanobis distance between {c_n_index} and {c_i_index}: {mahalanobis_distance}')
        if mahalanobis_distance < MAHALANOBIS_THRESHOLD:
            candidates.append(index_list[c_i_index])
            print(
                f'added candidate {index_list[c_i_index]} with distance {mahalanobis_distance} to {index_list[c_n_index]}')
        elif mahalanobis_distance > FAR_MAHALANOBIS_THRESHOLD:
            # Skip further candidates if distance is too large
            c_i_index += 2

    return candidates


def get_path(c_n, c_i, result):
    """
    Generate a simple path list between two camera indices.

    :param c_n: End camera index.
    :param c_i: Start camera index.
    :param result: Current optimized pose estimates.
    :return: List of indices from c_i to c_n.
    """
    return [index for index in range(c_i, c_n + 1)]


def load(base_filename):
    """
    Load serialized data from a pickle file.

    :param base_filename: Base filename without extension.
    :return: Loaded data object.
    """
    filename = base_filename + '.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print('Bundles loaded from', filename)
    return data


def get_index_list(result):
    """
    Extract and sort camera frame indices from the result.

    :param result: Current optimized pose estimates.
    :return: Sorted list of camera frame indices.
    """
    index_list = []
    for key in result.keys():
        cam_number = int(gtsam.DefaultKeyFormatter(key)[1:])
        index_list.append(cam_number)
    index_list.sort()
    return index_list


def init_dijksra_graph_relative_covariance_dict(result_without_closure, pose_graph_without_closure,
                                                cov_dict, dijkstra_graph):
    """
    Initialize the Dijkstra graph and relative covariance dictionary based on the pose graph without loop closures.

    :param result_without_closure: Pose estimates without loop closures.
    :param pose_graph_without_closure: Pose graph without loop closures.
    :param cov_dict: Dictionary to store relative covariances.
    :param dijkstra_graph: Dijkstra graph to store covariance paths.
    :return: Updated covariance dictionary and Dijkstra graph.
    """
    # Compute marginals for the current pose graph
    marginals = gtsam.Marginals(pose_graph_without_closure, result_without_closure)
    index_list = get_index_list(result_without_closure)

    # Iterate through consecutive camera frames to populate covariance dictionary and graph
    for i in range(len(index_list) - 1):
        c1 = gtsam.symbol('c', index_list[i])
        c2 = gtsam.symbol('c', index_list[i + 1])
        c_n_giving_c_i = get_relative_consecutive_covariance(c1, c2, marginals)
        cov_dict[index_list[i + 1]] = c_n_giving_c_i
        dijkstra_graph.add_edge(index_list[i], index_list[i + 1], c_n_giving_c_i)

    # Add the marginal covariance for the first camera frame
    cov_dict[index_list[0]] = marginals.marginalCovariance(gtsam.symbol('c', index_list[0]))
    return cov_dict, dijkstra_graph


def q_7_5_6(marginals, result_without_closure, pose_graph_without_closure):
    """
    Plot graphs of location uncertainty sizes for the pose graph with and without loop closures.

    :param marginals: Marginals object containing covariance information.
    :param result_without_closure: Pose estimates without loop closures.
    :param pose_graph_without_closure: Pose graph without loop closures.
    """
    # Initialize covariance dictionary and Dijkstra graph for the pose graph without loop closures
    relative_covariance_dict_without_closure = dict()
    dijkstra_graph_without_closure = Graph()
    marginals_without_closure = gtsam.Marginals(pose_graph_without_closure, result_without_closure)
    rel_cov_before_loop_c, dijkstra_graph = init_dijksra_graph_relative_covariance_dict(
        result_without_closure,
        pose_graph_without_closure,
        relative_covariance_dict_without_closure,
        dijkstra_graph_without_closure
    )

    index_list = get_index_list(result_without_closure)
    uncertainties_before = []

    # Calculate determinant of cumulative covariance for each camera frame without loop closures
    for c_n in index_list[:]:
        cur_index_list = dijkstra_graph.get_shortest_path(index_list[0], c_n)
        rel_cov = get_relative_covariance(cur_index_list, marginals_without_closure)
        det = np.linalg.det(rel_cov)
        uncertainties_before.append(det)

    # Plot uncertainties without loop closures
    plt.figure()
    plt.plot(index_list, uncertainties_before, label='Uncertainty without loop closures')

    uncertainties = []
    for c_n in index_list:
        # Calculate determinant of cumulative covariance with loop closures
        cur_index_list = cov_dijkstra_graph.get_shortest_path(index_list[0], c_n)
        det = np.linalg.det(get_relative_covariance(cur_index_list, marginals=marginals))
        uncertainties.append(det)

    # Plot uncertainties with loop closures
    plt.figure()
    plt.plot(index_list, uncertainties, label='Uncertainty with loop closures')
    plt.show()


# def find_loops(pose_graph: PoseGraph):
def find_loops(data,db):
    """
    Detect and insert loop closures into the pose graph.

    :param data: Tuple containing pose graph and result.
    :return: Updated pose graph and result with loop closures.
    """
    # pg = pose_graph.graph
    # result = pose_graph.result
    pg = data[0]
    result = data[1]
    if result is None:
        raise ValueError('No result found in the pose graph.')
    marginals = gtsam.Marginals(pg, result)
    index_list = get_index_list(result)

    # Initialize the Dijkstra graph and covariance dictionary
    init_dijksra_graph_relative_covariance_dict(result, pg, relative_covariance_dict, cov_dijkstra_graph)

    familiar_path = False
    frames_in_familiar_path = []

    # Iterate through each camera frame to find loop closures
    for c_n_index in tqdm(range(len(index_list))):
        good_candidates = get_good_candidates(c_n_index, marginals, result, index_list)
        if len(good_candidates) > 0:
            camera_number = index_list[c_n_index]
            if familiar_path:
                # If already in a familiar path, append candidates for later processing
                frames_in_familiar_path.append((camera_number, good_candidates))
                continue

            # Find the best candidate with sufficient inliers
            best_candidate, matches = consensus_matches(camera_number, good_candidates, db)
            if best_candidate is not None:
                familiar_path = True
                # Insert the loop closure into the pose graph
                pg, result = insert_to_pose_graph(camera_number, best_candidate,
                                                  matches, pg, result)
                print(f"loop detected between camera {camera_number} and camera {best_candidate}")
        else:
            if len(frames_in_familiar_path) > 0:
                # Process the accumulated familiar path frames in reverse order
                for camera_number, candidates in frames_in_familiar_path[::-1]:
                    best_candidate, matches = consensus_matches(camera_number, candidates, db)
                    if best_candidate is not None:
                        # Insert the loop closure into the pose graph
                        pg, result = insert_to_pose_graph(camera_number, best_candidate,
                                                          matches, pg, result)
                        print(
                            f"end of familiar segment, loop closure between camera {camera_number} "
                            f"and camera {best_candidate}"
                        )
                        break
                familiar_path = False
                frames_in_familiar_path = []
    # pose_graph.graph = pg
    # pose_graph.result = result
    # return pose_graph
    return pg, result


def plot_graph_along(camera_number, pose_graph, result):
    """
    Plot the trajectory of the pose graph after a loop closure.

    :param camera_number: Camera frame index where the closure was added.
    :param pose_graph: Current pose graph.
    :param result: Current optimized pose estimates.
    """
    marginals = gtsam.Marginals(pose_graph, result)
    plot_trajectory(camera_number, result, marginals=marginals,
                    title=f"graph plotted along the process after closure on"
                          f" camera {camera_number}, Q_7_5_4")


def insert_to_pose_graph(camera_number, best_candidate, matches, pose_graph, result):
    """
    Insert a loop closure between two camera frames into the pose graph.

    :param camera_number: Index of the current camera frame.
    :param best_candidate: Index of the best candidate camera frame for loop closure.
    :param matches: Feature matches between the two frames.
    :param pose_graph: Current pose graph.
    :param result: Current optimized pose estimates.
    :return: Updated pose graph and result.
    """
    # Compute relative pose and covariance between the two frames
    # try:
    rel_pose_new, rel_cov_new = get_relative_pose_and_cov(camera_number, best_candidate, pose_graph, result, matches, db)

    index_list = get_index_list(result)
    old_camera_index = index_list.index(camera_number)
    camera_before = index_list[old_camera_index - 1]
    rel_pose_old, rel_cov_old = get_relative_pose_and_cov(camera_before, camera_number, pose_graph, result, matches, db)


    # except Exception as e:
    #     print(f"bad point!!!!!!!!!!!!!!!! {result.atPoint3(7782220156096217146)}")
    #     return pose_graph, result


    # Add the relative pose to the pose graph and optimize
    pose_graph, result = update_pose_graph(pose_graph, result, rel_pose_new, rel_cov_new, camera_number, best_candidate)

    # Plot the updated graph
    # plot_graph_along(camera_number, pose_graph, result)

    return pose_graph, result


# Read extrinsic matrices to determine ground truth camera locations
mat = read_extrinsic_matrices()[:]
cameras_locations2 = []
for cam in mat:
    rot = cam[:3, :3]
    t = cam[:3, 3]
    cameras_locations2.append(-rot.T @ t)


def ransac_pnp(matches_l_l, prev_links, cur_links):
    """
    Perform RANSAC to find the best pose transformation between two frames.

    :param matches_l_l: List of feature matches between frames.
    :param prev_links: Feature links from the previous frame.
    :param cur_links: Feature links from the current frame.
    :return: Relative pose, indices of best matches, and number of inliers.
    """
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


def check_candidate_match(reference_key_frame, candiate_keyframe, db: TrackingDB):
    """
    Check and retrieve inlier matches between two candidate camera frames.

    :param reference_key_frame: Index of the reference camera frame.
    :param candiate_keyframe: Index of the candidate camera frame.
    :param db: Tracking database containing features and links.
    :return: List of inlier matches.
    """
    # Retrieve features and links from both frames
    keyframe1_features = db.features(reference_key_frame)
    keyframe1_links = db.all_frame_links(reference_key_frame)

    keyframe2_features = db.features(candiate_keyframe)
    keyframe2_links = db.all_frame_links(candiate_keyframe)

    # Match features between the two frames
    matches_l_l = MATCHER.match(keyframe1_features, keyframe2_features)

    # Perform RANSAC-based PnP to find inlier matches
    T, best_matches_idx, best_inliers = ransac_pnp(matches_l_l, keyframe1_links, keyframe2_links)

    # Return only the inlier matches
    try:
        percentage_inliers = best_inliers / len(matches_l_l)
    except Exception as e:
        # print(best_inliers)
        # print(matches_l_l)
        percentage_inliers = 0

    # print(f"Percentage of inliers: {percentage_inliers}")
    return [matches_l_l[i] for i in best_matches_idx], percentage_inliers


def create_bundle(first_frame_idx, second_frame_idx, bundle_graph, result, inliers, db: TrackingDB):
    """
    Create a factor graph (bundle) for two camera frames with inlier matches.

    :param first_frame_idx: Index of the first camera frame.
    :param second_frame_idx: Index of the second camera frame.
    :param bundle_graph: Existing bundle graph to which new factors will be added.
    :param result: Current optimized pose estimates.
    :param inliers: Inlier matches between the two frames.
    :param db: Tracking database containing features and links.
    :return: New factor graph and initial estimates for optimization.
    """
    # Create symbols for the camera frames
    first_frame_symbol = gtsam.symbol(CAMERA_SYMBOL, first_frame_idx)
    second_frame_symbol = gtsam.symbol(CAMERA_SYMBOL, second_frame_idx)

    # Initialize a new nonlinear factor graph and initial estimates
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Define a prior for the first camera frame (identity pose)
    first_pose = gtsam.Pose3()
    first_pose_sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
    graph.add(gtsam.PriorFactorPose3(first_frame_symbol, first_pose, first_pose_sigma))
    initial_estimate.insert(first_frame_symbol, gtsam.Pose3())

    # Calculate the relative pose and covariance between the two frames
    marginals, relative_pose_second, relative_cov_second = (
        calculate_relative_pose_cov(first_frame_symbol, second_frame_symbol, bundle_graph, result)
    )

    # Insert the relative pose into the initial estimates
    initial_estimate.insert(second_frame_symbol, relative_pose_second)

    # Add the relative pose factor to the graph
    noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov_second)
    factor = gtsam.BetweenFactorPose3(first_frame_symbol, second_frame_symbol, relative_pose_second, noise_model)
    graph.add(factor)

    # Create a stereo camera model for the first frame
    gtsam_frame_1 = gtsam.StereoCamera(first_pose, K_OBJECT)

    # Retrieve feature links for both frames
    first_frame_links = db.all_frame_links(first_frame_idx)
    second_frame_links = db.all_frame_links(second_frame_idx)

    # Add stereo factors for each inlier match
    for match in inliers:
        # Create a symbol for the 3D location of the matched feature
        location_symbol = gtsam.symbol(LOCATION_SYMBOL, match.queryIdx)
        first_link = first_frame_links[match.queryIdx]

        # Define an isotropic noise model for the stereo measurements
        sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

        # Add a stereo factor for the first frame
        graph.add(gtsam.GenericStereoFactor3D(
            gtsam.StereoPoint2(first_link.x_left, first_link.x_right, first_link.y),
            sigma, first_frame_symbol, location_symbol, K_OBJECT
        ))

        # Triangulate the 3D point from the first frame's stereo measurements
        reference_triangulated_point = gtsam_frame_1.backproject(
            gtsam.StereoPoint2(first_link.x_left, first_link.x_right, first_link.y)
        )
        assert reference_triangulated_point[2] > 0  # Ensure point is in front of the camera

        # Insert the triangulated 3D point into the initial estimates
        initial_estimate.insert(location_symbol, reference_triangulated_point)

        # Add a stereo factor for the second frame
        second_link = second_frame_links[match.trainIdx]
        graph.add(gtsam.GenericStereoFactor3D(
            gtsam.StereoPoint2(second_link.x_left, second_link.x_right, second_link.y),
            sigma, second_frame_symbol, location_symbol, K_OBJECT
        ))

    return graph, initial_estimate


def get_relative_pose_and_cov(first_frame_idx, second_frame_idx, bundle_graph, result, inliers, db: TrackingDB):
    """
    Optimize the pose graph by adding a new factor between two frames.

    :param first_frame_idx: Index of the first camera frame.
    :param second_frame_idx: Index of the second camera frame.
    :param bundle_graph: Existing bundle graph.
    :param result: Current optimized pose estimates.
    :param inliers: Inlier matches between the two frames.
    :param db: Tracking database containing features and links.
    :return: Relative pose and covariance between the two frames.
    """
    # Create a new bundle graph and initial estimates for the two frames
    graph, initial_estimate = create_bundle(first_frame_idx, second_frame_idx, bundle_graph, result, inliers, db)

    graph, new_res = optimize_graph(graph, initial_estimate)

    # Optimize the factor graph using Levenberg-Marquardt optimizer
    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    # result = optimizer.optimize()

    # Generate symbols for the camera frames
    first_frame_symbol = gtsam.symbol(CAMERA_SYMBOL, first_frame_idx)
    second_frame_symbol = gtsam.symbol(CAMERA_SYMBOL, second_frame_idx)

    # Extract the optimized poses for both frames
    first_pose = new_res.atPose3(first_frame_symbol)
    second_pose = new_res.atPose3(second_frame_symbol)

    # Compute the relative pose between the two frames
    relative_pose = first_pose.between(second_pose)

    # Define the keys for marginals computation
    keys = gtsam.KeyVector()
    keys.append(first_frame_symbol)
    keys.append(second_frame_symbol)

    # Compute marginals to obtain covariance information
    marginals = gtsam.Marginals(graph, new_res)
    information = marginals.jointMarginalInformation(keys).fullMatrix()

    # Extract the relative covariance matrix from the information matrix
    relative_cov = np.linalg.inv(information[-6:, -6:])

    return relative_pose, relative_cov


def consensus_matches(reference_key_frame, candidates_index_lst, data_base: TrackingDB):
    """
    Find the best candidate match among a list of candidates based on inlier matches.

    :param reference_key_frame: Index of the reference camera frame.
    :param candidates_index_lst: List of candidate camera frame indices.
    :param data_base: Tracking database containing features and links.
    :return: Best candidate index and corresponding inlier matches.
    """
    best_candidate = None
    best_matches = []

    # Iterate through candidates to find one with sufficient inliers
    for candidate in candidates_index_lst:
        matches, percentage = check_candidate_match(reference_key_frame, candidate, data_base)
        if len(matches) > INLIERS_THRESHOLD:
            print(f"cand: {candidate}, matches: {len(matches)} inliers percentage: {percentage}")
            best_candidate = candidate
            best_matches = matches
            return best_candidate, best_matches

    # If no candidate meets the inliers threshold, reset best_candidate
    if len(best_matches) > 0:
        if len(best_matches) < INLIERS_THRESHOLD:
            best_candidate = None
            best_matches = []
    return best_candidate, best_matches


def get_camera_location_from_gtsam(pose: gtsam.Pose3):
    """
    Extract the translation component (location) from a GTSAM Pose3 object.

    :param pose: GTSAM Pose3 object.
    :return: Translation vector representing the camera location.
    """
    return pose.translation()


def plot_matches(first_frame, second_frame):
    """
    Visualize inlier and outlier feature matches between two camera frames.

    :param first_frame: Index of the first camera frame.
    :param second_frame: Index of the second camera frame.
    """
    # Retrieve inlier matches
    inliers,_ = check_candidate_match(first_frame, second_frame, db)
    query_idx_list = [match.queryIdx for match in inliers]

    # Retrieve all matches and identify outliers
    all_matches = MATCHER.match(db.features(first_frame), db.features(second_frame))
    outliers = []
    for match in all_matches:
        if match.queryIdx not in query_idx_list:
            outliers.append(match)

    # Extract matched points for visualization
    inliers_points_first = [db.all_frame_links(first_frame)[match.queryIdx] for match in inliers]
    outliers_points_first = [db.all_frame_links(first_frame)[match.queryIdx] for match in outliers]

    inliers_points_second = [db.all_frame_links(second_frame)[match.trainIdx] for match in inliers]
    outliers_points_second = [db.all_frame_links(second_frame)[match.trainIdx] for match in outliers]

    # Read the corresponding images
    img1 = read_images(first_frame)[0]
    img2 = read_images(second_frame)[0]

    # Plot inliers and outliers on the first image
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter(
        [point.x_left for point in inliers_points_first],
        [point.y for point in inliers_points_first],
        c='r', s=2
    )
    plt.scatter(
        [point.x_left for point in outliers_points_first],
        [point.y for point in outliers_points_first],
        c='b', s=2
    )
    plt.title(f"first frame: {min(first_frame, second_frame)} inliers in red outliers in blue")

    # Plot inliers and outliers on the second image
    plt.subplot(2, 1, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter(
        [point.x_left for point in inliers_points_second],
        [point.y for point in inliers_points_second],
        c='r', s=2
    )
    plt.scatter(
        [point.x_left for point in outliers_points_second],
        [point.y for point in outliers_points_second],
        c='b', s=2
    )
    plt.title(f"second frame: {max(first_frame, second_frame)} inliers in red outliers in blue")

    plt.show()


def get_locations_from_gtsam(result):
    """
    Extract camera locations from the GTSAM result.

    :param result: Current optimized pose estimates.
    :return: List of camera locations.
    """
    locations = []
    index_list = get_index_list(result)
    for index in index_list:
        pose = result.atPose3(gtsam.symbol(CAMERA_SYMBOL, index))
        location = get_camera_location_from_gtsam(pose)
        # Verify that the location matches the pose's translation
        assert location[0] == pose.x()
        assert location[1] == pose.y()
        assert location[2] == pose.z()
        locations.append(location)
    return locations


def get_locations_ground_truths():
    """
    Retrieve ground truth camera locations from extrinsic matrices.

    :return: List of ground truth camera locations.
    """
    all_transformations = read_extrinsic_matrices()
    cameras_locations2 = []
    for cam in all_transformations:
        rot = cam[:3, :3]
        t = cam[:3, 3]
        cameras_locations2.append(-rot.T @ t)
    return cameras_locations2


def plot_trajectory2D_ground_truth(result, str):
    """
    Plot the estimated and ground truth 2D trajectories.

    :param result: Current optimized pose estimates.
    :param str: Description string for the plot title.
    """
    cameras_locations2 = get_locations_ground_truths()
    locations = get_locations_from_gtsam(result)
    plt.figure()

    # Plot estimated trajectory in red
    plt.plot(
        [location[0] for location in locations],
        [location[2] for location in locations],
        'r-',
        label='estimated trajectory'
    )

    # Plot ground truth trajectory in blue
    plt.plot(
        [location[0] for location in cameras_locations2],
        [location[2] for location in cameras_locations2],
        'b-',
        label='ground truth trajectory'
    )

    plt.title(f"q7_5_3 graph {str} red: estimated trajectory, blue: ground truth trajectory")
    plt.show()


def q_7_5_5(result, result_without_closure):
    """
    Plot graphs of absolute location errors with and without loop closures.

    :param result: Pose estimates with loop closures.
    :param result_without_closure: Pose estimates without loop closures.
    """
    # Extract locations from both results
    locations = get_locations_from_gtsam(result)
    locations_without_closure = get_locations_from_gtsam(result_without_closure)
    ground_truth_locations = get_locations_ground_truths()

    # Plot location errors with loop closure
    plt.figure()
    plt.plot(
        [np.linalg.norm(locations[i] - ground_truth_locations[i]) for i in range(len(locations))],
        color='r', label="L2 norm"
    )
    plt.plot(
        [np.linalg.norm(locations[i][0] - ground_truth_locations[i][0]) for i in range(len(locations))],
        color='b', label="x error"
    )
    plt.plot(
        [np.linalg.norm(locations[i][1] - ground_truth_locations[i][1]) for i in range(len(locations))],
        color='g', label="y error"
    )
    plt.plot(
        [np.linalg.norm(locations[i][2] - ground_truth_locations[i][2]) for i in range(len(locations))],
        color='y', label="z error"
    )

    plt.legend()
    plt.title("location error with loop closure")

    # Plot location errors without loop closure
    plt.figure()
    plt.plot(
        [np.linalg.norm(locations_without_closure[i] - ground_truth_locations[i]) for i in
         range(len(locations_without_closure))],
        color='r', label="L2 norm"
    )
    plt.plot(
        [np.linalg.norm(locations_without_closure[i][0] - ground_truth_locations[i][0]) for i in
         range(len(locations_without_closure))],
        color='b', label="x error"
    )
    plt.plot(
        [np.linalg.norm(locations_without_closure[i][1] - ground_truth_locations[i][1]) for i in
         range(len(locations_without_closure))],
        color='g', label="y error"
    )
    plt.plot(
        [np.linalg.norm(locations_without_closure[i][2] - ground_truth_locations[i][2]) for i in
         range(len(locations_without_closure))],
        color='y', label="z error"
    )

    plt.legend()
    plt.title("location error without loop closure")


def plot_pose_graphs_XZ_ground_truth(results_with_closure, results_without_closure, pose_graph_closure,
                                     pose_graph_no_closure):
    """
    Plot the XZ plane of pose graphs with and without loop closures alongside ground truth.

    :param results_with_closure: Pose estimates with loop closures.
    :param results_without_closure: Pose estimates without loop closures.
    :param pose_graph_closure: Pose graph with loop closures.
    :param pose_graph_no_closure: Pose graph without loop closures.
    """
    plt.figure()

    # Extract camera indices and ground truth locations
    cameras_idx = get_index_list(results_with_closure)
    ground_truth_locations = get_locations_ground_truths()
    truth_x = [ground_truth_locations[i][0] for i in cameras_idx]
    truth_z = [ground_truth_locations[i][2] for i in cameras_idx]
    truth_y = [ground_truth_locations[i][1] for i in cameras_idx]

    # Plot ground truth trajectory in green
    plt.plot(truth_x, truth_z, 'g.', label="ground truth", markersize=0.9)

    # Plot estimated poses with loop closures in red
    poses = gtsam.utilities.allPose3s(results_with_closure)
    marginals_closure = gtsam.Marginals(pose_graph_closure, results_with_closure)
    x_list_closure = []
    z_list_closure = []
    y_list_closure = []
    cov_closure = []
    location_uncertainty_closure = []
    for key in poses.keys():
        pose = poses.atPose3(key)
        x_list_closure.append(pose.x())
        z_list_closure.append(pose.z())
        y_list_closure.append(pose.y())
        marginal_covariance = marginals_closure.marginalCovariance(key)
        location_uncertainty_closure.append(np.linalg.det(marginal_covariance[3:, 3:]))
        cov_closure.append(np.linalg.det(marginal_covariance))

    plt.plot(x_list_closure, z_list_closure, 'r.', label="with closure", markersize=0.9)

    # Plot estimated poses without loop closures in blue
    poses = gtsam.utilities.allPose3s(results_without_closure)
    marginals_no_closure = gtsam.Marginals(pose_graph_no_closure, results_without_closure)
    x_list = []
    z_list = []
    y_list = []
    cov_without_closure = []
    location_uncertainty = []
    for key in poses.keys():
        pose = poses.atPose3(key)
        x_list.append(pose.x())
        z_list.append(pose.z())
        y_list.append(pose.y())
        marginal_covariance = marginals_no_closure.marginalCovariance(key)
        location_uncertainty.append(np.linalg.det(marginal_covariance[3:, 3:]))
        cov_without_closure.append(np.linalg.det(marginal_covariance))
    plt.plot(x_list, z_list, 'b.', label="without closure", markersize=0.9)
    plt.title("pose graph with and without closure")
    plt.legend()

    # Plot absolute errors between estimated and ground truth trajectories
    plt.figure()
    diff_closure = [
        np.linalg.norm([x_list_closure[i] - truth_x[i], y_list_closure[i] - truth_y[i], z_list_closure[i] - truth_z[i]])
        for i in range(len(x_list_closure))
    ]
    diff_no_closure = [
        np.linalg.norm([x_list[i] - truth_x[i], y_list[i] - truth_y[i], z_list[i] - truth_z[i]])
        for i in range(len(x_list))
    ]
    plt.plot(diff_closure, 'r-', label="abs error with closure")
    plt.plot(diff_no_closure, 'b-', label="abs error without closure")
    plt.title("absolute error with and without closure")
    plt.legend()

    # Plot log determinants of covariance matrices
    plt.figure()
    cov_closure = np.array(cov_closure)
    cov_without_closure = np.array(cov_without_closure)

    log_closer = np.log(cov_closure)
    log_without_closure = np.log(cov_without_closure)

    log_uncertainty_closure = np.log(np.array(location_uncertainty_closure))
    log_uncertainty = np.log(np.array(location_uncertainty))

    plt.plot(log_closer, 'r-', label="with closure")
    plt.plot(log_without_closure, 'b-', label="without closure")
    plt.title("log det of covariance with and without closure")
    plt.legend()

    # Plot log of location uncertainties
    plt.figure()
    plt.plot(log_uncertainty_closure, 'r-', label="with closure")
    plt.plot(log_uncertainty, 'b-', label="without closure")
    plt.title("log location uncertainty with and without closure")
    plt.legend()

    plt.show()


def q_7_5(pose_graph, result, data):
    """
    Perform various plotting and analysis tasks on the pose graph with and without loop closures.

    :param pose_graph: Updated pose graph with loop closures.
    :param result: Optimized pose estimates with loop closures.
    :param data: Tuple containing the original pose graph and result without loop closures.
    """
    pose_graph_without_closure = data[0]
    result_without_closure = data[1]
    ground_truth_locations = get_locations_ground_truths()

    # Plot the pose graph along the process without loop closures
    plot_graph_along(0, pose_graph_without_closure, result_without_closure)
    plt.show()

    # Plot pose graphs with and without loop closures alongside ground truth
    plot_pose_graphs_XZ_ground_truth(result, result_without_closure, pose_graph, pose_graph_without_closure)
    plt.show()

    # Visualize feature matches between specific frames
    plot_matches(1488, 42)

    # Plot 2D trajectories compared to ground truth
    plot_trajectory2D_ground_truth(result_without_closure, "without closure, q7_5_4")
    plot_trajectory2D_ground_truth(result, "with closure, q7_5_4")

    # Plot absolute location errors
    q_7_5_5(result, result_without_closure)

    # Plot location uncertainty sizes
    marginals = gtsam.Marginals(pose_graph, result)
    q_7_5_6(marginals, result_without_closure, pose_graph_without_closure)


# init_dijksra_graph_relative_covariance_dict(data_list[1], data_list[0], relative_covariance_dict,
#                                             cov_dijkstra_graph)
# if __name__ == '__main__':
#
#     # Load the initial pose graph and result from a serialized file
#
#     db = get_tracking_database("/Users/mac/67604-SLAM-video-navigation/final_project/SIFT_DB")
#
#     key_frames = bundle.get_keyframes(db)
#     # pose_graph = PoseGraph()
#     # all_bundles = []
#     # for key_frame in key_frames:
#     #     first_frame = key_frame[0]
#     #     last_frame = key_frame[1]
#     #     graph, initial, cameras_dict, frames_dict = bundle.create_single_bundle(key_frame[0], key_frame[1], db)
#     #     graph, result = bundle.optimize_graph(graph, initial)
#     #     # print(f"bad point {result.atPoint3(bad_symbol)}")
#     #
#     #     bundle_dict = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
#     #                    'result': result, 'keyframes': key_frame}
#     #     all_bundles.append(bundle_dict)
#     #     pose_graph.add_bundle(bundle_dict)
#     #     print(f"Bundle {key_frame} added to the pose graph")
#     # save(all_bundles,"/Users/mac/67604-SLAM-video-navigation/final_project/SIFT_BUNDLES")
#     # pose_graph.save("/Users/mac/67604-SLAM-video-navigation/final_project/sift_p_graph_no_lc")
#     # pose_graph.optimize()
#     # pose_graph.save("/Users/mac/67604-SLAM-video-navigation/final_project/sift_p_graph_no_lc")
#     pose_graph = PoseGraph.load("/Users/mac/67604-SLAM-video-navigation/final_project/sift_p_graph_no_lc")
#
#     # Initialize the relative covariance dictionary and Dijkstra graph
#     data_list = [pose_graph.graph, pose_graph.result]
#
#     init_dijksra_graph_relative_covariance_dict(data_list[1], data_list[0], relative_covariance_dict,
#                                                 cov_dijkstra_graph)
#
#     pg, res = find_loops(data_list,db)
#     pose_graph.graph = pg
#     pose_graph.result = res
#     pose_graph.save("/Users/mac/67604-SLAM-video-navigation/final_project/sift_p_graph_with_LC")
#     # save((pg, res, relative_covariance_dict, cov_dijkstra_graph), "updated_pose_graph_results")
#
#     # Load the updated pose graph results with loop closures
#     # pg, res, relative_covariance_dict, cov_dijkstra_graph = load("updated_pose_graph_results")
#
#     # Perform analysis and plotting on the pose graph
#     # data_list = load(path)
#     # q_7_5(pg, res, data_list)
#     #
#     # # Plot the final updated pose graph trajectory
#     # plot_trajectory(0, res, title="updated pose graph", scale=1)
#
#     # Display all plots
#     plt.show()
