import math
from tqdm import tqdm
import gtsam
from gtsam.utils.plot import plot_trajectory
import numpy as np
from final_project.algorithms.ransac import ransac_pnp
from final_project.backend.GTSam.bundle import K_OBJECT, optimize_graph
from final_project.backend.GTSam.gtsam_utils import get_symbol
from final_project.backend.GTSam.pose_graph import PoseGraph, calculate_relative_pose_cov
from final_project.backend.database.tracking_database import TrackingDB
from final_project.backend.loop.graph import Graph
from final_project.algorithms.matching import MATCHER

# Threshold constants for Mahalanobis distance and inliers
MAHALANOBIS_THRESHOLD = 220
INLIERS_THRESHOLD = 120
FAR_MAHALANOBIS_THRESHOLD = MAHALANOBIS_THRESHOLD * 7
MAX_CANDIDATES = 15

KEY_FRAME_GAP = 10  # Minimum gap between key frames to consider for loop closure

# Symbols used for locations and cameras in the pose graph
LOCATION_SYMBOL = 'l'
CAMERA_SYMBOL = 'c'

# Dictionary to store relative covariances and initialize a Dijkstra graph for covariance paths
relative_covariance_dict = dict()
new_edges = []

cov_dijkstra_graph = Graph()


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

    # # Update the result with the new global pose
    # result.erase(end_frame_symbol)
    # result.insert(end_frame_symbol, global_pose)

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


def get_relative_covariance_along_path(index_list, marginals):
    """
    Compute the cumulative relative covariance along a path of camera frames.

    :param index_list: List of camera frame indices representing the path.
    :param marginals: Marginals object containing covariance information.
    :return: Cumulative covariance matrix.
    """
    cov = None
    if len(index_list) == 1:
        c = gtsam.symbol('c', index_list[0])
        cov = marginals.marginalCovariance(c)
        return cov
    for i in range(len(index_list) - 1):
        edge = (index_list[i], index_list[i + 1])
        if str(edge) in relative_covariance_dict:
            c_n_giving_c_i = relative_covariance_dict[str(edge)]
        else:
            c1 = gtsam.symbol('c', index_list[i])
            c2 = gtsam.symbol('c', index_list[i + 1])
            c_n_giving_c_i = get_relative_consecutive_covariance(c1, c2, marginals)
            relative_covariance_dict[edge] = c_n_giving_c_i
        if cov is None:
            cov = c_n_giving_c_i
        else:
            cov = cov + c_n_giving_c_i
    return cov


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
    c_n_giving_c_i = get_relative_covariance_along_path(cur_index_list, marginals)

    # Compute the relative information matrix
    # relative_information = np.linalg.inv(c_n_giving_c_i)

    # Generate symbols for the camera frames
    symbol_cn = get_symbol(index_list[c_n_idx])
    symbol_ci = get_symbol(index_list[c_i_idx])

    # Calculate the Mahalanobis distance between the two camera poses
    # mahalanobis_distance = calculate_mahalanobis_distance(symbol_cn, symbol_ci, result, relative_information)
    # return mahalanobis_distance

    gtsam_cov = gtsam.noiseModel.Gaussian.Covariance(c_n_giving_c_i)
    mahalanobis_distance = np.sqrt(
        2 * gtsam.BetweenFactorPose3(symbol_cn, symbol_ci, gtsam.Pose3(), gtsam_cov).error(result))
    # print(mahalanobis_distance)
    return mahalanobis_distance


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
            candidates.append((index_list[c_i_index], mahalanobis_distance))
            # print(
            #     f'added candidate {index_list[c_i_index]} with distance {mahalanobis_distance} to {index_list[c_n_index]}')
        elif mahalanobis_distance > FAR_MAHALANOBIS_THRESHOLD:
            # Skip further candidates if distance is too large
            c_i_index += 2

    candidates.sort(key=lambda x: x[1])
    candidates = candidates[:MAX_CANDIDATES]
    candidate_index = [c[0] for c in candidates]
    return candidate_index


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

        keys = gtsam.KeyVector()
        keys.append(c1)
        keys.append(c2)
        marginal_information = marginals.jointMarginalInformation(keys)
        inf_c2_giving_c1 = marginal_information.at(c2, c2)
        cov_c2_giving_c1 = np.linalg.inv(inf_c2_giving_c1)
        c_n_giving_c_i = cov_c2_giving_c1

        cov_dict[str((index_list[i], index_list[i + 1]))] = c_n_giving_c_i

        keys = gtsam.KeyVector()
        keys.append(c1)
        keys.append(c2)
        inf_c1_giving_c2 = marginal_information.at(c1, c1)
        cov_c1_giving_c2 = np.linalg.inv(inf_c1_giving_c2)
        c_i_giving_c_n = cov_c1_giving_c2

        cov_dict[str((index_list[i + 1], index_list[i]))] = c_i_giving_c_n

        # c_n_giving_c_i = get_relative_consecutive_covariance(c1, c2, marginals)
        cov_dict[index_list[i + 1]] = c_n_giving_c_i
        dijkstra_graph.add_edge(index_list[i], index_list[i + 1], c_n_giving_c_i)

    # Add the marginal covariance for the first camera frame
    cov_dict[index_list[0]] = marginals.marginalCovariance(gtsam.symbol('c', index_list[0]))
    return cov_dict, dijkstra_graph


def find_loops(pose_graph, db):
    """
    Detect and insert loop closures into the pose graph.

    :param data: Tuple containing pose graph and result.
    :return: Updated pose graph and result with loop closures.
    """
    pg = pose_graph.graph
    result = pose_graph.result
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
        # if index_list[c_n_index] not in loops_frames:
        #     continue
        good_candidates = get_good_candidates(c_n_index, marginals, result, index_list)
        if len(good_candidates) > 0:
            camera_number = index_list[c_n_index]
            if familiar_path:
                # If already in a familiar path, append candidates for later processing
                frames_in_familiar_path.append((camera_number, good_candidates))
                continue

            # Find the best candidate with sufficient inliers
            best_candidate, matches, rel_T_to_candidate = consensus_matches(camera_number, good_candidates, db)
            if best_candidate is not None:
                familiar_path = True
                # Insert the loop closure into the pose graph
                pg, result = insert_to_pose_graph(camera_number, best_candidate,
                                                  matches, pg, result, db, rel_T_to_candidate)
                print(f"loop detected between camera {camera_number} and camera {best_candidate}")
        else:
            if len(frames_in_familiar_path) > 0:
                # Process the accumulated familiar path frames in reverse order
                for camera_number, candidates in frames_in_familiar_path[::-1]:
                    best_candidate, matches, rel_T_to_candidate = consensus_matches(camera_number, candidates, db)
                    if best_candidate is not None:
                        # Insert the loop closure into the pose graph
                        pg, result = insert_to_pose_graph(camera_number, best_candidate,
                                                          matches, pg, result, db, rel_T_to_candidate)
                        print(
                            f"end of familiar segment, loop closure between camera {camera_number} "
                            f"and camera {best_candidate}"
                        )
                        break
                familiar_path = False
                frames_in_familiar_path = []
    pose_graph.graph = pg
    pose_graph.result = result
    return pose_graph, pg, result


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


def insert_to_pose_graph(camera_number, best_candidate, matches, pose_graph, result, db, rel_T_to_candidate):
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
    rel_pose_new, rel_cov_new = get_relative_pose_and_cov(camera_number, best_candidate, pose_graph, result, matches,
                                                          db, rel_T_to_candidate)
    # Add the edge to a list for later optimization
    global new_edges
    new_edges.append((camera_number, best_candidate))

    # Add the relative pose to the pose graph and optimize
    pose_graph, result = update_pose_graph(pose_graph, result, rel_pose_new, rel_cov_new, camera_number, best_candidate)

    # update the dictionary of relative cov between the two frames
    global relative_covariance_dict
    init_dijksra_graph_relative_covariance_dict(result, pose_graph, relative_covariance_dict, cov_dijkstra_graph)
    marginals = gtsam.Marginals(pose_graph, result)
    for edge in new_edges:
        c1_symbol = gtsam.symbol('c', edge[0])
        c2_symbol = gtsam.symbol('c', edge[1])
        rel_cov = get_relative_consecutive_covariance(c1_symbol, c2_symbol, marginals)
        relative_covariance_dict[(edge[0], edge[1])] = rel_cov
        relative_covariance_dict[(edge[1], edge[0])] = rel_cov
        cov_dijkstra_graph.add_edge(camera_number, best_candidate, rel_cov)

    return pose_graph, result


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
    T, best_matches_idx, best_inliers = ransac_pnp(matches_l_l, keyframe1_links, keyframe2_links, inliers_percent=40)

    # Return only the inlier matches
    try:
        percentage_inliers = best_inliers / len(matches_l_l)
    except Exception as e:
        # print(best_inliers)
        # print(matches_l_l)
        percentage_inliers = 0

    # print(f"Percentage of inliers: {percentage_inliers}")
    return [matches_l_l[i] for i in best_matches_idx], percentage_inliers, T


def create_bundle(first_frame_idx, second_frame_idx, bundle_graph, result, inliers, db: TrackingDB, rel_T):
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
    first_pose_sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    graph.add(gtsam.PriorFactorPose3(first_frame_symbol, first_pose, first_pose_sigma))
    initial_estimate.insert(first_frame_symbol, gtsam.Pose3())

    # Calculate the relative pose and covariance between the two frames
    marginals, relative_pose_second, relative_cov_second = (
        calculate_relative_pose_cov(first_frame_symbol, second_frame_symbol, bundle_graph, result)
    )

    # Insert the relative pose into the initial estimates
    initial_estimate.insert(second_frame_symbol, rel_T)

    # Add the relative pose factor to the graph
    noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov_second)

    # Create a factor between the two camera frames
    factor = gtsam.BetweenFactorPose3(first_frame_symbol, second_frame_symbol, rel_T, noise_model)
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
        factor = gtsam.GenericStereoFactor3D(
            gtsam.StereoPoint2(first_link.x_left, first_link.x_right, first_link.y),
            sigma, first_frame_symbol, location_symbol, K_OBJECT
        )

        graph.add(factor)

        # Triangulate the 3D point from the first frame's stereo measurements
        reference_triangulated_point = gtsam_frame_1.backproject(
            gtsam.StereoPoint2(first_link.x_left, first_link.x_right, first_link.y)
        )
        assert reference_triangulated_point[2] > 0  # Ensure point is in front of the camera

        # Insert the triangulated 3D point into the initial estimates
        initial_estimate.insert(location_symbol, reference_triangulated_point)
        x = factor.error(initial_estimate)
        # Add a stereo factor for the second frame
        second_link = second_frame_links[match.trainIdx]
        factor = gtsam.GenericStereoFactor3D(
            gtsam.StereoPoint2(second_link.x_left, second_link.x_right, second_link.y),
            sigma, second_frame_symbol, location_symbol, K_OBJECT
        )
        graph.add(factor)
        x = factor.error(initial_estimate)

    return graph, initial_estimate


def get_relative_pose_and_cov(first_frame_idx, second_frame_idx, bundle_graph, result, inliers, db: TrackingDB, rel_T):
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
    graph, initial_estimate = create_bundle(first_frame_idx, second_frame_idx, bundle_graph, result, inliers, db, rel_T)

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
    rel_T_to_candidate = None
    for candidate in candidates_index_lst:
        matches, percentage, rel_T_to_candidate = check_candidate_match(reference_key_frame, candidate, data_base)
        if len(matches) > INLIERS_THRESHOLD:
            print(f"cand: {candidate}, matches: {len(matches)} inliers percentage: {percentage}")
            best_candidate = candidate
            best_matches = matches
            return best_candidate, best_matches, rel_T_to_candidate

    # If no candidate meets the inliers threshold, reset best_candidate
    if len(best_matches) > 0:
        if len(best_matches) < INLIERS_THRESHOLD:
            best_candidate = None
            best_matches = []
    return best_candidate, best_matches, rel_T_to_candidate
