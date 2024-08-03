import pickle

from matplotlib import pyplot as plt

import arguments
import ex3
from ex2 import read_cameras
from tracking_database import TrackingDB
import ex5
import numpy as np
import gtsam
from gtsam.utils import plot
from ex3 import read_extrinsic_matrices
from ex5 import extract_keyframes, create_factor_graph, get_factor_point, get_factor_symbols, optimize_graph

# global variables:
# loading data
K, M1, M2 = read_cameras()
P, Q = K @ M1, K @ M2
LOCATION = 108
CAMERA = 99


# def  get_negative_z_points(result, graph, db):
#     keys_toremove = []
#     removed = False
#     new_graph = gtsam.NonlinearFactorGraph()
#     new_initial = gtsam.Values()
#
#     for i, key in enumerate(result.keys()):
#         if gtsam.symbolChr(key) == LOCATION:
#             point = result.atPoint3(key)
#             if point[2] < 0:
#                 keys_toremove.append(key)
#                 removed = True
#                 result.erase(key)
#             else:
#                 new_initial.insert(key, point)
#         elif gtsam.symbolChr(key) == CAMERA:
#             pose = result.atPose3(key)
#             new_initial.insert(key, pose)
#
#     for i in range(graph.size()):
#         factor = graph.at(i)
#         for key_to_remove in keys_toremove:
#             if key_to_remove not in factor.keys():
#                 new_graph.add(factor)
#                 # filtered_points.append(key)
#     # return filtered_points
#     if removed:
#         return result, graph, removed
#     else:
#         return new_initial, new_graph, removed


def get_graph_and_result(tracking_db):
    all_bundles = dict()
    all_graphs = []
    all_results = []
    key_frames_poses = dict()

    t = read_extrinsic_matrices()
    keyframes_indices = extract_keyframes(db, t)

    # inserting the first pose to the keyframes_poses
    key_frames_poses[0] = gtsam.Pose3()
    for i, key_frames in enumerate(keyframes_indices):
        graph, initial, cameras_dict, frames_dict = create_factor_graph(key_frames[0], key_frames[1], tracking_db)

        new_graph, result = optimize_graph(graph, initial)

        all_bundles[i] = {'graph': new_graph, 'initial': result, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
                      'result': result, 'keyframes': key_frames}

        all_graphs.append(new_graph)
        all_results.append(result)

    data = all_bundles, graphs, results
    save(data, arguments.BUNDLES_PATH)
    return all_bundles, all_graphs, all_results


def save(data, base_filename):
    """ save TrackingDB to base_filename+'.pkl' file. """
    if data is not None:
        filename = base_filename + '.pkl'
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        print('bundle saved to: ', filename)


def calculate_relative_pose_cov(first_frame_symbol, last_frame_symbol, bundle_graph, result):
    marginals = gtsam.Marginals(bundle_graph, result)

    # Obtain marginal covariances directly
    keys = gtsam.KeyVector()
    keys.append(first_frame_symbol)
    keys.append(last_frame_symbol)

    # # Calculate the marginal covariance
    # marginal_cov = marginals.jointMarginalCovariance(keys).fullMatrix()

    # calculating the pose
    first_frame_pose = result.atPose3(first_frame_symbol)
    last_frame_pose = result.atPose3(last_frame_symbol)
    relative_pose = first_frame_pose.between(last_frame_pose)

    # Compute the information matrix
    # information = np.linalg.inv(marginal_cov)
    information = marginals.jointMarginalInformation(keys).fullMatrix()
    print(marginals.jointMarginalInformation(keys).at(last_frame_symbol,last_frame_symbol))
    # Extract the relative covariance
    relative_cov = np.linalg.inv(information[-6:, -6:])
    return marginals, relative_pose, relative_cov


def q_6_1(bundle):
    graph = bundle['graph']
    result = bundle['result']
    first_frame_num, last_frame_num = bundle['keyframes']
    cameras = bundle['cameras_dict']
    first_cam_symbol = cameras[first_frame_num]
    last_cam_symbol = cameras[last_frame_num]

    marginals, relative_pose, relative_cov = calculate_relative_pose_cov(first_cam_symbol, last_cam_symbol,
                                                                         graph, result)

    # Plot the resulting frame locations as a 3D graph including the covariance of the locations.
    # (all the frames in the bundle, not just the 1st and last)

    plt.clf()
    # todo check if this is the right values of all frames and cov
    plot.plot_trajectory(fignum=1, values=result, marginals=marginals, title='trajectory_6_1', scale=1)

    plt.show()

    # print the relative pose and the cov associated with it
    print(f"Q 6.1 relative pose {relative_pose}")
    print(f"Q 6.1 relative pose {relative_cov}")

    return relative_pose, relative_cov


def q_6_2(bundles):
    # Initialize the factor graph
    pose_graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # graph_scale_factor is for the graph visualization
    graph_scale_factor = 0.001

    # Iterate through all bundles
    last_pose = None
    for i in range(len(bundles)):
        bundle = bundles[i]
        cameras_dict = bundle['cameras_dict']
        current_graph = bundle['graph']
        current_result = bundle['result']
        start_frame_symbol = cameras_dict[bundle['keyframes'][0]]
        end_frame_symbol = cameras_dict[bundle['keyframes'][1]]

        # extract the relative poses of the first and last cameras in the bundle
        first_frame_pose = current_result.atPose3(start_frame_symbol)
        last_frame_pose = current_result.atPose3(end_frame_symbol)
        relative_pose = first_frame_pose.between(last_frame_pose)

        # extract the relative covariance of the first and last cameras in the bundle
        marginals = gtsam.Marginals(current_graph, current_result)
        keys = gtsam.KeyVector()
        keys.append(start_frame_symbol)
        keys.append(end_frame_symbol)

        information = marginals.jointMarginalInformation(keys).fullMatrix()

        # Extract the relative covariance from the information matrix
        relative_cov = np.linalg.inv(information[-6:, -6:]) * graph_scale_factor

        # Add the relative pose factor to the pose graph
        noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov)

        if i == 0:
            # Add a prior factor to the first frame in the pose graph, and a measurement to the initial estimate
            first_frame_cov = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * graph_scale_factor)
            factor = gtsam.PriorFactorPose3(start_frame_symbol, first_frame_pose, first_frame_cov)
            pose_graph.add(factor)
            initial_estimate.insert(start_frame_symbol, first_frame_pose)
            last_pose = first_frame_pose

        # Add the relative pose factor to the pose graph
        factor = gtsam.BetweenFactorPose3(start_frame_symbol, end_frame_symbol, relative_pose, noise_model)
        pose_graph.add(factor)

        global_pose = last_pose.transformPoseFrom(relative_pose)
        last_pose = global_pose

        # Add initial poses to the initial estimate if not already added
        if not initial_estimate.exists(start_frame_symbol):
            initial_estimate.insert(start_frame_symbol, first_frame_pose)
        if not initial_estimate.exists(end_frame_symbol):
            initial_estimate.insert(end_frame_symbol, global_pose)

    # Optimize the factor graph
    optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph, initial_estimate)
    result = optimizer.optimize()

    # Plot the initial and final poses
    plot.plot_trajectory(1, initial_estimate, title="Initial Poses")
    plot.plot_trajectory(2, result, title="Optimized Poses")

    # Print the error before and after optimization
    initial_error = pose_graph.error(initial_estimate)
    final_error = pose_graph.error(result)

    print("Initial Error:", initial_error)
    print("Final Error:", final_error)

    # Plot the final trajectory with covariances
    marginals = gtsam.Marginals(pose_graph, result)
    plot.plot_trajectory(3, result, marginals=marginals, scale=1)

    return result, marginals


def load(base_filename):
    """ load TrackingDB to base_filename+'.pkl' file. """
    filename = base_filename + '.pkl'

    with open(filename, 'rb') as file:
        data = pickle.load(file)
        bundles, graphs, results = data
    print('Bundles loaded from', filename)
    return bundles, graphs, results


if __name__ == '__main__':
    db = TrackingDB()
    serialized_path = arguments.DATA_HEAD + "/docs/AKAZE/db/db_3359"
    db.load(serialized_path)
    # bundles, graphs, results = get_graph_and_result(db)
    bundles, graphs, results = load(arguments.BUNDLES_PATH)
    q_6_1(bundles[0])
    q_6_2(bundles)
    plt.show()
