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
from ex5 import extract_keyframes, create_graph
#global variables:
# loading data
K, M1, M2 = read_cameras()
P, Q = K @ M1, K @ M2


def get_graph_and_result(tracking_db, keyframe_indices):
    bundles = dict()
    graphs = []
    results = []
    key_frames_poses = dict()
    i = 0
    t = read_extrinsic_matrices()
    ts = np.array(t)
    keyframes_indices = extract_keyframes(db, t)
    #inserting the first pose to the keyframes_poses
    key_frames_poses[0] = gtsam.Pose3()
    for i, key_frames in enumerate(keyframes_indices):
        graph, initial, cameras_dict, frames_dict = create_graph(db, key_frames[0], key_frames[1], ts)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
        bundles[i] = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
                      'result': result, 'keyframes': key_frames}
        graphs.append(graph)
        results.append(result)
    return bundles, graphs, results


def calculate_relative_pose_cov(first_frame_symbol, last_frame_symbol, bundle_graph, result):


    marginals = gtsam.Marginals(bundle_graph, result)

    # Obtain marginal covariances directly
    keys = gtsam.KeyVector()
    keys.append(first_frame_symbol)
    keys.append(last_frame_symbol)
    marginal_cov = marginals.jointMarginalCovariance(keys).fullMatrix()

    #calculating the pose
    first_frame_pose = result.atPose3(first_frame_symbol)
    last_frame_pose = result.atPose3(last_frame_symbol)
    relative_pose = first_frame_pose.between(last_frame_pose)

    # Compute the information matrix
    information = np.linalg.inv(marginal_cov)

    # Extract the relative covariance
    relative_cov = np.linalg.inv(information[-6:, -6:])
    return marginals, relative_pose, relative_cov

def q_6_1(graph, result, first_frame_symbol, last_frame_symbol):
    Marginals,relative_pose, relative_cov = calculate_relative_pose_cov(first_frame_symbol,last_frame_symbol,
                                                                        graph, result)

    #Plot the resulting frame locations as a 3D graph including the covariance of the locations.
    #(all the frames in the bundle, not just the 1st and last)

    plt.clf()
    #todo check if this is the right values of all frames and cov
    plot.plot_trajectory(fignum=1, values=result, marginals=Marginals, title='trajectory_6_1', scale=1)

    plt.show()

    #print the relative pose and the cov associated with it
    print(relative_pose)
    print(relative_cov)

    return relative_pose, relative_cov

def get_frame_symbol(idx):
    return "c"+str(idx)

def q_6_2(keyframe_indices, kf_graphs, optimised_results,bundles):
    # Initialize the factor graph
    pose_graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Iterate through all bundles
    for i in range(len(kf_graphs)):
        try:
            print(f"Creating marginals{i}")
            marginals = gtsam.Marginals(kf_graphs[i], optimised_results[i])
        except:
            print(f"Marginals not created, {i}")
            continue

        bund = bundles[i]
        cameras_dict = bund['cameras_dict']
        keyframe = bund['keyframes']
        # result = bund['result']
        keyframe = keyframe_indices[i]
        # for i in range(keyframe_indices[i]):
        start_frame_symbol = cameras_dict[keyframe[0]]
        end_frame_symbol = cameras_dict[keyframe[1]]


        # Use the provided function to calculate relative pose and covariance
        # marginals, relative_pose, relative_cov = calculate_relative_pose_cov(start_frame_symbol, end_frame_symbol,
        #                                                                      kf_graphs[i],
        #                                                                      optimised_results[i])
        first_frame_pose = optimised_results[i].atPose3(start_frame_symbol)
        last_frame_pose = optimised_results[i].atPose3(end_frame_symbol)
        relative_pose = first_frame_pose.between(last_frame_pose)

        keys = gtsam.KeyVector()
        keys.append(start_frame_symbol)
        keys.append(end_frame_symbol)
        # marginals.marginalCovariance(cameras_dict[cam])
        marginal_cov = marginals.jointMarginalCovariance(keys).fullMatrix()

        information = np.linalg.inv(marginal_cov)

        # Extract the relative covariance
        relative_cov = np.linalg.inv(information[-6:, -6:])


        # Add the relative pose factor to the pose graph
        noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov)
        # noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        factor = gtsam.BetweenFactorPose3(start_frame_symbol, end_frame_symbol, relative_pose, noise_model)
        pose_graph.add(factor)

        # Add initial poses to the initial estimate if not already added
        if not initial_estimate.exists(start_frame_symbol):
            initial_estimate.insert(start_frame_symbol, optimised_results[i].atPose3(start_frame_symbol))
        if not initial_estimate.exists(end_frame_symbol):
            initial_estimate.insert(end_frame_symbol, optimised_results[i].atPose3(end_frame_symbol))

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

if __name__ == '__main__':
    db = TrackingDB()
    serialized_path = arguments.DATA_HEAD + "/docs/NEWEST_AKAZE"
    db.load(serialized_path)
    t = ex3.read_extrinsic_matrices()
    keyframe_indices = ex5.extract_keyframes(db, t)
    bundles, graphs, results = get_graph_and_result(db,keyframe_indices)
    print(len(bundles))
    first_bundle_graph = graphs[1]
    first_result = results[1]# change depending on the change in the ex5 - we want to only have the first bundle

    # t = ex5.calculate_transformations(db, 0,arguments.LEN_DATA)
    #todo this is the true transformation the previous one is the one we use

    last_frame_symbol = get_frame_symbol(keyframe_indices[1])
    first_frame_symbol = get_frame_symbol(0)
    bundel_0 = bundles[1]
    cameras = bundel_0['cameras_dict']
    first_cam_symbol = cameras[4]
    last_cam_symbol = cameras[keyframe_indices[1][1]]
    q_6_1(first_bundle_graph,first_result,first_cam_symbol, last_cam_symbol)

    q_6_2(keyframe_indices,graphs,results, bundles)





