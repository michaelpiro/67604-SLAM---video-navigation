import pickle
import random
from typing import Dict, List, Tuple

import gtsam
from matplotlib import pyplot as plt
from gtsam.utils.plot import plot_trajectory
from final_project import arguments
from final_project.algorithms.triangulation import linear_least_squares_triangulation
from final_project.backend.GTSam.pose_graph import PoseGraph
from final_project.backend.database.tracking_database import TrackingDB
from final_project.backend.GTSam.gtsam_utils import get_factor_symbols, calculate_global_transformation, save, \
    get_poses_from_gtsam, get_index_list, T_B_from_T_A, gtsam_pose_to_T, calculate_dist_traveled
from final_project.backend.GTSam.bundle import K_OBJECT, get_keyframes, create_single_bundle, optimize_graph
from final_project.backend.GTSam.gtsam_utils import load

import numpy as np

from final_project.backend.loop.graph import Graph
from final_project.backend.loop.loop_closure import find_loops, get_locations_ground_truths, \
    plot_trajectory2D_ground_truth, init_dijksra_graph_relative_covariance_dict, get_relative_covariance, \
    cov_dijkstra_graph

SUBSET_FACTOR = 0.05

####################################################################################################
# TRACKING ANALYSIS
####################################################################################################
from final_project.utils import K, M2, M1


def track_length(tracking_db: TrackingDB, trackId) -> int:
    return len(tracking_db.frames(trackId))


def total_number_of_tracks(tracking_db: TrackingDB) -> int:
    return tracking_db.track_num()


def number_of_frames(tracking_db: TrackingDB) -> int:
    return tracking_db.frame_num()


def mean_track_length(tracking_db: TrackingDB) -> float:
    track_ids = tracking_db.all_tracks()
    lengths = [track_length(tracking_db, trackId) for trackId in track_ids if
               track_length(tracking_db, trackId) > 1]
    return np.mean(lengths) if lengths else 0


def max_track_length(tracking_db: TrackingDB) -> int:
    track_ids = tracking_db.all_tracks()
    lengths = [track_length(tracking_db, trackId) for trackId in track_ids if
               track_length(tracking_db, trackId) > 1]
    return max(lengths) if lengths else 0


def min_track_length(tracking_db: TrackingDB) -> int:
    track_ids = tracking_db.all_tracks()
    lengths = [track_length(tracking_db, trackId) for trackId in track_ids if
               track_length(tracking_db, trackId) > 1]
    return min(lengths) if lengths else 0


def mean_number_of_frame_links(tracking_db: TrackingDB) -> float:
    if not tracking_db.frameId_to_trackIds_list:
        return 0
    return tracking_db.link_num() / tracking_db.frame_num()


def compute_outgoing_tracks(tracking_db: TrackingDB) -> Dict[int, int]:
    outgoing_tracks = {}
    for frameId in sorted(tracking_db.frameId_to_trackIds_list.keys()):
        next_frameId = frameId + 1
        if next_frameId in tracking_db.frameId_to_trackIds_list:
            current_tracks = set(tracking_db.frameId_to_trackIds_list[frameId])
            next_tracks = set(tracking_db.frameId_to_trackIds_list[next_frameId])
            outgoing_tracks[frameId] = len(current_tracks.intersection(next_tracks))
        else:
            outgoing_tracks[frameId] = 0
    return outgoing_tracks


def plot_connectivity_graph(outgoing_tracks: Dict[int, int]):
    frames = sorted(outgoing_tracks.keys())
    counts = [outgoing_tracks[frame] for frame in frames]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, counts)
    plt.xlabel('Frame ID')
    plt.ylabel('Number of Outgoing Tracks')
    plt.title('Connectivity Graph: Outgoing Tracks per Frame')
    plt.grid(True)


def compute_matches_count(tracking_db: TrackingDB) -> Dict[int, int]:
    matches_count = {}
    for frameId in tracking_db.all_frames():
        matches_count[frameId] = len(tracking_db.all_frame_links(frameId))
    return matches_count


def plot_matches_count_graph(matches_count: Dict[int, int]):
    frames = sorted(matches_count.keys())
    counts = [matches_count[frame] for frame in frames]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, counts)
    plt.xlabel('Frame ID')
    plt.ylabel('Number of Matches')
    plt.title('Matches Count Graph: Number of Matches per Frame')
    plt.grid(True)


def plot_inliers_percentage_graph(inliers_percentage_dict: Dict[int, float]):
    frames = sorted(inliers_percentage_dict.keys())
    percentages = [inliers_percentage_dict[frame] for frame in frames]

    plt.figure(figsize=(20, 10))
    plt.plot(frames, percentages)
    plt.xlabel('Frame ID')
    plt.ylabel('Percentage of Inliers')
    plt.title('Percentage of Inliers per Frame')
    plt.grid(True)


def calculate_track_lengths(tracking_db: TrackingDB) -> List[int]:
    track_lengths = [len(frames) for trackId, frames in tracking_db.trackId_to_frames.items()]
    return track_lengths


def plot_track_length_histogram(track_lengths: List[int]):
    plt.figure(figsize=(10, 6))
    plt.hist(track_lengths, bins=range(1, max(track_lengths) + 2), edgecolor='black')
    plt.xlabel('Track Length')
    plt.ylabel('Frequency')
    plt.title('Track Length Histogram')
    plt.grid(True)

    plt.yscale('log')
    plt.xlim((0, 150))


def plot_mean_factor_error(all_bundles):
    init_errors = []
    final_errors = []
    first_keyframes = []
    for bundle in all_bundles:
        result = bundle['result']
        graph = bundle['graph']
        initial_estimates = bundle['initial']
        keyframes = bundle['keyframes']

        first_keyframe = keyframes[0]
        num_factors = graph.size()

        init_errors.append(graph.error(initial_estimates) / num_factors)
        final_errors.append(graph.error(result) / num_factors)
        first_keyframes.append(first_keyframe)

    plt.figure(figsize=(10, 6))
    plt.plot(first_keyframes, np.log(np.array(init_errors)), label='Initial Error')
    plt.plot(first_keyframes, np.log(np.array(final_errors)), label='Final Error')
    plt.xlabel('First Keyframe ID')
    plt.ylabel('Log mean Factor Error')
    plt.title('Mean Factor Error vs. First Keyframe ID (Log Scale)')
    plt.legend()
    plt.grid(True)


def calculate_frame_projection_errors(db: TrackingDB, graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values,
                                      result: gtsam.Values,
                                      first_keyframe):
    init_frame_errors = []
    final_frame_errors = []
    num_factors = graph.size()
    kf_cam_symbol = gtsam.symbol('c', first_keyframe)
    init_stereo_cam = gtsam.StereoCamera(initial.atPose3(kf_cam_symbol), K_OBJECT)
    final_stereo_cam = gtsam.StereoCamera(result.atPose3(kf_cam_symbol), K_OBJECT)

    # tracks = db.tracks(first_keyframe)
    for i in range(num_factors):
        factor = graph.at(i)
        if isinstance(factor, gtsam.GenericStereoFactor3D):
            camera_symbol, location_symbol = get_factor_symbols(factor)
            camera_number = int(gtsam.DefaultKeyFormatter(camera_symbol)[1:])
            if camera_number == first_keyframe:
                measurement = factor.measured()
                location_init = initial.atPoint3(location_symbol)
                location_final = result.atPoint3(location_symbol)

                projected_point_to_init = init_stereo_cam.project(location_init)
                projected_left_cam_init = projected_point_to_init.uL(), projected_point_to_init.v()
                # projected_right_cam = projected_point.uR(), projected_point.v()

                projected_point_to_final = final_stereo_cam.project(location_final)
                projected_left_cam_final = projected_point_to_final.uL(), projected_point_to_final.v()
                # projected_right_cam = projected_point.uR(), projected_point.v()

                measurement_left = measurement.uL(), measurement.v()
                # measurement_right = measurement.uR(), measurement.v()

                distance_left_init = round(
                    np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam_init)), 2)
                distance_left_final = round(
                    np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam_final)), 2)
                # distance_left = round(np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam)), 2)
                # distance_right = round(np.linalg.norm(np.array(measurement_right) - np.array(projected_right_cam)), 2)

                init_frame_errors.append(distance_left_init)
                final_frame_errors.append(distance_left_final)
    return init_frame_errors, final_frame_errors


def plot_median_projection_error(db, all_bundles):
    median_init_errors = []
    median_final_errors = []
    first_keyframes = []
    for bundle in all_bundles:
        result = bundle['result']
        graph = bundle['graph']
        initial_estimates = bundle['initial']
        keyframes = bundle['keyframes']

        first_keyframe = keyframes[0]
        init_frame_errors, final_frame_errors = calculate_frame_projection_errors(db, graph, initial_estimates, result,
                                                                                  first_keyframe)
        median_init_errors.append(np.median(init_frame_errors))
        median_final_errors.append(np.median(final_frame_errors))
        first_keyframes.append(first_keyframe)

    plt.figure(figsize=(10, 6))
    plt.plot(first_keyframes, np.log(np.array(median_init_errors)), label='Initial Error')
    plt.plot(first_keyframes, np.log(np.array(median_final_errors)), label='Final Error')
    plt.xlabel('Keyframes')
    plt.ylabel('Median Projection Error in log scale')
    plt.title('Median Projection Error vs. First Keyframes in log scale')
    plt.legend()
    plt.grid(True)


def map_camera_to_bundle(camera: int, keyframes):
    for i, keyframe in enumerate(keyframes):

        first_keyframe = keyframe[0]
        second_keyframe = keyframe[1]
        if first_keyframe <= camera <= second_keyframe:
            return i
    return None


def get_projection_errors(tracks, bundles, keyframes, db: TrackingDB):
    distance_dict = {}
    reprojection_erros_left = {}
    reprojection_erros_right = {}

    for trackId in tracks:
        frames = db.frames(trackId)
        track_first_frame = min(frames)
        for frame in frames:
            bundle_idx = map_camera_to_bundle(frame, keyframes)
            bundle = bundles[bundle_idx]
            result = bundle['result']
            point_symbol = gtsam.symbol('l', trackId)
            camera_symbol = gtsam.symbol('c', frame)
            try:
                point = result.atPoint3(point_symbol)
                camera = result.atPose3(camera_symbol)
            except Exception as e:
                continue
            # point = result.atPoint3(point_symbol)
            # camera = result.atPose3(camera_symbol)

            stereo_cam = gtsam.StereoCamera(camera, K_OBJECT)
            projected_point = stereo_cam.project(point)
            projected_left_cam = projected_point.uL(), projected_point.v()
            projected_right_cam = projected_point.uR(), projected_point.v()
            link = db.link(frame, trackId)

            distance_left = round(np.linalg.norm(np.array((link.x_left, link.y)) - np.array(projected_left_cam)), 2)
            distance_right = round(np.linalg.norm(np.array((link.x_right, link.y)) - np.array(projected_right_cam)), 2)

            frame_dist = int(frame - track_first_frame)
            if frame_dist not in reprojection_erros_left:
                reprojection_erros_left[frame_dist] = []
                reprojection_erros_right[frame_dist] = []
            reprojection_erros_left[frame_dist].append(distance_left)
            reprojection_erros_right[frame_dist].append(distance_right)

    dist_keys = list(reprojection_erros_left.keys())
    for key in dist_keys:
        distance_dict[key] = np.median(reprojection_erros_left[key]), np.median(reprojection_erros_right[key])
    return reprojection_erros_left, reprojection_erros_right


def plot_reprojection_error_vs_track_length(tracking_db: TrackingDB, bundles, keyframes):
    def get_tracks_subset(db: TrackingDB, subset_size: int):
        all_tracks = db.all_tracks()
        eligible_tracks = [track_Id for track_Id in all_tracks if track_length(db, track_Id) > 1]
        if not eligible_tracks:
            return None
        return np.random.choice(eligible_tracks, subset_size)

    def read_kth_camera(k):
        filename = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt'
        with open(filename, 'r') as file:
            for current_line_number, line in enumerate(file, start=0):
                if current_line_number == k:
                    camera = line.strip()
                    break
        numbers = list(map(float, camera.split()))
        matrix = np.array(numbers).reshape(3, 4)
        return matrix

    def plot_reprojection_errors(reprojection_errors: Dict[int, Tuple[float, float]], title=None):
        the_frames = sorted(reprojection_errors.keys())
        left_errors = [reprojection_errors[frame][0] for frame in the_frames]
        right_errors = [reprojection_errors[frame][1] for frame in the_frames]

        plt.figure(figsize=(10, 6))
        plt.plot(the_frames, left_errors, label='Left Camera')
        plt.plot(the_frames, right_errors, label='Right Camera')
        plt.xlabel('distance from reference frame')
        plt.ylabel('projection Error')
        plt.xlim(0, max(the_frames))
        plt.ylim(0, 8)
        if title:
            plt.title(title)
        else:
            plt.title('projection Error vs track length')
        plt.legend()
        plt.grid(True)

    num_tracks = int(tracking_db.track_num() * SUBSET_FACTOR)
    print(f'num_tracks: {num_tracks}')
    track_ids = get_tracks_subset(tracking_db, num_tracks)
    distance_dict = {}
    reprojection_erros_left = {}
    reprojection_erros_right = {}
    for trackId in track_ids:
        track_last_frame = tracking_db.last_frame_of_track(trackId)
        frames = tracking_db.frames(trackId)
        left_camera_mat = read_kth_camera(track_last_frame)
        link = tracking_db.link(track_last_frame, trackId)

        p = K @ left_camera_mat
        q = K @ M2 @ np.vstack((left_camera_mat, np.array([0, 0, 0, 1])))

        world_point = linear_least_squares_triangulation(p, q, (link.x_left, link.y), (link.x_right, link.y))
        world_point_4d = np.append(world_point, 1).reshape(4, 1)

        projections = {}

        for frameId in frames:
            left_camera_mat = read_kth_camera(frameId)
            p = K @ left_camera_mat
            q = K @ M2 @ np.vstack((left_camera_mat, np.array([0, 0, 0, 1])))
            projection_left, projection_right = ((p @ world_point_4d).T, (q @ world_point_4d).T)

            projection_left = projection_left / projection_left[:, 2][:, np.newaxis]
            projection_right = projection_right / projection_right[:, 2][:, np.newaxis]
            projections[frameId] = (projection_left, projection_right)
            link = tracking_db.link(frameId, trackId)
            points_vec_left = np.array([link.x_left, link.y, 1])
            points_vec_right = np.array([link.x_right, link.y, 1])
            frame_dist = int(track_last_frame - frameId)
            if frame_dist not in reprojection_erros_left:
                reprojection_erros_left[frame_dist] = []
                reprojection_erros_right[frame_dist] = []
            reprojection_erros_left[int(track_last_frame - frameId)].append(
                np.linalg.norm(points_vec_left - projection_left[0:2]))
            reprojection_erros_right[int(track_last_frame - frameId)].append(
                np.linalg.norm(points_vec_right - projection_right[0:2]))

    dist_keys = list(reprojection_erros_left.keys())
    for key in dist_keys:
        distance_dict[key] = np.median(reprojection_erros_left[key]), np.median(reprojection_erros_right[key])

    plot_reprojection_errors(distance_dict, 'PnP projection Error vs track length')
    # plt.savefig('reprojection_error_vs_track_length_1.png')

    rep_left, rep_right = get_projection_errors(track_ids, bundles, keyframes, tracking_db)
    dist_keys = list(rep_left.keys())
    for key in dist_keys:
        distance_dict[key] = np.median(rep_left[key]), np.median(rep_right[key])

    plot_reprojection_errors(distance_dict, 'Bundle Adjustment projection Error vs track length')
    # plt.savefig('reprojection_error_vs_track_length_2.png')


import cv2
import numpy as np

from final_project.Inputs import read_extrinsic_matrices
from final_project.arguments import LEN_DATA_SET, SIFT_DB_PATH, GROUND_TRUTH_PATH
from final_project.backend.GTSam.gtsam_utils import get_locations_from_gtsam
from final_project.backend.database.tracking_database import TrackingDB
from final_project.backend.GTSam.gtsam_utils import calculate_relative_transformation
import matplotlib.pyplot as plt


def calculate_camera_locations(camera_transformations):
    loc = np.array([0, 0, 0])
    for T in camera_transformations:
        R = T[:, :3]
        t = T[:, 3:]
        loc = np.vstack((loc, (-R.transpose() @ t).reshape((3))))
    return loc[1:, :]


def plot_trajectory_over_all(db: TrackingDB, result, result_without_closure):
    """
    plot the trajectory over the whole dataset, with respect ot ground truth, PNP, bundle adjustment and
    """
    # Retrieve transformations and calculate locations
    ground_truth_transformations = read_extrinsic_matrices(n=LEN_DATA_SET)
    PNP_transformations = calculate_global_transformation(db, 1, LEN_DATA_SET)
    pnp_transformations = np.array([PNP_transformations[i] for i in range(len(PNP_transformations))])
    # Calculate trajectories
    ground_truth_locations = calculate_camera_locations(ground_truth_transformations)
    PNP_locations = calculate_camera_locations(pnp_transformations)
    loop_closures_locations = get_locations_from_gtsam(result)
    bundle_adjustment_locations = get_locations_from_gtsam(result_without_closure)

    # Extract X and Z coordinates for each trajectory
    gt_x, gt_z = zip(*[(loc[0], loc[2]) for loc in ground_truth_locations])
    pnp_x, pnp_z = zip(*[(loc[0], loc[2]) for loc in PNP_locations])
    lc_x, lc_z = zip(*[(loc[0], loc[2]) for loc in loop_closures_locations])
    ba_x, ba_z = zip(*[(loc[0], loc[2]) for loc in bundle_adjustment_locations])

    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot each trajectory with a distinct color and label
    plt.scatter(gt_x, gt_z, color='red', marker='o', label='Ground Truth', s=2)
    plt.scatter(pnp_x, pnp_z, color='blue', marker='x', label='PNP Transformations', s=2)
    plt.scatter(lc_x, lc_z, color='green', marker='^', label='Loop Closures', s=2)
    plt.scatter(ba_x, ba_z, color='purple', marker='s', label='Bundle Adjustment', s=2)

    # Add labels, title, legend, and grid
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Relative Position of the Four Cameras (Top-Down View)')
    plt.legend(loc='upper right')  # Add legend for different trajectories
    plt.grid(True)
    plt.axis('equal')  # Ensure the aspect ratio is equal


def calculate_error_deg(estimated_Transformations, ground_truth_transformations):
    """
    calculate the error of the degrees(between estimations and ground truth)
    :param estimated_Transformations: estimated transformations
    :param ground_truth_transformations: ground truth transformations
    """
    assert len(estimated_Transformations) == len(ground_truth_transformations)
    errors = []
    for Ra, Rb in zip(estimated_Transformations, ground_truth_transformations):
        Ra = Ra[:3, :3]
        Rb = Rb[:3, :3]
        R = Ra.T @ Rb
        rvec, _ = cv2.Rodrigues(R)
        error = np.linalg.norm(rvec) * 180 / np.pi
        errors.append(error)
    return errors


def Absolute_PnP_estimation_error(db: TrackingDB, result, result_without_closure):
    """
    absolute PNP estimation error, includes x,y,z, and deg errors
    """
    # Retrieve transformations and calculate locations
    ground_truth_transformations = read_extrinsic_matrices(n=LEN_DATA_SET)

    PNP_transformations = calculate_global_transformation(db, 1, LEN_DATA_SET)

    # Calculate trajectories
    ground_truth_locations = calculate_camera_locations(ground_truth_transformations)
    pnp_transformation_list = [PNP_transformations[i] for i in range(len(PNP_transformations))]
    PNP_locations = calculate_camera_locations(pnp_transformation_list)

    # Extract X and Z coordinates for each trajectory
    gt_x, gt_y, gt_z = ground_truth_locations[:, 0], ground_truth_locations[:, 1], ground_truth_locations[:, 2]
    pnp_x, pnp_y, pnp_z = PNP_locations[:, 0], PNP_locations[:, 1], PNP_locations[:, 2]
    # X axis error, Y axis error, Z axis error, Total location error norm (m)
    x_error = np.abs(pnp_x - gt_x)
    y_error = np.abs(pnp_y - gt_y)
    z_error = np.abs(pnp_z - gt_z)

    norm_error = np.linalg.norm(ground_truth_locations - PNP_locations, axis=1)
    # Angle error (deg)
    angle_errors = calculate_error_deg(pnp_transformation_list, ground_truth_transformations)

    # Plotting
    errors = [x_error, y_error, z_error, norm_error, angle_errors]
    error_labels = ['X Error (m)', 'Y Error (m)', 'Z Error (m)', 'Total Location Error Norm (m)', 'Angle Error (deg)']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    frame_num = [i for i in range(LEN_DATA_SET)]
    plt.figure(figsize=(10, 6))
    for i in range(len(errors) - 1):
        plt.plot(frame_num, errors[i], color=colors[i], label=error_labels[i])

    # Adding labels and title
    plt.ylabel('Norm Error in meters')
    plt.title('Distance PnP Estimation Errors')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.figure(figsize=(10, 6))
    plt.plot(frame_num, errors[-1], color=colors[-1], label=error_labels[-1])
    # Adding labels and title
    plt.ylabel('Angle Error in degrees')
    plt.title('Angle PnP Estimation Errors')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')


def absolute_pg(result, result_without_closure):
    """
    Plot graphs of absolute location errors with and without loop closures.

    :param result: Pose estimates with loop closures.
    :param result_without_closure: Pose estimates without loop closures.
    """
    # Extract locations from both results
    locations = get_locations_from_gtsam(result)
    locations_without_closure = get_locations_from_gtsam(result_without_closure)

    poses_with_closure = get_poses_from_gtsam(result)
    poses_without_closure = get_poses_from_gtsam(result_without_closure)
    cameras = get_index_list(result)

    ground_truth_transformations = read_extrinsic_matrices(n=LEN_DATA_SET)
    relevant_ground_truth_transformations = [ground_truth_transformations[i] for i in cameras]

    # Calculate trajectories
    ground_truth_locations = calculate_camera_locations(ground_truth_transformations)

    # Calculate angles
    angles_with_closure = calculate_error_deg(poses_with_closure, relevant_ground_truth_transformations)
    angles_without_closure = calculate_error_deg(poses_without_closure, relevant_ground_truth_transformations)
    # Plot location errors with loop closure
    plt.figure()
    frames = cameras
    plt.plot(
        frames,
        [np.linalg.norm(locations[i] - ground_truth_locations[cameras[i]]) for i in range(len(locations))],
        color='r', label="L2 norm"
    )
    plt.plot(
        frames,
        [np.linalg.norm(locations[i][0] - ground_truth_locations[cameras[i]][0]) for i in range(len(locations))],
        color='b', label="x error"
    )
    plt.plot(
        frames,
        [np.linalg.norm(locations[i][1] - ground_truth_locations[cameras[i]][1]) for i in range(len(locations))],
        color='g', label="y error"
    )
    plt.plot(
        frames,
        [np.linalg.norm(locations[i][2] - ground_truth_locations[cameras[i]][2]) for i in range(len(locations))],
        color='y', label="z error"
    )

    plt.title("Absolute pose graph estimation errors with loop closure")
    plt.xlabel("Frame number")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(
        cameras, angles_with_closure,
        color='purple', label="angle error"
    )
    plt.title("Absolute pose graph estimation angle errors with loop closure")
    plt.xlabel("Frame number")
    plt.ylabel("Error (deg)")
    plt.legend()
    plt.grid()

    # Plot location errors without loop closure
    plt.figure()
    plt.plot(
        frames,
        [np.linalg.norm(locations_without_closure[i] - ground_truth_locations[cameras[i]]) for i in
         range(len(locations_without_closure))],
        color='r', label="L2 norm"
    )
    plt.plot(
        frames,
        [np.linalg.norm(locations_without_closure[i][0] - ground_truth_locations[cameras[i]][0]) for i in
         range(len(locations_without_closure))],
        color='b', label="x error"
    )
    plt.plot(
        frames,
        [np.linalg.norm(locations_without_closure[i][1] - ground_truth_locations[cameras[i]][1]) for i in
         range(len(locations_without_closure))],
        color='g', label="y error"
    )
    plt.plot(
        frames,
        [np.linalg.norm(locations_without_closure[i][2] - ground_truth_locations[cameras[i]][2]) for i in
         range(len(locations_without_closure))],
        color='y', label="z error"
    )
    plt.title("Absolute pose graph estimation errors without loop closure")
    plt.xlabel("Frame number")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(
        cameras, angles_without_closure,
        color='purple', label="angle error"
    )
    plt.title("Absolute pose graph estimation angle errors without loop closure")
    plt.xlabel("Frame number")
    plt.ylabel("Error (deg)")
    plt.legend()
    plt.grid()


def calculate_norm_error(estimated_transformtions, ground_truth_transformations):
    """claculate the norm error of every transformation between estimated and ground truth"""
    # Calculate trajectories
    ground_truth_locations = calculate_camera_locations(ground_truth_transformations)
    estimated_locations = calculate_camera_locations(estimated_transformtions)
    # X axis error, Y axis error, Z axis error, Total location error norm (m)
    norm_error = np.linalg.norm(np.array(ground_truth_locations) - np.array(estimated_locations), axis=1)
    return norm_error


def plot_relative_error_consequtive_kf(bundles: Dict, db: TrackingDB):
    """
    plot the relative error between each conseqtive kfs in ground truth, PNP, and bundle
    """

    def get_angle_error(pose):
        R = pose[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        return np.linalg.norm(rvec) * 180 / np.pi

    def get_location(pose):
        R = pose[:3, :3]
        t = pose[:3, 3]
        return (-R.transpose() @ t).reshape((3))

    def get_rel_T(c1, c2):
        if c1.shape[0] == 3:
            c1 = np.vstack((c1, np.array([0, 0, 0, 1])))
        if c2.shape[0] == 3:
            c2 = np.vstack((c2, np.array([0, 0, 0, 1])))
        return c2 @ np.linalg.inv(c1)

    gt = read_extrinsic_matrices(GROUND_TRUTH_PATH, LEN_DATA_SET)
    # pnp_transformations = calculate_global_transformation(db, 1, LEN_DATA_SET)
    # np.save("/Users/mac/67604-SLAM-video-navigation/final_project/pnp_global_transformations.npy", pnp_transformations)
    pnp_transformations = np.load("/Users/mac/67604-SLAM-video-navigation/final_project/pnp_global_transformations.npy")
    bundles_list = [bundles[i] for i in range(len(bundles))]
    gt_rel = list()
    bundle_rel = list()
    pnp_rel = list()
    x_axis = list()
    pnp_location_error = list()
    bundle_location_error = list()

    pnp_angle_error = list()
    bundle_angle_error = list()

    for bundle in bundles_list:
        result = bundle["result"]
        key_frames = bundle["keyframes"]
        first_frame = key_frames[0]
        last_frame = key_frames[-1]
        x_axis.append(last_frame)
        pose_last_kf = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", last_frame)))
        first_pose = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", first_frame)))
        # pose = pose_last_kf.between(result.atPose3(gtsam.symbol("c", first_frame)))
        # pose = result.atPose3(gtsam.symbol("c", first_frame)).between(pose_last_kf)
        # bundle_rel.append(gtsam_pose_to_T(pospose_last_kfe_last_kf))

        # est_bundle_displacement = T_B_from_T_A(first_pose, pose_last_kf)
        # gt_displacement = T_B_from_T_A(gt[first_frame], gt[last_frame])
        # est_pnp_displacement = T_B_from_T_A(pnp_transformations[first_frame], pnp_transformations[last_frame])
        # bundle_rel_error = T_B_from_T_A(gt_displacement, est_bundle_displacement)
        # pnp_rel_error = T_B_from_T_A(gt_displacement, est_pnp_displacement)

        est_bundle_displacement = get_rel_T(first_pose, pose_last_kf)
        gt_displacement = get_rel_T(gt[first_frame], gt[last_frame])
        est_pnp_displacement = get_rel_T(pnp_transformations[first_frame], pnp_transformations[last_frame])

        bundle_rel_error = get_rel_T(gt_displacement, est_bundle_displacement)
        pnp_rel_error = get_rel_T(gt_displacement, est_pnp_displacement)

        bundle_rel.append(bundle_rel_error)
        pnp_rel.append(pnp_rel_error)

        bundle_angle_error.append(get_angle_error(bundle_rel_error))
        pnp_angle_error.append(get_angle_error(pnp_rel_error))

        bundle_location_error.append(np.linalg.norm(get_location(bundle_rel_error)))
        pnp_location_error.append(np.linalg.norm(get_location(pnp_rel_error)))

        x = 1

    bundle_rel = np.array(bundle_rel)
    pnp_rel = np.array(pnp_rel)

    np.save("/Users/mac/67604-SLAM-video-navigation/final_project/bundle_rel.npy", bundle_rel)
    np.save("/Users/mac/67604-SLAM-video-navigation/final_project/pnp_rel.npy", pnp_rel)
    # pnp_norm_err = calculate_norm_error(pnp_rel, gt_rel)
    # bundle_norm_err = calculate_norm_error(bundle_rel, gt_rel)
    # pnp_deg_err = calculate_error_deg(pnp_rel, gt_rel)
    # bunle_deg_err = calculate_error_deg(bundle_rel, gt_rel)

    # Plotting
    # errors = [pnp_norm_err, bundle_norm_err, pnp_deg_err, bunle_deg_err]
    errors = [pnp_location_error, bundle_location_error, pnp_angle_error, bundle_angle_error]
    error_labels = ['PNP norm Error (m)', 'bundle norm Error (m)', 'PNP angle Error (deg)', 'bundle angle Error(deg)']
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # plt.bar(error_labels, errors, color=['blue', 'green', 'red', 'yellow'])
    plt.figure(figsize=(10, 6))
    # for i in range(len(errors)):
    plt.plot(x_axis, errors[0], color=colors[0], label=error_labels[0])
    plt.plot(x_axis, errors[1], color=colors[1], label=error_labels[1])

    # Adding labels and title
    plt.ylabel('Error Magnitude')
    plt.title('Relative pose estimation location error of every two consecutive keyframe, PNP and bundles')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.figure(figsize=(10, 6))
    # for i in range(len(errors)):
    plt.plot(x_axis, errors[2], color=colors[2], label=error_labels[2])
    plt.plot(x_axis, errors[3], color=colors[3], label=error_labels[3])

    # Adding labels and title
    plt.ylabel('Error Magnitude')
    plt.title('Relative pose estimation angle error of every two consecutive keyframe, PNP and bundles')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Show plot


def rel_pnp_seq_err(db):
    """
    plot the relative error between each conseqtive kfs in ground truth, PNP, and bundle
    """

    def get_angle_error(pose):
        R = pose[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        return np.linalg.norm(rvec) * 180 / np.pi

    def get_location(pose):
        R = pose[:3, :3]
        t = pose[:3, 3]
        return (-R.transpose() @ t).reshape((3))

    def get_rel_T(c1, c2):
        if c1.shape[0] == 3:
            c1 = np.vstack((c1, np.array([0, 0, 0, 1])))
        if c2.shape[0] == 3:
            c2 = np.vstack((c2, np.array([0, 0, 0, 1])))
        return c2 @ np.linalg.inv(c1)

    sub_section_length = [100, 400, 800]
    gt = read_extrinsic_matrices(GROUND_TRUTH_PATH, LEN_DATA_SET)
    # pnp_global_trans = calculate_relative_transformation(db, 1, LEN_DATA_SET)
    pnp_global_trans = np.load("/Users/mac/67604-SLAM-video-navigation/final_project/pnp_global_transformations.npy")
    # pnp_global_trans = [pnp_global_trans[i] for i in range(len(pnp_global_trans))]
    accumulate_distance = calculate_dist_traveled(gt)
    # plt.figure(figsize=(10, 6))
    all_norm_err = list()
    all_deg_err = list()
    all_x_axis = list()
    for j, length in enumerate(sub_section_length):
        # gt_rel = list()
        pnp_rel = list()
        x_axis = list()
        angles_errors = list()
        location_errors = list()
        for i in range(LEN_DATA_SET - length):
            first_frame = i
            last_frame = i + length
            if last_frame >= LEN_DATA_SET:
                break
            x_axis.append(first_frame)
            dist_traveled = accumulate_distance[last_frame] - accumulate_distance[first_frame]

            gt_displacement = get_rel_T(gt[first_frame], gt[last_frame])
            est_pnp_displacement = get_rel_T(pnp_global_trans[first_frame], pnp_global_trans[last_frame])

            pnp_rel_error = get_rel_T(gt_displacement, est_pnp_displacement)
            # pnp_rel.append(pnp_rel_error)

            pnp_angle_error = get_angle_error(pnp_rel_error) / dist_traveled
            pnp_location_error = np.linalg.norm(get_location(pnp_rel_error)) / dist_traveled
            angles_errors.append(pnp_angle_error)
            location_errors.append(pnp_location_error)

        all_x_axis.append(x_axis)
        all_norm_err.append(location_errors)
        all_deg_err.append(angles_errors)

    # Print the average error and median of all the subsequence lengths
    location_err_means = [np.mean(all_norm_err[i]) for i in range(len(sub_section_length))]
    angles_err_means = [np.mean(all_deg_err[i]) for i in range(len(sub_section_length))]
    location_err_medians = [np.median(all_norm_err[i]) for i in range(len(sub_section_length))]
    angles_err_medians = [np.median(all_deg_err[i]) for i in range(len(sub_section_length))]
    print("\nResults for relative PnP sequence error")
    for i in range(len(sub_section_length)):
        print(f"Mean of norm error for subsequence length {sub_section_length[i]} is {location_err_means[i]}, PnP")
        print(f"Mean of angle error for subsequence length {sub_section_length[i]} is {angles_err_means[i]}, PnP")
        print(f"Median of norm error for subsequence length {sub_section_length[i]} is {location_err_medians[i]}, PnP")
        print(f"Median of angle error for subsequence length {sub_section_length[i]} is {angles_err_medians[i]}, PnP")

    over_all_location_err_mean = np.mean(np.array(location_err_means))
    print(f"Overall mean of norm error is {over_all_location_err_mean}, PnP")
    over_all_angle_err_mean = np.mean(np.array(angles_err_means))
    print(f"Overall mean of angle error is {over_all_angle_err_mean}, PnP")
    # Plotting
    plt.figure(figsize=(10, 6))
    errors = all_norm_err
    error_labels = [f'PNP norm err of seq length {i}' for i in sub_section_length]
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i in range(len(errors)):
        plt.plot(all_x_axis[i], errors[i], color=colors[i], label=error_labels[i])
    plt.plot(all_x_axis[0], [over_all_location_err_mean] * len(all_x_axis[0]), color='black',
             label='Mean location error', linestyle='-.')

    # Adding labels and title
    plt.ylabel('Norm Error Magnitude')
    plt.title('Relative pose estimation location error as a function of subsequence length, PNP')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.figure(figsize=(10, 6))
    errors = all_deg_err
    error_labels = [f'PNP angle err of seq length {i}' for i in sub_section_length]

    for i in range(len(errors)):
        plt.plot(all_x_axis[i], errors[i], color=colors[i], label=error_labels[i])
    plt.plot(all_x_axis[0], [over_all_angle_err_mean] * len(all_x_axis[0]), color='black',
             label='Mean angle error', linestyle='-.')

    # Adding labels and title
    plt.ylabel('Error in degrees')
    plt.title('Relative pose estimation angle error as a function of subsequence length, PNP')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # plt.ylim(0, 1.5)
    plt.legend()


def get_closest_keyframe_index(key_frame_list, frame_index):
    """
    get the closest keyframe index to the given frame index
    """
    closest_index = min(key_frame_list, key=lambda x: abs(x - frame_index))
    return closest_index


def calculate_bundles_global_transformation(bundles_list: List):
    """Calculate the global transformation of every keyframe in the bundles list"""
    global_transformations = []
    for i in range(len(bundles_list)):
        bundle = bundles_list[i]
        result = bundle["result"]
        key_frames = bundle["keyframes"]
        first_frame = key_frames[0]
        last_frame = key_frames[-1]
        if i == 0:
            pose_first_kf = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", first_frame)))
            pose_last_kf = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", last_frame)))
            relative_trans = T_B_from_T_A(pose_first_kf, pose_last_kf)

            global_transformations.append(M1[:3, :])
            global_transformations.append(relative_trans[:3, :])
            # pose_last_kf = result.atPose3(gtsam.symbol("c", last_frame))
            # relative_trans = gtsam_pose_to_T(pose_last_kf)
            # global_trans = relative_trans @ np.vstack((global_transformations[-1], np.array([0, 0, 0, 1])))
            # global_transformations.append(global_trans)
            continue
        pose_first_kf = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", first_frame)))
        pose_last_kf = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", last_frame)))
        relative_trans = T_B_from_T_A(pose_first_kf, pose_last_kf)

        global_trans = relative_trans @ np.vstack((global_transformations[-1], np.array([0, 0, 0, 1])))
        global_transformations.append(global_trans[:3, :])
    return global_transformations


def rel_bundle_seq_err(bundles_list: List):
    def get_angle_error(pose):
        R = pose[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        return np.linalg.norm(rvec) * 180 / np.pi

    def get_location(pose):
        R = pose[:3, :3]
        t = pose[:3, 3]
        return (-R.transpose() @ t).reshape((3))

    def get_rel_T(c1, c2):
        if c1.shape[0] == 3:
            c1 = np.vstack((c1, np.array([0, 0, 0, 1])))
        if c2.shape[0] == 3:
            c2 = np.vstack((c2, np.array([0, 0, 0, 1])))
        return c2 @ np.linalg.inv(c1)

    sub_section_length = [100, 400, 800]
    gt = read_extrinsic_matrices(GROUND_TRUTH_PATH, LEN_DATA_SET)
    accumulate_distance = calculate_dist_traveled(gt)
    all_key_frames_list = [0] + [bundle["keyframes"][-1] for bundle in bundles_list]
    global_bundle_transformation = calculate_bundles_global_transformation(bundles_list)
    global_bundle_transformation = np.load("/Users/mac/67604-SLAM-video-navigation/final_project/bundles_global_t.npy")
    all_norm_err = list()
    all_deg_err = list()
    all_x_axis = list()
    for length in sub_section_length:
        gt_rel = list()
        bundle_rel = list()
        x_axis = list()
        angles_errors = list()
        location_errors = list()
        for i, bundle in enumerate(bundles_list):
            key_frames = bundle["keyframes"]
            first_frame = key_frames[0]
            if first_frame + length > LEN_DATA_SET:
                break
            x_axis.append(first_frame)
            closest_key_frame = get_closest_keyframe_index(all_key_frames_list, first_frame + length)
            closest_kf_index = all_key_frames_list.index(closest_key_frame)
            dist_traveled = accumulate_distance[closest_key_frame] - accumulate_distance[first_frame]

            first_frame_pose = global_bundle_transformation[i]
            last_frame_pose = global_bundle_transformation[closest_kf_index]
            est_bundle_displacement = get_rel_T(first_frame_pose, last_frame_pose)
            gt_displacement = get_rel_T(gt[first_frame], gt[closest_key_frame])

            bundle_rel_error = get_rel_T(gt_displacement, est_bundle_displacement)

            angle_error = get_angle_error(bundle_rel_error) / dist_traveled
            location_error = np.linalg.norm(get_location(bundle_rel_error)) / dist_traveled
            angles_errors.append(angle_error)
            location_errors.append(location_error)

        all_x_axis.append(x_axis)
        all_norm_err.append(location_errors)
        all_deg_err.append(angles_errors)

    # # Print the average error and median of all the subsequence lengths
    location_err_means = [np.mean(all_norm_err[i]) for i in range(len(sub_section_length))]
    angles_err_means = [np.mean(all_deg_err[i]) for i in range(len(sub_section_length))]
    location_err_medians = [np.median(all_norm_err[i]) for i in range(len(sub_section_length))]
    angles_err_medians = [np.median(all_deg_err[i]) for i in range(len(sub_section_length))]

    # print("\nResults for relative bundle sequence error")
    # for i in range(len(sub_section_length)):
    #     print(
    #         f"Mean of norm error for subsequence length {sub_section_length[i]} is {location_err_means[i]}, Bundle Adjusment")
    #     print(
    #         f"Mean of angle error for subsequence length {sub_section_length[i]} is {angles_err_means[i]}, Bundle Adjusment")
    #     print(
    #         f"Median of norm error for subsequence length {sub_section_length[i]} is {location_err_medians[i]}, Bundle Adjusment")
    #     print(
    #         f"Median of angle error for subsequence length {sub_section_length[i]} is {angles_err_medians[i]}, Bundle Adjusment")
    #
    over_all_location_err_mean = np.mean(np.array(location_err_means))
    print(f"Overall mean of norm error is {over_all_location_err_mean}, Bundle Adjusment")
    over_all_angle_err_mean = np.mean(np.array(angles_err_means))
    print(f"Overall mean of angle error is {over_all_angle_err_mean}, Bundle Adjusment")

    # Plotting
    plt.figure(figsize=(10, 6))
    errors = all_norm_err
    error_labels = [f'Bundle norm err of seq length {i}' for i in sub_section_length]
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i in range(len(errors)):
        plt.plot(all_x_axis[i], errors[i], color=colors[i], label=error_labels[i])
    plt.plot(all_x_axis[0], [over_all_location_err_mean] * len(all_x_axis[0]), color='black',
             label='Mean location error', linestyle='-.')

    # Adding labels and title
    plt.ylabel('Norm Error Magnitude')
    plt.title('Relative pose estimation location error as a function of subsequence length, Bundle')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.figure(figsize=(10, 6))
    errors = all_deg_err
    error_labels = [f'Bundle angle err of seq length {i}' for i in sub_section_length]

    for i in range(len(errors)):
        plt.plot(all_x_axis[i], errors[i], color=colors[i], label=error_labels[i])
    plt.plot(all_x_axis[0], [over_all_angle_err_mean] * len(all_x_axis[0]), color='black',
             label='Mean angle error', linestyle='-.')

    # Adding labels and title
    plt.ylabel('Error in degrees')
    plt.title('Relative pose estimation angle error as a function of subsequence length, Bundle')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # plt.ylim(0, 1.5)
    plt.legend()

    # # Plotting
    #
    # all_errors = [all_norm_err_bundle, all_norm_err_lc, all_norm_err_no_lc]
    # names = ['Bundle', 'Loop Closure', 'Without Loop Closure']
    # for j in range(len(all_errors)):
    #     errors = all_errors[j]
    #     plt.figure(figsize=(10, 6))
    #     error_labels = [f'{names[j]} norm error of seq length {k}' for k in sub_section_length]
    #     colors = ['blue', 'green', 'red', 'purple', 'orange']
    #
    #     for i in range(len(errors)):
    #         plt.plot(all_x_axis[i], errors[i], color=colors[i], label=error_labels[i])
    #
    #     location_err_means = [np.mean(errors[i]) for i in range(len(sub_section_length))]
    #     over_all_location_err_mean = np.mean(np.array(location_err_means))
    #     plt.plot(all_x_axis[0], [over_all_location_err_mean] * len(all_x_axis[0]), color='black',
    #              label='Mean location error', linestyle='-.')
    #     print(f"overall mean location error for {names[j]} is {over_all_location_err_mean}")
    #
    #     # Adding labels and title
    #     plt.ylabel('Norm Error Magnitude')
    #     plt.title(f'Relative pose estimation location error as a function of subsequence length, {names[j]}')
    #     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    #     plt.legend()
    #
    # all_errors = [all_deg_err_bundle, all_deg_err_lc, all_deg_err_no_lc]
    # for j in range(len(all_errors)):
    #     errors = all_errors[j]
    #     plt.figure(figsize=(10, 6))
    #     error_labels = [f'{names[j]} angle error of seq length {k}' for k in sub_section_length]
    #     # error_labels = [f'Bundles angle err of seq length {i}' for i in sub_section_length]
    #     for i in range(len(errors)):
    #         plt.plot(all_x_axis[i], errors[i], color=colors[i], label=error_labels[i])
    #     # plt.plot(all_x_axis[0], [over_all_angle_err_mean] * len(all_x_axis[0]), color='black', label='Mean angle error',
    #     #          linestyle='-.')
    #     angle_err_means = [np.mean(errors[i]) for i in range(len(sub_section_length))]
    #     over_all_angle_err_mean = np.mean(np.array(angle_err_means))
    #     plt.plot(all_x_axis[0], [over_all_angle_err_mean] * len(all_x_axis[0]), color='black',
    #              label='Mean angle error', linestyle='-.')
    #     print(f"overall mean angle error for {names[j]} is {over_all_angle_err_mean}")
    #
    #     # Adding labels and title
    #     plt.ylabel('Error in degrees')
    #     plt.title(f'Relative pose estimation angle error as a function of subsequence length, {names[j]}')
    #     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    #     plt.legend()


def calculate_uncertenties_loc_deg(pose_graph, result):
    """
    Plot graphs of location uncertainty sizes for the pose graph with and without loop closures.

    :param result: Pose estimates without loop closures.
    :param pose_graph: Pose graph without loop closures.
    """
    # Initialize covariance dictionary and Dijkstra graph for the pose graph without loop closures
    marginals = gtsam.Marginals(pose_graph, result)
    locations_uncertainties = []
    rotations_uncertienties = []
    poses = gtsam.utilities.allPose3s(result)
    for key in poses.keys():
        marginal_covariance = marginals.marginalCovariance(key)
        locations_uncertainties.append(np.linalg.det(marginal_covariance[3:, 3:]))
        rotations_uncertienties.append(np.linalg.det(marginal_covariance[:3, :3]))
    # # Calculate determinant of cumulative covariance for each camera frame without loop closures
    # for c_n in index_list[:]:
    #     # cur_index_list = dijkstra_graph.get_shortest_path(index_list[0], c_n)
    #     # rel_cov = get_relative_covariance(cur_index_list, marginals)
    #     locations_uncertainties.append(np.linalg.det(rel_cov[3:, 3:]))
    #     rotations_uncertienties.append(np.linalg.det(rel_cov[:3, :3]))

    return locations_uncertainties, rotations_uncertienties


def plot_loc_deg_uncertainties(pg_lc, pg_no_lc):
    """
    Plot location and rotation uncertainties for the pose graph with and without loop closures.

    :param pg_lc: Pose graph with loop closures.
    :param pg_no_lc: Pose graph without loop closures.
    """
    # Calculate uncertainties for the pose graph with and without loop closures
    loc_uncertainties_lc, rot_uncertainties_lc = calculate_uncertenties_loc_deg(pg_lc.graph, pg_lc.result)
    loc_uncertainties_no_lc, rot_uncertainties_no_lc = calculate_uncertenties_loc_deg(pg_no_lc.graph,
                                                                                      pg_no_lc.result)

    log_loc_uncertainties_lc = np.log(np.array(loc_uncertainties_lc))
    log_loc_uncertainties_no_lc = np.log(np.array(loc_uncertainties_no_lc))
    # log_rot_uncertainties_lc = np.log(np.array(rot_uncertainties_lc))
    log_rot_uncertainties_lc = np.array(rot_uncertainties_lc)
    # log_rot_uncertainties_no_lc = np.log(np.array(rot_uncertainties_no_lc))
    log_rot_uncertainties_no_lc = np.array(rot_uncertainties_no_lc)

    # Plot locations uncertainties with and without loop closures
    plt.figure()
    plt.plot(get_index_list(pg_lc.result), log_loc_uncertainties_lc,
             label='log Location uncertainty with loop closures', color='red')

    plt.plot(get_index_list(pg_no_lc.result), log_loc_uncertainties_no_lc,
             label='log Location uncertainty without loop closures', color='blue')

    # Plot locations uncertainties with and without loop closures
    plt.figure()
    plt.plot(get_index_list(pg_lc.result), log_rot_uncertainties_lc,
             label='log rotation uncertainty with loop closures', color='red')
    plt.plot(get_index_list(pg_no_lc.result), log_rot_uncertainties_no_lc,
             label='log rotation uncertainty without loop closures', color='blue')

    # # Plot location uncertainties
    # plt.figure(figsize=(10, 6))
    # plt.plot(loc_uncertainties_lc, label='Location uncertainty with loop closures')
    # plt.plot(loc_uncertainties_no_lc, label='Location uncertainty without loop closures')
    # plt.title('Location uncertainties for pose graph with and without loop closures')
    # plt.xlabel('Frame number')
    # plt.ylabel('Determinant of covariance matrix')
    # plt.legend()
    #
    # # Plot rotation uncertainties
    # plt.figure(figsize=(10, 6))
    # plt.plot(rot_uncertainties_lc, label='Rotation uncertainty with loop closures')
    # plt.plot(rot_uncertainties_no_lc, label='Rotation uncertainty without loop closures')
    # plt.title('Rotation uncertainties for pose graph with and without loop closures')
    # plt.xlabel('Frame number')
    # plt.ylabel('Determinant of covariance matrix')
    # plt.legend()


def load_bundles_list(base_filename):
    """ load TrackingDB to base_filename+'.pkl' file. """
    filename = base_filename + '.pkl'

    with open(filename, 'rb') as file:
        data = pickle.load(file)
        bundles = data
    print('Bundles loaded from', filename)
    return bundles


def run_analysis(db, bundles_list, pose_graph_lc, pose_graph_no_lc):
    bundles_dict = {}
    for i in range(len(bundles_list)):
        bundles_dict[i] = bundles_list[i]
    total_tracks = total_number_of_tracks(db)
    num_frames = number_of_frames(db)
    mean_length = mean_track_length(db)
    max_length = max_track_length(db)
    min_length = min_track_length(db)
    mean_frame_links = mean_number_of_frame_links(db)

    print(f"Total number of tracks: {total_tracks}")
    print(f"Total number of frames: {num_frames}")
    print(f"Mean track length: {mean_length}, Max track length: {max_length}, Min track length: {min_length}")
    print(f"Mean number of frame links: {mean_frame_links}")
    #
    # Compute outgoing tracks
    outgoing_tracks = compute_outgoing_tracks(db)

    # Plot the connectivity graph
    plot_connectivity_graph(outgoing_tracks)
    # plt.show()
    plt.savefig('connectivity_graph.png')
    # Compute matches count
    matches_count = compute_matches_count(db)

    # Plot the matches count graph
    plot_matches_count_graph(matches_count)
    # plt.show()
    plt.savefig('matches_count_graph.png')
    # Plot the inliers percentage graph
    plot_inliers_percentage_graph(db.frameID_to_inliers_percent)
    # plt.show()
    plt.savefig('inliers_percentage_graph.png')
    # Calculate track lengths
    track_lengths = calculate_track_lengths(db)

    # Plot the track length histogram
    plot_track_length_histogram(track_lengths)
    # plt.show()
    # plt.savefig('track_length_histogram.png')
    # Plot the reprojection error vs track length
    plot_reprojection_error_vs_track_length(db, bundles_dict, get_keyframes(db))
    # plt.show()

    # Plot mean factor error
    plot_mean_factor_error(bundles_list)
    # plt.show()
    # plt.savefig('mean_factor_error.png')

    # Plot median projection error
    plot_median_projection_error(db, bundles_list)
    # plt.show()

    # Plot trajectory overall
    plot_trajectory_over_all(db, pose_graph_lc.result, pose_graph_no_lc.result)
    # plt.show()

    # absolute_pnp_estimation_error
    result_with_lc = pose_graph_lc.result
    result_without_lc = pose_graph_no_lc.result
    Absolute_PnP_estimation_error(db, result_with_lc, result_without_lc)
    # plt.show()

    # relative error between consecutive keyframes, bundles and pnp
    plot_relative_error_consequtive_kf(bundles_dict, db)
    # plt.show()

    # relative Bundle estimation error over sub sections
    rel_bundle_seq_err(bundles_list)
    # plt.show()

    # relative PnP estimation error over sub sections
    rel_pnp_seq_err(db)
    # plt.show()

    # Plot location and rotation uncertainties for the pose graph with and without loop closures
    plot_loc_deg_uncertainties(pose_graph_lc, pose_graph_no_lc)

    plt.show()


if __name__ == '__main__':
    data_base = TrackingDB()
    data_base.load("/Users/mac/67604-SLAM-video-navigation/final_project/SIFT_DB")
    bundles = load_bundles_list("/Users/mac/67604-SLAM-video-navigation/final_project/AKAZE_DB")

    global_bundles_t = np.load("/Users/mac/67604-SLAM-video-navigation/final_project/bundles_global_t.npy")
    locations = calculate_camera_locations(global_bundles_t)
    plt.figure()
    plt.plot(locations[:, 0], locations[:, 2], label='Camera locations')
    plt.show()
    # bundles_global_t = np.save("/Users/mac/67604-SLAM-video-navigation/final_project/bundles_global_t.npy", global_bundles_t)
    rel_bundle_seq_err(bundles)
    plt.show()
    pose_graph_lc = PoseGraph.load("/Users/mac/67604-SLAM-video-navigation/final_project/sift_p_graph_with_LC")
    pose_graph_no_lc = PoseGraph.load("/Users/mac/67604-SLAM-video-navigation/final_project/sift_p_graph")

    run_analysis(data_base, bundles, pose_graph_lc, pose_graph_no_lc)
    plt.show()
    exit(0)
    # bundles_global_t = []
    # last_t = np.eye(4)
    # for bundle in bundles:
    #     result = bundle["result"]
    #     key_frames = bundle["keyframes"]
    #     first_frame = key_frames[0]
    #     last_frame = key_frames[-1]
    #     pose_last_kf = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", last_frame)))
    #     first_pose = gtsam_pose_to_T(result.atPose3(gtsam.symbol("c", first_frame)))
    #     est_bundle_displacement = T_B_from_T_A(first_pose, pose_last_kf)
    #
    #     global_mat = est_bundle_displacement @ last_t
    #     last_t = global_mat
    #     bundles_global_t.append(global_mat)
    # np.save("/Users/mac/67604-SLAM-video-navigation/final_project/bundles_global_t.npy", bundles_global_t)

    # x = get_keyframes(data_base)
    # print(x)
    # x = [
    #     (0, 11), (11, 18), (18, 24), (24, 29), (29, 41), (41, 50), (50, 57), (57, 64), (64, 71), (71, 76), (76, 81), (
    #         81, 88), (88, 93), (93, 102), (102, 111), (111, 116), (116, 121), (121, 132), (132, 141), (141, 149), (
    #         149, 156), (156, 163), (163, 169), (169, 175), (175, 181), (181, 188), (188, 195), (195, 202), (202, 209), (
    #         209, 216), (216, 223), (223, 229), (229, 235), (235, 241), (241, 247), (247, 252), (252, 257), (257, 262), (
    #         262, 270), (270, 278), (278, 286), (286, 294), (294, 299), (299, 307), (307, 316), (316, 327), (327, 334), (
    #         334, 341), (341, 352), (352, 360), (360, 367), (367, 373), (373, 379), (379, 385), (385, 391), (391, 396), (
    #         396, 403), (403, 410), (410, 417), (417, 426), (426, 438), (438, 462), (462, 486), (486, 493), (493, 499), (
    #         499, 506), (506, 514), (514, 521), (521, 527), (527, 532), (532, 537), (537, 542), (542, 547), (547, 552), (
    #         552, 557), (557, 562), (562, 567), (567, 572), (572, 577), (577, 582), (582, 587), (587, 592), (592, 597), (
    #         597, 602), (602, 608), (608, 614), (614, 621), (621, 629), (629, 640), (640, 648), (648, 654), (654, 659), (
    #         659, 664), (664, 672), (672, 677), (677, 684), (684, 689), (689, 694), (694, 699), (699, 705), (705, 711), (
    #         711, 717), (717, 723), (723, 729), (729, 736), (736, 744), (744, 753), (753, 762), (762, 770), (770, 777), (
    #         777, 783), (783, 788), (788, 793), (793, 798), (798, 803), (803, 808), (808, 813), (813, 818), (818, 824), (
    #         824, 829), (829, 836), (836, 844), (844, 854), (854, 861), (861, 866), (866, 871), (871, 876), (876, 881), (
    #         881, 886), (886, 891), (891, 896), (896, 901), (901, 907), (907, 912), (912, 917), (917, 922), (922, 927), (
    #         927, 933), (933, 938), (938, 944), (944, 949), (949, 954), (954, 959), (959, 964), (964, 969), (969, 975), (
    #         975, 980), (980, 985), (985, 990), (990, 995), (995, 1000), (1000, 1006), (1006, 1013), (1013, 1018), (
    #         1018, 1029), (1029, 1037), (1037, 1044), (1044, 1051), (1051, 1059), (1059, 1066), (1066, 1071),
    #     (1071, 1077), (
    #         1077, 1082), (1082, 1087), (1087, 1092), (1092, 1097), (1097, 1102), (1102, 1107), (1107, 1115),
    #     (1115, 1124), (
    #         1124, 1129), (1129, 1138), (1138, 1143), (1143, 1150), (1150, 1160), (1160, 1171), (1171, 1178),
    #     (1178, 1183), (
    #         1183, 1194), (1194, 1203), (1203, 1211), (1211, 1218), (1218, 1224), (1224, 1230), (1230, 1235),
    #     (1235, 1241), (
    #         1241, 1247), (1247, 1254), (1254, 1261), (1261, 1269), (1269, 1274), (1274, 1285), (1285, 1307),
    #     (1307, 1317), (
    #         1317, 1322), (1322, 1329), (1329, 1337), (1337, 1342), (1342, 1347), (1347, 1352), (1352, 1357),
    #     (1357, 1362), (
    #         1362, 1367), (1367, 1372), (1372, 1377), (1377, 1382), (1382, 1387), (1387, 1392), (1392, 1397),
    #     (1397, 1402), (
    #         1402, 1407), (1407, 1412), (1412, 1417), (1417, 1422), (1422, 1427), (1427, 1432), (1432, 1437),
    #     (1437, 1442), (
    #         1442, 1449), (1449, 1457), (1457, 1462), (1462, 1467), (1467, 1472), (1472, 1480), (1480, 1489),
    #     (1489, 1497), (
    #         1497, 1504), (1504, 1510), (1510, 1516), (1516, 1521), (1521, 1526), (1526, 1532), (1532, 1537),
    #     (1537, 1543), (
    #         1543, 1549), (1549, 1555), (1555, 1561), (1561, 1567), (1567, 1573), (1573, 1578), (1578, 1584),
    #     (1584, 1590), (
    #         1590, 1595), (1595, 1601), (1601, 1608), (1608, 1616), (1616, 1624), (1624, 1631), (1631, 1636),
    #     (1636, 1642), (
    #         1642, 1648), (1648, 1653), (1653, 1658), (1658, 1663), (1663, 1668), (1668, 1673), (1673, 1679),
    #     (1679, 1684), (
    #         1684, 1689), (1689, 1698), (1698, 1707), (1707, 1716), (1716, 1724), (1724, 1729), (1729, 1734),
    #     (1734, 1741), (
    #         1741, 1746), (1746, 1753), (1753, 1758), (1758, 1763), (1763, 1769), (1769, 1775), (1775, 1781),
    #     (1781, 1786), (
    #         1786, 1791), (1791, 1796), (1796, 1801), (1801, 1806), (1806, 1811), (1811, 1816), (1816, 1821),
    #     (1821, 1826), (
    #         1826, 1831), (1831, 1841), (1841, 1856), (1856, 1861), (1861, 1868), (1868, 1876), (1876, 1885),
    #     (1885, 1892), (
    #         1892, 1897), (1897, 1902), (1902, 1908), (1908, 1913), (1913, 1918), (1918, 1923), (1923, 1928),
    #     (1928, 1933), (
    #         1933, 1938), (1938, 1943), (1943, 1948), (1948, 1953), (1953, 1958), (1958, 1963), (1963, 1968),
    #     (1968, 1973), (
    #         1973, 1978), (1978, 1983), (1983, 1989), (1989, 1995), (1995, 2002), (2002, 2011), (2011, 2021),
    #     (2021, 2028), (
    #         2028, 2035), (2035, 2042), (2042, 2049), (2049, 2056), (2056, 2062), (2062, 2068), (2068, 2074),
    #     (2074, 2080), (
    #         2080, 2086), (2086, 2092), (2092, 2098), (2098, 2104), (2104, 2109), (2109, 2115), (2115, 2121),
    #     (2121, 2127), (
    #         2127, 2134), (2134, 2143), (2143, 2159), (2159, 2175), (2175, 2184), (2184, 2191), (2191, 2198),
    #     (2198, 2205), (
    #         2205, 2210), (2210, 2215), (2215, 2221), (2221, 2226), (2226, 2234), (2234, 2243), (2243, 2248),
    #     (2248, 2253), (
    #         2253, 2260), (2260, 2267), (2267, 2272), (2272, 2278), (2278, 2283), (2283, 2288), (2288, 2293),
    #     (2293, 2298), (
    #         2298, 2303), (2303, 2308), (2308, 2313), (2313, 2318), (2318, 2325), (2325, 2334), (2334, 2342),
    #     (2342, 2347), (
    #         2347, 2354), (2354, 2362), (2362, 2369), (2369, 2376), (2376, 2382), (2382, 2388), (2388, 2394),
    #     (2394, 2399), (
    #         2399, 2404), (2404, 2409), (2409, 2414), (2414, 2419), (2419, 2424), (2424, 2429), (2429, 2434),
    #     (2434, 2439), (
    #         2439, 2444), (2444, 2450), (2450, 2456), (2456, 2461), (2461, 2466), (2466, 2471), (2471, 2476),
    #     (2476, 2482), (
    #         2482, 2488), (2488, 2494), (2494, 2501), (2501, 2509), (2509, 2517), (2517, 2524), (2524, 2530),
    #     (2530, 2536), (
    #         2536, 2541), (2541, 2546), (2546, 2551), (2551, 2556), (2556, 2561), (2561, 2566), (2566, 2572),
    #     (2572, 2578), (
    #         2578, 2585), (2585, 2593), (2593, 2600), (2600, 2605), (2605, 2610), (2610, 2615), (2615, 2626),
    #     (2626, 2634), (
    #         2634, 2641), (2641, 2647), (2647, 2653), (2653, 2658), (2658, 2664), (2664, 2669), (2669, 2675),
    #     (2675, 2681), (
    #         2681, 2687), (2687, 2693), (2693, 2698), (2698, 2704), (2704, 2709), (2709, 2715), (2715, 2720),
    #     (2720, 2726), (
    #         2726, 2733), (2733, 2740), (2740, 2747), (2747, 2755), (2755, 2766), (2766, 2776), (2776, 2781),
    #     (2781, 2786), (
    #         2786, 2791), (2791, 2796), (2796, 2801), (2801, 2806), (2806, 2811), (2811, 2816), (2816, 2821),
    #     (2821, 2826), (
    #         2826, 2831), (2831, 2837), (2837, 2843), (2843, 2848), (2848, 2853), (2853, 2858), (2858, 2863),
    #     (2863, 2868), (
    #         2868, 2873), (2873, 2880), (2880, 2887), (2887, 2892), (2892, 2897), (2897, 2902), (2902, 2910),
    #     (2910, 2915), (
    #         2915, 2920), (2920, 2927), (2927, 2932), (2932, 2937), (2937, 2942), (2942, 2947), (2947, 2952),
    #     (2952, 2957), (
    #         2957, 2962), (2962, 2967), (2967, 2972), (2972, 2977), (2977, 2982), (2982, 2987), (2987, 2993),
    #     (2993, 3000), (
    #         3000, 3008), (3008, 3014), (3014, 3019), (3019, 3024), (3024, 3029), (3029, 3037), (3037, 3044),
    #     (3044, 3050), (
    #         3050, 3055), (3055, 3060), (3060, 3065), (3065, 3070), (3070, 3075), (3075, 3080), (3080, 3085),
    #     (3085, 3090), (
    #         3090, 3095), (3095, 3100), (3100, 3105), (3105, 3110), (3110, 3115), (3115, 3120), (3120, 3125),
    #     (3125, 3130), (
    #         3130, 3135), (3135, 3140), (3140, 3145), (3145, 3150), (3150, 3155), (3155, 3160), (3160, 3165),
    #     (3165, 3170), (
    #         3170, 3175), (3175, 3180), (3180, 3188), (3188, 3195), (3195, 3202), (3202, 3210), (3210, 3217),
    #     (3217, 3222), (
    #         3222, 3227), (3227, 3232), (3232, 3237), (3237, 3242), (3242, 3247), (3247, 3252), (3252, 3257),
    #     (3257, 3262), (
    #         3262, 3267), (3267, 3277), (3277, 3288), (3288, 3293), (3293, 3300), (3300, 3305), (3305, 3315),
    #     (3315, 3325), (
    #         3325, 3334), (3334, 3340), (3340, 3346), (3346, 3356), (3356, 3359)]
    # key_frames = [0]
    # for i in x:
    #     key_frames.append(i[1])
    # matrix = read_extrinsic_matrices()
    # matrix_pnp = np.load("/Users/mac/67604-SLAM-video-navigation/final_project/pnp_global_transformations.npy")
    # matrix = [matrix[i] for i in key_frames]
    # matrix_pnp = [matrix_pnp[i] for i in key_frames]
    # cameras = calculate_camera_locations(matrix)
    # cameras_pnp = calculate_camera_locations(matrix_pnp)
    # cameras_bundle = calculate_camera_locations(bundles_global_t)
    # plt.figure()
    # plt.plot(cameras[:, 0], cameras[:, 2], 'g', label='Ground Truth', linewidth=0.05, marker='o', markersize=0.3)
    # plt.plot(cameras_bundle[:, 0], cameras_bundle[:, 2], 'r', label='Bundle Adjustment', linewidth=0.05, marker='o',
    #          markersize=0.3)
    #
    # # add lines between camera i to bundle i
    # for i in range(len(cameras) - 1):
    #     plt.plot([cameras[i][0], cameras_bundle[i][0]], [cameras[i][2], cameras_bundle[i][2]], 'b', linewidth=0.1)
    #
    # plt.xlabel('X [m]')
    # plt.ylabel('Z [m]')
    # plt.title('ground truth vs bundle adjustment')
    # plt.legend()
    # plt.grid()
    # # plt.show(block=False)
    #
    # plt.figure()
    # plt.plot(cameras[:, 0], cameras[:, 2], 'g', label='Ground Truth', linewidth=0.05, marker='o', markersize=0.3)
    # plt.plot(cameras_pnp[:, 0], cameras_pnp[:, 2], 'r', label='PnP', linewidth=0.05, marker='o', markersize=0.3)
    #
    # # add lines between camera i to bundle i
    # for i in range(len(cameras) - 1):
    #     plt.plot([cameras[i][0], cameras_pnp[i][0]], [cameras[i][2], cameras_pnp[i][2]], 'b', linewidth=0.1)
    # plt.xlabel('X [m]')
    # plt.ylabel('Z [m]')
    # plt.title('ground truth vs PnP')
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    #
    # #
    #
    # # # plot_reprojection_error_vs_track_length(data_base, bundles, get_keyframes(data_base))
    # #
    # # # absolute_pg(pose_graph_lc.result,pose_graph_no_lc.result)
    # # # Absolute_PnP_estimation_error(data_base, pose_graph_lc.result, pose_graph_no_lc.result)
    # #
    # bundles_dict = {}
    # for i in range(len(bundles)):
    #     bundles_dict[i] = bundles[i]
    # plot_relative_error_consequtive_kf(bundles_dict, data_base)
    # #
    # bundle_rel = np.load("/Users/mac/67604-SLAM-video-navigation/final_project/bundle_rel.npy")
    # # pnp_rel = np.load("/Users/mac/67604-SLAM-video-navigation/final_project/pnp_rel.npy")
    # bundles_vectors = calculate_camera_locations(bundle_rel)
    # plt.figure()
    # plt.plot(cameras_bundle[:, 0], cameras_bundle[:, 2], 'r', label='Bundle Adjustment', linewidth=0.05, marker='o',
    #          markersize=0.3)
    # plt.plot(cameras[:, 0], cameras[:, 2], 'g', label='Ground Truth', linewidth=0.05, marker='o', markersize=0.3)
    #
    # # add lines between camera i to bundle i
    # for i in range(len(bundles_vectors)):
    #     # adds the vector to the point
    #     plt.plot([cameras_bundle[i][0], cameras_bundle[i][0] + bundles_vectors[i][0]],
    #              [cameras_bundle[i][2], cameras_bundle[i][2] + bundles_vectors[i][2]], 'b', linewidth=0.1)
    #
    # for i in range(len(cameras) - 1):
    #     plt.plot([cameras[i][0], cameras_bundle[i][0]], [cameras[i][2], cameras_bundle[i][2]], 'orange', linewidth=0.3)
    #
    # plt.xlabel('X [m]')
    # plt.ylabel('Z [m]')
    # plt.title('ground truth vs bundle adjustment')
    # plt.legend()
    # plt.grid()
    # plt.figure()
    # plt.plot(bundles_vectors[:, 0], bundles_vectors[:, 2], 'r', label='Bundle Adjustment', linewidth=0.05, marker='o',
    #          markersize=0.3)
    #
    #
    # # plot_reprojection_error_vs_track_length(data_base, bundles, get_keyframes(data_base))
    # plt.grid()
    # plt.title('Bundle Adjustment relative error')
    # plt.xlabel('X [m]')
    # plt.ylabel('Z [m]')
    # plt.show()
    #
    # # run_analysis(data_base, bundles, pose_graph_lc, pose_graph_no_lc)
    #
    # # key_frames = get_keyframes(db)
    # # #
    # # all_bundles = []
    # # pose_graph_lc = PoseGraph()
    # # pose_graph_no_lc = PoseGraph()
    # # for key_frame in key_frames:
    # #     first_frame = key_frame[0]
    # #     last_frame = key_frame[1]
    # #     graph, initial, cameras_dict, frames_dict = create_single_bundle(key_frame[0], key_frame[1], db)
    # #     graph, result = optimize_graph(graph, initial)
    # #
    # #     bundle_dict = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
    # #                    'result': result, 'keyframes': key_frame}
    # #     all_bundles.append(bundle_dict)
    # #     pose_graph_lc.add_bundle(bundle_dict)
    # #     pose_graph_no_lc.add_bundle(bundle_dict)
    # #     print(f"Bundle {key_frame} added to the pose graph")
    # # #
    # # # result = pose_graph.optimize()
    # # # pose_graph.save("/Users/mac/67604-SLAM-video-navigation/final_project/sift_p_graph")
