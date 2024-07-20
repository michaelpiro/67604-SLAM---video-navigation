import os
import gtsam
import matplotlib.pyplot as plt
import numpy
import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Sequence, Optional
from timeit import default_timer as timer
from tracking_database import TrackingDB, Link, MatchLocation
from tqdm import tqdm
import random
from tracking_database import TrackingDB, Link, MatchLocation
from ex4 import K, M2, M1, P, Q, triangulate_last_frame
from ex3 import rodriguez_to_mat

BASE_LINE_SIGN = -1

FEATURE = cv2.AKAZE_create()
FEATURE.setNOctaves(2)
FEATURE.setThreshold(0.005)
MATCHER = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
# DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'
DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))
TRANSFORMATIONS_PATH = 'transformations_ex3.npy'

# k_object = gtsam.Cal3_S2(718.856, 718.856, 0, 607.1928, 185.2157, 0.53716572)
k_object = gtsam.Cal3_S2Stereo(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], BASE_LINE_SIGN * M2[0, 3])


def q_5_1(tracking_db: TrackingDB):
    def find_random_track_of_length(tracking_db: TrackingDB, length: int) -> Optional[int]:
        eligible_tracks = [trackId for trackId, frames in tracking_db.trackId_to_frames.items() if
                           len(frames) >= length]

        if not eligible_tracks:
            return None
        return random.choice(eligible_tracks)

    def read_kth_camera(k):
        filename = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt'
        with open(filename, 'r') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == k:
                    camera = line.strip()
                    break
        numbers = list(map(float, camera.split()))
        matrix = np.array(numbers).reshape(3, 4)
        return matrix

    def plot_reprojection_errors(reprojection_errors: Dict[int, Tuple[float, float]], factors_errors):
        frames_keys = sorted(reprojection_errors.keys())
        left_errors = [reprojection_errors[f][0] for f in frames_keys]
        right_errors = [reprojection_errors[f][1] for f in frames_keys]
        factor = [factors_errors[f] for f in frames_keys]

        plt.figure(figsize=(10, 6))
        plt.plot(frames_keys, left_errors, label='Left Camera')
        plt.plot(frames_keys, right_errors, label='Right Camera')
        plt.xlabel('distance from reference frame')
        plt.ylabel('projection Error')
        plt.title('projection Error vs track length')
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 6))
        plt.plot(frames_keys, factor, label='Factor error')
        plt.xlabel('distance from reference frame')
        plt.ylabel('Factor Error')
        plt.title('Factor Error vs track length')
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 6))
        plt.plot(left_errors, factor, label='Factor error')
        plt.xlabel('left camera error')
        plt.ylabel('Factor Error')
        plt.title('Factor Error as a function of left camera error')
        plt.legend()
        plt.grid(True)

    calculate_plot_reprojection_err(find_random_track_of_length, plot_reprojection_errors, tracking_db)


def calculate_plot_reprojection_err(find_random_track_of_length, plot_reprojection_errors, tracking_db):
    transformations = np.load(TRANSFORMATIONS_PATH)
    trackId = find_random_track_of_length(tracking_db, 10)
    frames = tracking_db.frames(trackId)
    track_last_frame = tracking_db.last_frame_of_track(trackId)
    link = tracking_db.link(track_last_frame, trackId)
    stereo_point2d = link.x_left, link.x_right, link.y
    calculated_left_camera_matrix = transformations[track_last_frame]
    # calculated_left_camera_matrix = read_kth_camera(track_last_frame)

    # Calculate the inverse rotation
    rotation = calculated_left_camera_matrix[:3, :3]
    translation = calculated_left_camera_matrix[:3, 3]
    inverse_rotation = rotation.T

    # Calculate the inverse translation
    inverse_translation = -inverse_rotation @ translation

    # Create the left camera pose
    left_cam_pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))

    # Create the stereo camera
    gtsam_frame = gtsam.StereoCamera(left_cam_pose, k_object)

    # triangulate the point in the last frame
    reference_triangulated_point = gtsam_frame.backproject(
        gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))

    l2_error = {}
    estimated_points = []
    sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
    sigma_6d = gtsam.noiseModel.Unit.Create(6)
    errors = {}

    # Create the symbol for the landmark
    l1 = gtsam.symbol('l', 1)

    # graph = gtsam.NonlinearFactorGraph()
    # c1 = gtsam.symbol('c', 1)
    # q1 = gtsam.symbol('q', 1)
    # pose_c1 = gtsam.Pose3()
    # loc_q1 = gtsam.Point3(0, 0, 0)
    # initialEstimate = gtsam.Values()
    # initialEstimate.insert(c1, pose_c1)
    # initialEstimate.insert(l1, loc_q1)
    for i in range(len(frames)):
        # extract the link x,y coordinates for the left and right camera
        frame = frames[i]
        link = tracking_db.link(frame, trackId)
        stereo_point2d_l = np.array([link.x_left, link.y])
        stereo_point2d_r = np.array([link.x_right, link.y])

        # Calculate the left camera pose
        calculated_left_camera_matrix = transformations[frame]
        rotation = calculated_left_camera_matrix[:3, :3]
        translation = calculated_left_camera_matrix[:3, 3]
        inverse_rotation = rotation.T

        # Calculate the inverse translation
        inverse_translation = -inverse_rotation @ translation

        # Create the left camera pose
        left_cam_pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))

        # Create the stereo camera
        gtsam_frame = gtsam.StereoCamera(left_cam_pose, k_object)

        # Triangulate the point in the last frame
        projected_point_estimated = gtsam_frame.project(reference_triangulated_point)

        # Estimate the the point feature coordinate in the left and right camera
        left_point_esti = np.array([projected_point_estimated.uL(), projected_point_estimated.v()])
        right_point_esti = np.array([projected_point_estimated.uR(), projected_point_estimated.v()])

        estimated_points.append((left_point_esti, right_point_esti))
        reprojection_error_left = np.linalg.norm(stereo_point2d_l - left_point_esti)
        reprojection_error_right = np.linalg.norm(stereo_point2d_r - right_point_esti)
        l2_error[int(track_last_frame - frame)] = (reprojection_error_left, reprojection_error_right)

        # Create the factor
        initialEstimate = gtsam.Values()
        c = gtsam.symbol('c', i)
        fac = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, c, l1,
                                          k_object)

        # insert the initial estimation of the left camera pose
        initialEstimate.insert(c, left_cam_pose)

        # insert the initial estimation of the triangulated point
        initialEstimate.insert(l1, gtsam.Point3(reference_triangulated_point))

        # Extract the error of the factor and store it
        err = fac.error(initialEstimate)
        errors[int(track_last_frame - frame)] = err

    # for i in range(len(frames)):
    #     initialEstimate = gtsam.Values()
    #     frame = frames[i]
    #     c = gtsam.symbol('c', i)
    #     q = gtsam.symbol('q', i)
    #     link = tracking_db.link(frame, trackId)
    #     stereo_point2d = link.x_left, link.x_right, link.y
    #     if i == len(frames) - 1:
    #         origin = gtsam.Pose3()
    #         graph.add(gtsam.PriorFactorPose3(c, origin, sigma_6d))
    #         # initialEstimate.insert(c, origin)
    #
    #         # graph.add(gtsam.PriorFactorPoint3(q, gtsam.Point3(0, 0, 0)))
    #     fac = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, c, q,
    #                                       k_object)
    #
    #     graph.add(fac)
    #     # print()
    #     calculated_left_camera_matrix = transformations[frame]
    #     rotation = calculated_left_camera_matrix[:3, :3]
    #     translation = calculated_left_camera_matrix[:3, 3]
    #     inverse_rotation = rotation.T
    #     inverse_translation = -inverse_rotation @ translation
    #     left_cam_pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))
    #     gtsam_frame = gtsam.StereoCamera(left_cam_pose, k_object)
    #     # triangulate the point in the last frame
    #     triangulated_point = gtsam_frame.backproject(
    #         gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
    #     initialEstimate.insert(c, left_cam_pose)
    #     initialEstimate.insert(q, triangulated_point)
    #     err = fac.error(initialEstimate)
    #     errors[frame] = err

    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    # print(fac.error(initialEstimate))
    # print(f"before optimization: {optimizer.error()}")
    # result = optimizer.optimize()
    # print(f"after optimization: {optimizer.error()}")
    # pose_c1 = result.atPose3(gtsam.symbol('c', 0))
    # loc_q1 = result.atPoint3(gtsam.symbol('q', 0))
    # print(f"pose_c1: {pose_c1}")
    # print(f"loc_q1: {loc_q1}")
    # # gtsam.utils.plot_3d_points
    # # error = graph.error(result)
    # print("Error of the graph:", error)

    plot_reprojection_errors(l2_error, errors)


# # Select keyframes based on a criterion (e.g., every 10 frames)
# # keyframe_indices = list(range(0, len(transformations), 10))
# keyframe_indices = extract_keyframes(tracking_db)
# # Initialize the factor graph and initial estimates
# graph = gtsam.NonlinearFactorGraph()
# initial_estimate = gtsam.Values()
#
# # Add camera poses and landmark estimates to the initial estimate
# for i, frame_idx in enumerate(keyframe_indices):
#     pose_symbol = symbol('c', i)
#     camera_matrix = transformations[frame_idx]
#     rotation = camera_matrix[:3, :3]
#     translation = camera_matrix[:3, 3]
#     inverse_rotation = rotation.T
#     inverse_translation = -inverse_rotation @ translation
#     pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))
#     initial_estimate.insert(pose_symbol, pose)
#
# # Add reprojection factors
# for track_id in db.trackId_to_frames:
#     frames = db.trackId_to_frames[track_id]
#     for frame in frames:
#         if frame in keyframe_indices:
#             frame_idx = keyframe_indices.index(frame)
#             pose_symbol = symbol('c', frame_idx)
#             link = db.link(frame, track_id)
#             stereo_point2d = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
#             graph.add(gtsam.GenericStereoFactor3D(stereo_point2d, gtsam.noiseModel.Isotropic.Sigma(3, 1.0), pose_symbol,
#                                                   symbol('p', track_id), K))


def get_relevant_tracks_in_keyframes(db, first_frame_idx, last_frame_idx):
    tracks = []
    all_tracks = set()
    frame_set = set(range(first_frame_idx, last_frame_idx + 1))
    for i in range(first_frame_idx + 1, last_frame_idx):
        current_frame_id = i
        last_frame_idx = i - 1
        common_tracks = set(db.tracks(current_frame_id)).intersection(set(db.tracks(last_frame_idx)))
        all_tracks.update(common_tracks)
    for track_id in all_tracks:
        length = len(db.frames(track_id))
        intersect = len(set(db.frames(track_id)).intersection(frame_set))
        if length > 4 and intersect > 3:
            tracks.append(track_id)
    return tracks


def calculate_transformations(db: TrackingDB, first_frame_idx, last_frame_idx, last_frame_transform=M1):
    transformations = dict()
    transformations[first_frame_idx] = last_frame_transform
    for i in range(first_frame_idx, last_frame_idx):
        last_frame_id = i
        current_frame_id = i + 1
        common_tracks = set(db.tracks(last_frame_id)).intersection(
            set(db.tracks(current_frame_id)))
        links_last_frame = [db.link(last_frame_id, track_id) for track_id in common_tracks]
        links_current_frame = [db.link(current_frame_id, track_id) for track_id in common_tracks]

        # calculate the transformation between the two frames
        # traingulate the links
        triangulated_links = triangulate_last_frame(db, P, Q, links_current_frame)
        if len(triangulated_links) < 4:
            raise Exception("Not enough points to triangulate and perform PnP")
        # calculate the transformation
        diff_coeff = np.zeros((5, 1))
        links_last_frame = np.array(links_last_frame)
        img_points = np.array([(link.x_left, link.y) for link in links_last_frame])
        # links_current_frame = np.array(links_current_frame)
        success, rotation_vector, translation_vector = cv2.solvePnP(triangulated_links, img_points, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)

        new_trans = rodriguez_to_mat(rotation_vector, translation_vector) if success else None
        if new_trans is None:
            raise Exception("PnP failed")

        last_transform = transformations[last_frame_id]
        global_transform = new_trans @ np.vstack((last_transform, np.array([0, 0, 0, 1])))
        transformations[current_frame_id] = global_transform
    return transformations


def create_factors_between_keyframes(graph, first_frame_idx, last_frame_idx, db, k_matrix, initial_estimate):
    # all_tracks_in_keyframes = set()
    gtsam_frames = dict()
    camera_symbols = dict()
    relevant_tracks = get_relevant_tracks_in_keyframes(db, first_frame_idx, last_frame_idx)
    transformations = calculate_transformations(db, first_frame_idx, last_frame_idx)
    print(f" first frame {first_frame_idx} last frame {last_frame_idx} relevant {len(relevant_tracks)}")
    for i in range(last_frame_idx + 1 - first_frame_idx):
        frame_id = first_frame_idx + i
        all_tracks_in_frame = db.tracks(frame_id)
        # all_tracks_in_keyframes.update(all_tracks_in_frame)
        pose_symbol = gtsam.symbol('c', frame_id)
        camera_matrix = transformations[frame_id]
        rotation = camera_matrix[:3, :3]
        translation = camera_matrix[:3, 3]
        inverse_rotation = rotation.T
        inverse_translation = -inverse_rotation @ translation
        pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))
        initial_estimate.insert(pose_symbol, pose)
        # Create the stereo camera
        gtsam_frame = gtsam.StereoCamera(pose, k_matrix)
        gtsam_frames[frame_id] = gtsam_frame
        camera_symbols[frame_id] = pose_symbol
        if i == 0:
            graph.add(gtsam.PriorFactorPose3(pose_symbol, pose, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)))

    for track_id in relevant_tracks:
        frames = db.frames(track_id)
        frames = sorted(frames)
        frames = [frame for frame in frames if first_frame_idx <= frame <= last_frame_idx]
        track_last_frame = frames[-1]
        link = db.link(track_last_frame, track_id)
        stereo_point2d = link.x_left, link.x_right, link.y

        # triangulate the point in the last frame
        # gtsam_last_frame = gtsam_frames[track_last_frame]
        # reference_triangulated_point = gtsam_last_frame.backproject(
        #     gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))


        # initial_estimate.insert(location_symbol, reference_triangulated_point)

        location_symbol = gtsam.symbol('l', track_id)
        for j in range(len(frames)):
            print(frames)
            frame = frames[j]
            link = db.link(frame, track_id)
            if j == 0:
                stereo_point2d = link.x_left, link.x_right, link.y
                #
                # # Create the stereo camera
                gt_frame_2 = gtsam_frames[frame]

                # # # Triangulate the point in the last frame
                projected_point_estimated = gt_frame_2.backproject(
                    gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))

                initial_estimate.insert(location_symbol, gtsam.Point3(projected_point_estimated))

            # Create the factor
            sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
            c = camera_symbols[frame]
            fac = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, c,
                                              location_symbol,
                                              k_object)

            graph.add(fac)

            # insert the initial estimation of the triangulated point


def extract_keyframes(db):
    keyframe_indices = list(range(0, 100, 10))
    return [1,5]
    return keyframe_indices


def create_graph(db, keyframe_indices):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    k_matrix = k_object
    create_factors_between_keyframes(graph, keyframe_indices[0], keyframe_indices[1], db, k_matrix, initial_estimate)
    return graph, initial_estimate


if __name__ == '__main__':
    print(M1)
    all_frames_serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB3000"
    serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB_all_after_changing the percent"
    # db = create_DB(path, LEN_DATA_SET)
    # db.serialize(serialized_db_path)
    db = TrackingDB()
    db.load(serialized_db_path)
    indices = extract_keyframes(db)
    graph, initial = create_graph(db, indices)
    # Run bundle adjustment
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    # Analyze errors
    initial_error = graph.error(initial)
    final_error = graph.error(result)
    num_factors = graph.size()

    print(f"Initial total error: {initial_error}")
    print(f"Final total error: {final_error}")
    print(f"Number of factors: {num_factors}")
    print(f"Average factor error before optimization: {initial_error / num_factors}")
    print(f"Average factor error after optimization: {final_error / num_factors}")

    # Plot the optimized trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    gtsam.utils.plot.plot_trajectory(result, indices, ax=ax)

    plt.show()
    # q_5_1(db)
    # plt.show()
