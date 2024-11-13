import os
import gtsam
import numpy as np
import cv2

import final_project.arguments as arguments

from final_project.backend.database.tracking_database import TrackingDB
from final_project.utils import K, M2, M1, P, Q, rodriguez_to_mat, read_extrinsic_matrices, visualize_track
from final_project.algorithms.triangulation import triangulate_last_frame, triangulate_links, \
    linear_least_squares_triangulation
from final_project.backend.GTSam.gtsam_utils import get_inverse, get_factor_point, get_factor_symbols
from final_project.arguments import *

BASE_LINE_SIGN = -1

TRANSFORMATIONS_FILE_NAME = 'transformations_ex3.npy'
POSE_SIGMA = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1, 1, 1, 1]) * 1)
K_OBJECT = gtsam.Cal3_S2Stereo(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], BASE_LINE_SIGN * M2[0, 3])


def get_relevant_tracks_in_keyframes(db, first_frame_idx, last_frame_idx):
    """get all the tracks that participate in more then one frame between the first and last frame."""
    all_tracks = set()
    frame_set = set(db.tracks(first_frame_idx))
    for i in range(first_frame_idx + 1, last_frame_idx + 1):
        current_frame_id = i
        current_frame_tracks = set(db.tracks(current_frame_id))
        intersection = frame_set.intersection(current_frame_tracks)
        all_tracks.update(intersection)
        frame_set = current_frame_tracks
    return all_tracks


def get_bundles_rel_Ts(db: TrackingDB, first_frame_idx, last_frame_idx):
    """Calculate the realtive transformations between the first and every frame between
        the first and the last frame."""
    transformations = dict()
    transformations[0] = np.vstack((M1, np.array([0, 0, 0, 1])))
    diff_coeff = np.zeros((5, 1))
    for i in range(1, last_frame_idx - first_frame_idx + 1):
        current_frame_id = i
        prev_frame_id = current_frame_id - 1

        # get the relevant tracks
        common_tracks = list(set(db.tracks(current_frame_id)).intersection(
            set(db.tracks(prev_frame_id))))
        links_prev_frame = [db.link(prev_frame_id, track_id) for track_id in common_tracks]
        links_curr_frame = [db.link(current_frame_id, track_id) for track_id in common_tracks]

        # calculate the transformation between the two frames
        # traingulate the links

        ############################################
        # NEW PART
        # CHANGE THE TRIANGULATION TO USE THE PREV FRAME AS THE REFERENCE
        triangulated_links = triangulate_last_frame(db, P, Q, links_prev_frame)
        ############################################
        if len(triangulated_links) < 4:
            raise Exception("Not enough points to triangulate and perform PnP")
        # calculate the transformation

        ############################################
        # NEW PART
        # CHANGE THE IMG_POINTS TO BE THE CURRENT FRAME POINTS
        img_points = np.array([(link.x_left, link.y) for link in links_curr_frame])
        ############################################
        success, rotation_vector, translation_vector = cv2.solvePnP(triangulated_links, img_points, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)

        T = rodriguez_to_mat(rotation_vector, translation_vector) if success else None
        if T is None:
            raise Exception("PnP failed")

        prev_frame_transform = transformations[prev_frame_id]
        transformations[current_frame_id] = T @ np.vstack((prev_frame_transform[:3, :], np.array([0, 0, 0, 1])))
    for camera_id in transformations.keys():
        transformations[camera_id] = transformations[camera_id][:3, :]
    return transformations


def create_single_bundle(first_frame_idx, last_frame_idx, db):
    """ Create a factor graph for the given frames."""
    gtsam_frames = dict()
    camera_symbols = dict()
    relevant_tracks = set()
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # calculate the relative transformations
    relative_transformations = get_bundles_rel_Ts(db, first_frame_idx, last_frame_idx)

    # add the cameras poses to the graph
    for i in range(last_frame_idx + 1 - first_frame_idx):
        frame_id = first_frame_idx + i
        pose_symbol = gtsam.symbol('c', frame_id)
        camera_pose = relative_transformations[i]

        # add the first camera pose prior
        if i == 0:
            pose = gtsam.Pose3()
            graph.add(gtsam.PriorFactorPose3(pose_symbol, pose, POSE_SIGMA))
            initial_estimate.insert(pose_symbol, pose)
            gtsam_frame = gtsam.StereoCamera(pose, K_OBJECT)
            gtsam_frames[frame_id] = gtsam_frame
            camera_symbols[frame_id] = pose_symbol
            continue

        else:
            last_tracks = set(db.tracks(frame_id - 1))
            this_frame_tracks = set(db.tracks(frame_id))
            intersection = last_tracks.intersection(this_frame_tracks)
            relevant_tracks.update(intersection)
            rotation = camera_pose[:3, :3]
            translation = camera_pose[:3, 3]
            word_to_cam_pose = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation))
            cam_to_word_pose = word_to_cam_pose.inverse()

            initial_estimate.insert(pose_symbol, cam_to_word_pose)

        # Create the stereo camera
        gtsam_frame = gtsam.StereoCamera(cam_to_word_pose, K_OBJECT)
        gtsam_frames[frame_id] = gtsam_frame
        camera_symbols[frame_id] = pose_symbol

    relevant_tracks = list(get_relevant_tracks_in_keyframes(db, first_frame_idx, last_frame_idx))

    # add the stereo factors to the graph
    for track_id in relevant_tracks:
        tracks_frames = sorted(db.frames(track_id), reverse=True)
        track_last_frame = min(db.last_frame_of_track(track_id), last_frame_idx)
        location_symbol = gtsam.symbol('l', track_id)
        triangulated_frame_id = None
        for frame_id in tracks_frames:
            if frame_id > last_frame_idx or frame_id < first_frame_idx:
                continue
            else:
                pose_symbol = camera_symbols[frame_id]
                link = db.link(frame_id, track_id)
                stereo_point2d = link.x_left, link.x_right, link.y
                assert stereo_point2d[0] > stereo_point2d[1]

                if frame_id == track_last_frame or frame_id == last_frame_idx:
                    # triangulate the point in the last frame and add measurement

                    gtsam_frame = gtsam_frames[frame_id]
                    reference_triangulated_point = gtsam_frame.backproject(
                        gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
                    # triangulated_point = triangulate_links([link],P,Q)[0]
                    # gtsam_frame.project(gtsam.Point3(triangulated_point[0]))
                    triangulated_frame_id = frame_id
                    if reference_triangulated_point[2] < 0:
                        print(f"Triangulated point negative Z, frame: {frame_id}, track: {track_id}, point: {reference_triangulated_point}")
                        break
                    assert reference_triangulated_point[2] > 0

                    initial_estimate.insert(location_symbol, reference_triangulated_point)

                # Create the factor
                noise = np.array([1, 1, 1]) + 1.5 * (abs(frame_id - triangulated_frame_id))
                sigma = gtsam.noiseModel.Diagonal.Sigmas(noise)
                # sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

                factor = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma,
                                                     pose_symbol,
                                                     location_symbol,
                                                     K_OBJECT)

                graph.add(factor)

    return graph, initial_estimate, camera_symbols, gtsam_frames


def get_negative_z_points(result, graph):
    """ Remove points with negative z values, from the graph and the result. """
    graph_size = graph.size()
    new_graph = gtsam.NonlinearFactorGraph()
    values_to_remove = []
    factors_to_remove = []
    for i in range(graph_size):
        factor = graph.at(i)
        if isinstance(factor, gtsam.GenericStereoFactor3D):
            point = get_factor_point(factor, result)
            if point[2] < 0 or point[2] > 1000:
                camera_symbol, track_symbol = get_factor_symbols(factor)
                values_to_remove.append(track_symbol)
                factors_to_remove.append(i)
            else:
                new_graph.add(factor)
        else:
            new_graph.add(factor)

    values_to_remove = list(set(values_to_remove))
    for value_to_remove in values_to_remove:
        result.erase(value_to_remove)
    return result, new_graph, len(values_to_remove) > 0


def optimize_graph(graph, initial):
    """ Optimize the graph using the Levenberg-Marquardt optimizer in iterative manner.
    Remove points with negative z values from the graph and the result after optimization until there are no such points."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    result, new_graph, removed = get_negative_z_points(result, graph)
    # if removed:
    #     print("the results keys before filtering", len(result.keys()))
    while removed:
        optimizer = gtsam.LevenbergMarquardtOptimizer(new_graph, result)
        result = optimizer.optimize()
        result, new_graph, removed = get_negative_z_points(result, new_graph)
    return new_graph, result


def calc_locations_angle(t1, t2):
    """ Calculate the angle between two cameras in the x-y plane for keyframe selection"""
    if t1.shape[0] != t1.shape[1]:
        t1 = np.vstack((t1, np.array([0, 0, 0, 1])))
    if t2.shape[0] != t2.shape[1]:
        t2 = np.vstack((t2, np.array([0, 0, 0, 1])))

    R = cv2.Rodrigues(t1[:3, :3] @ t2[:3, :3].T)[0]
    angle = np.linalg.norm(R) * 180 / np.pi

    return angle


def extract_keyframes(db: TrackingDB, transformations):
    """ Extract keyframes from the database"""
    keyFrames = []  # all keyframes start with zero
    frames = db.all_frames()
    num_frames = len(frames)
    i = 0
    minimum_gap = 5
    max_dist = 8.0
    track_losing_factor = 0.2
    max_gap = 21

    theta_max_traveled = 12
    theta_max_from_initial_pose = 10

    while i < num_frames - 1:
        t_init = transformations[i]
        old_tracks = set(db.tracks(i))
        start = min(i + 1, num_frames - 1)
        total_angle_diff = 0
        total_distance = 0
        prev_transform = transformations[i]

        for j in range(start, min(i + max_gap, num_frames)):
            next_transform = transformations[j]

            dist = calculate_distance_between_keyframes(prev_transform, next_transform)
            angle = calc_locations_angle(prev_transform, next_transform)
            angle_from_initial = calc_locations_angle(t_init, next_transform)

            new_tracks = set(db.tracks(j))
            common_tracks = old_tracks.intersection(new_tracks)
            tracks_ratio = len(common_tracks) / len(old_tracks)

            total_distance += dist
            total_angle_diff += angle

            # update parameters
            old_tracks = new_tracks
            prev_transform = next_transform

            # check if the minimal gap is reached
            if j < i + minimum_gap:
                continue

            # check conditions for keyframe selection
            ratio_cond = tracks_ratio < track_losing_factor
            max_gap_cond = j == i + max_gap - 1
            end_cond = j == num_frames - 1
            dist_cond = total_distance > max_dist
            angle_traveled_cond = angle > theta_max_traveled
            angle_from_initial_cond = angle_from_initial > theta_max_from_initial_pose

            if ratio_cond or max_gap_cond or end_cond or dist_cond or angle_traveled_cond or angle_from_initial_cond:
                keyFrames.append((i, j))
                i = j
                break

        if j == min(i + max_gap - 1, num_frames - 1):
            i = j + 1

    return keyFrames


def calculate_distance_between_keyframes(t1, t2):
    """ Calculate the distance in meters between two cameras for keyframe selection"""
    cam1_rotation = t1[:3, :3]
    cam1_translation = t1[:3, 3]
    cam1_position = -cam1_rotation.T @ cam1_translation

    cam2_rotation = t2[:3, :3]
    cam2_translation = t2[:3, 3]
    cam2_position = -cam2_rotation.T @ cam2_translation

    return np.linalg.norm(cam1_position - cam2_position)


def get_keyframes(db):
    t = read_extrinsic_matrices()
    keyframes_indices = extract_keyframes(db, t)
    return keyframes_indices


def get_all_bundles(db):
    bundles = dict()
    key_frames_poses = dict()
    cameras_matrix = [M1]
    t = read_extrinsic_matrices()
    keyframes_indices = extract_keyframes(db, t)

    key_frames_poses[0] = gtsam.Pose3()
    for i, key_frames in enumerate(keyframes_indices):
        graph, initial, cameras_dict, frames_dict = create_single_bundle(key_frames[0], key_frames[1], db)
        graph, result = optimize_graph(graph, initial)

        bundles[i] = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
                      'result': result, 'keyframes': key_frames}

        last_camera_symbol = cameras_dict[key_frames[1]]

        final_cam_trans = result.atPose3(last_camera_symbol).translation()
        final_cam_rot = result.atPose3(last_camera_symbol).rotation().matrix().T

        final_cam_trans = -final_cam_rot @ final_cam_trans

        final_matrix = np.hstack((final_cam_rot, final_cam_trans.reshape(-1, 1)))
        last_transformation = cameras_matrix[-1]
        last_transformation = np.vstack((last_transformation, np.array([0, 0, 0, 1])))
        global_transformation = final_matrix @ last_transformation
        global_transformation = global_transformation[:3, :]
        cameras_matrix.append(global_transformation)

    return bundles
